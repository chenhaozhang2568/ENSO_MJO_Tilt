# -*- coding: utf-8 -*-
"""
E:\Projects\ENSO_MJO_Tilt\src\02_mvEOF.py

Run:
cd /d E:\Projects\ENSO_MJO_Tilt
python src\02_mvEOF.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path


# ======================
# INPUTS (from Step2)
# ======================
OLR_BP_PATH = r"E:\Datas\ClimateIndex\processed\olr_bp_1979-2024.nc"
U_BP_PATH   = r"E:\Datas\ERA5\processed\era5_u850_u200_bp_1979-2024.nc"


# ======================
# OUTPUTS
# ======================
OUT_DIR = Path(r"E:\Datas\ClimateIndex\processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# TIME RANGE
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"  # later change to "2024-12-31"


# ======================
# PAPER-STYLE SETTINGS
# ======================
LAT_BAND = (-15.0, 15.0)                 # 15S–15N
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}     # Nov–Apr
TRACK_LON_MIN = 60.0
TRACK_LON_MAX = 180.0
MAX_WEST_JUMP_DEG = 5.0                  # allow westward back-jump up to 5 degrees


# ======================
# EVENT CRITERIA (paper-style operational)
# ======================
OLR_MIN_THRESH = -15.0       # reconstructed OLR threshold (negative = active)
MIN_EAST_DISP_DEG = 50.0     # propagation distance (deg)
AMP_THRESH = None            # optional amplitude screening; keep None to match paper-style logic here


# gates for "must pass IO and MC"
IO_GATE_LON = 90.0
MC_GATE_LON = 110.0


# DP tracking robustness (not changing science, only avoiding double-minimum jumping)
DP_JUMP_WEIGHT = 0.008       # 0.005~0.01
DP_NEAR_MIN_DELTA = 5.0      # keep candidate minima within daily_min + DELTA
DP_MAX_CANDS = 10


# 轨迹段允许的 "track 缺测" 间隔（通常只在你 full-period 边缘 NaN 才出现）
TRACK_GAP_DAYS = 2


# 一个事件段内，至少要有多少天满足强度阈值（-15）才算“真的有MJO活跃”
MIN_ACTIVE_DAYS_IN_EVENT = 5


# 强度判断用中心点附近的 OLR（避免 daily_min 的噪声与 RuntimeWarning）
ACTIVE_USE_CENTER_OLR = True


# ======================
# EOF 缓存优化
# ======================
# 若已存在 EOF 计算结果，可跳过 SVD 计算直接读取缓存
SKIP_EOF_IF_EXISTS = True
EOF_CACHE_NC = OUT_DIR / f"mjo_mvEOF_step3_1979-2022.nc"


# ======================
# UTILITIES
# ======================
def _open_ds(path: str) -> xr.Dataset:
    return xr.open_dataset(path, engine="netcdf4")


def _rename_latlon_if_needed(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    return ds


def _to_lon_180(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords:
        return ds
    lon = ds["lon"].values
    lon180 = ((lon + 180) % 360) - 180
    return ds.assign_coords(lon=lon180).sortby("lon")


def _sel_lat_band(ds: xr.Dataset, lat_band: tuple[float, float]) -> xr.Dataset:
    latmin, latmax = lat_band
    lat = ds["lat"].values
    if lat.size >= 2 and (lat[1] - lat[0]) < 0:  # decreasing
        return ds.sel(lat=slice(latmax, latmin))
    return ds.sel(lat=slice(latmin, latmax))


def _winter_mask(time: xr.DataArray) -> xr.DataArray:
    m = time.dt.month
    return xr.apply_ufunc(lambda x: np.isin(x, list(WINTER_MONTHS)), m)


def _standardize_by_space_time_std(x: np.ndarray) -> tuple[np.ndarray, float]:
    s = np.nanstd(x)
    if (not np.isfinite(s)) or s == 0:
        s = 1.0
    return x / s, float(s)


def _svd_eof(X: np.ndarray, nmode: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X: (T, P) float64, finite
    returns EOFs (P,nmode), PCs (T,nmode), svals (nmode,)
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    PCs = U[:, :nmode] * S[:nmode]
    EOFs = Vt[:nmode, :].T
    return EOFs, PCs, S[:nmode]


def _phase8(pc1: np.ndarray, pc2: np.ndarray) -> np.ndarray:
    ang = np.arctan2(pc2, pc1)
    ang = (ang + 2*np.pi) % (2*np.pi)

    ph = np.zeros_like(ang, dtype=np.int16)
    ok = np.isfinite(ang)
    ph[ok] = (np.floor(ang[ok] / (np.pi/4.0)).astype(np.int16) + 1)
    return ph


def _drop_nan_training_days(olr_tr: xr.DataArray, u850_tr: xr.DataArray, u200_tr: xr.DataArray):
    good = (
        np.isfinite(olr_tr).all(("lat", "lon"))
        & np.isfinite(u850_tr).all(("lat", "lon"))
        & np.isfinite(u200_tr).all(("lat", "lon"))
    ).fillna(False)

    # --- critical fix for dask boolean indexer ---
    if hasattr(good.data, "compute"):  # dask-backed
        good = good.compute()

    # use boolean indexer directly
    idx = np.nonzero(good.values)[0]  # numpy int index, 最稳
    olr_tr2  = olr_tr.isel(time=idx)
    u850_tr2 = u850_tr.isel(time=idx)
    u200_tr2 = u200_tr.isel(time=idx)
    return olr_tr2, u850_tr2, u200_tr2, good


def _local_minima_indices(y: np.ndarray) -> np.ndarray:
    n = y.size
    if n == 0:
        return np.array([], dtype=int)
    idx = []
    for i in range(n):
        if i == 0:
            if n == 1 or y[i] <= y[i+1]:
                idx.append(i)
        elif i == n - 1:
            if y[i] <= y[i-1]:
                idx.append(i)
        else:
            if y[i] <= y[i-1] and y[i] <= y[i+1]:
                idx.append(i)
    return np.array(idx, dtype=int)


def _track_center_with_candidates(
    recon: xr.DataArray,
    lon_min: float,
    lon_max: float,
    max_west_jump: float
) -> xr.DataArray:
    """
    DP tracking to obtain a continuous daily center trajectory from reconstructed OLR.

    recon: (time, lon) reconstructed OLR (lat-mean already applied)
    Window: lon_min..lon_max
    Rule: allow westward back-jump up to max_west_jump (deg); larger westward jump is forbidden.
    Objective: minimize sum_t [ OLR(t, lon_i) + DP_JUMP_WEIGHT*|jump| ]
    Candidate set each day: local minima near daily minimum (within DP_NEAR_MIN_DELTA), fallback to global min.

    Output: center_lon_track(time) (may be NaN only if a day is all-NaN in the window)
    """
    dom = recon.sel(lon=slice(lon_min, lon_max))
    lons = dom["lon"].values.astype(float)
    T = dom.sizes["time"]

    center = np.full(T, np.nan, dtype=float)

    t = 0
    while t < T:
        y0 = dom.isel(time=t).values.astype(float)
        if not np.isfinite(y0).any():
            center[t] = np.nan
            t += 1
            continue

        # continuous valid segment [t0, t1] with no all-NaN days
        t0 = t
        t1 = t
        while t1 + 1 < T:
            yy = dom.isel(time=t1 + 1).values.astype(float)
            if not np.isfinite(yy).any():
                break
            t1 += 1

        cand_lon_list: list[np.ndarray] = []
        cand_val_list: list[np.ndarray] = []

        for tt in range(t0, t1 + 1):
            yy = dom.isel(time=tt).values.astype(float)
            daily_min = float(np.nanmin(yy))

            cands = _local_minima_indices(yy)
            if cands.size == 0:
                cands = np.array([int(np.nanargmin(yy))], dtype=int)
            else:
                keep = yy[cands] <= (daily_min + DP_NEAR_MIN_DELTA)
                cands = cands[keep]
                if cands.size == 0:
                    cands = np.array([int(np.nanargmin(yy))], dtype=int)

            cands = cands[np.argsort(yy[cands])]
            cands = cands[:DP_MAX_CANDS]

            cand_lon_list.append(lons[cands].astype(float))
            cand_val_list.append(yy[cands].astype(float))

        # DP
        D = t1 - t0 + 1
        dp: list[np.ndarray] = []
        bp: list[np.ndarray] = []

        for d in range(D):
            K = len(cand_val_list[d])
            dp.append(np.full(K, np.inf, dtype=float))
            bp.append(np.full(K, -1, dtype=int))

        dp[0] = cand_val_list[0].copy()  # value_weight=1

        for d in range(1, D):
            vals = cand_val_list[d]
            lls = cand_lon_list[d]
            prev_vals = cand_val_list[d - 1]
            prev_lls = cand_lon_list[d - 1]

            for j in range(len(vals)):
                best_cost = np.inf
                best_k = -1
                for k in range(len(prev_vals)):
                    jump = lls[j] - prev_lls[k]
                    if jump < -max_west_jump:
                        continue
                    cost = dp[d - 1][k] + vals[j] + DP_JUMP_WEIGHT * abs(jump)
                    if cost < best_cost:
                        best_cost = cost
                        best_k = k
                dp[d][j] = best_cost
                bp[d][j] = best_k

        last = int(np.argmin(dp[-1]))
        path = [last]
        for d in range(D - 1, 0, -1):
            last = int(bp[d][last])
            if last < 0:
                last = 0
            path.append(last)
        path = path[::-1]

        for i_day, j in enumerate(path):
            center[t0 + i_day] = float(cand_lon_list[i_day][j])

        t = t1 + 1

    return xr.DataArray(center, coords={"time": dom["time"]}, dims=("time",), name="center_lon_track")


def _nan_stat_multi(da, name, idx_list=(0, 30, 60, 61, 1000, -1000, -61, -60, -1)):
    T = da.sizes.get("time", da.shape[0])
    print(f"\n[{name}] shape={da.shape}, T={T}")
    for i in idx_list:
        if i >= T or i < -T:
            print(f"  time_idx={i:>5}: skip (out of range)")
            continue
        x = da.isel(time=i).values
        print(f"  time_idx={i:>5}: any_nan={np.isnan(x).any()}, nan_count={np.isnan(x).sum()}, total={x.size}")


def _sample_olr_at_track_center(recon_latmean: xr.DataArray, center_track: np.ndarray) -> np.ndarray:
    """
    recon_latmean: (time, lon) 15S-15N mean reconstructed OLR
    center_track:  (time,) tracked lon (degE, same lon coords as recon_latmean)
    return olr_center[t]: OLR value at nearest lon grid to center_track[t]
    """
    lons = recon_latmean["lon"].values.astype(float)
    y2d = recon_latmean.values.astype(float)  # (time, lon)
    T = y2d.shape[0]

    out = np.full(T, np.nan, dtype=float)
    if T == 0:
        return out

    # lons should be sorted; use nearest index search
    for t in range(T):
        c = center_track[t]
        if not np.isfinite(c):
            continue
        j = int(np.argmin(np.abs(lons - float(c))))
        out[t] = y2d[t, j]

    return out


def _build_events_track_first(
    time: pd.DatetimeIndex,
    center_track: np.ndarray,
    olr_center: np.ndarray,
    amp: np.ndarray | None,
    winter_mask: np.ndarray
) -> pd.DataFrame:
    """
    改进A：事件骨架由连续轨迹 center_track 决定；-15 只用于事件段内“强度天数”计数。

    事件段定义（冬季内）：
      - center_track 连续有效（允许 TRACK_GAP_DAYS 个 NaN 间隔）
      - 相邻“有效轨迹点”间允许 west jump >= -MAX_WEST_JUMP_DEG（超过则切段）

    事件筛选：
      - disp = max(track) - min(track) >= MIN_EAST_DISP_DEG
      - gates: min(track) <= IO_GATE_LON and max(track) >= MC_GATE_LON
      - 段内满足 (olr_center <= OLR_MIN_THRESH) 的天数 >= MIN_ACTIVE_DAYS_IN_EVENT
      - （可选）AMP_THRESH 与 amp
    """
    n = len(time)
    if winter_mask.size != n:
        raise ValueError("winter_mask length mismatch")

    # 强度标记：只做“段内计数”，不用于连通性
    if ACTIVE_USE_CENTER_OLR:
        active_flag = np.isfinite(olr_center) & (olr_center <= OLR_MIN_THRESH)
    else:
        # fallback：如果你想继续用别的 active 定义，可以在这里替换
        active_flag = np.isfinite(olr_center) & (olr_center <= OLR_MIN_THRESH)

    # 冬季限制
    active_flag = active_flag & winter_mask.astype(bool)

    if AMP_THRESH is not None and amp is not None:
        active_flag = active_flag & np.isfinite(amp) & (amp >= AMP_THRESH)

    events = []
    eid = 0

    # diagnostics
    fail_disp = 0
    fail_gate = 0
    fail_active = 0
    fail_short = 0
    fail_west = 0

    in_seg = False
    seg_start = -1
    seg_end = -1
    nan_gap = 0

    prev_valid_lon = np.nan

    def finalize_segment(s0: int, s1: int):
        nonlocal eid, events
        nonlocal fail_disp, fail_gate, fail_active, fail_short

        if s0 < 0 or s1 < s0:
            return

        seg_lons = center_track[s0:s1+1].astype(float)
        seg_ok = np.isfinite(seg_lons)
        if not seg_ok.any():
            fail_short += 1
            return

        lons_valid = seg_lons[seg_ok]
        lon_min = float(np.nanmin(lons_valid))
        lon_max = float(np.nanmax(lons_valid))
        disp = lon_max - lon_min
        passed_gates = (lon_min <= IO_GATE_LON) and (lon_max >= MC_GATE_LON)

        # 强度天数计数（段内）
        seg_active = active_flag[s0:s1+1]
        n_active = int(np.nansum(seg_active.astype(int)))

        if n_active < MIN_ACTIVE_DAYS_IN_EVENT:
            fail_active += 1
            return

        if disp < MIN_EAST_DISP_DEG:
            fail_disp += 1
            return

        if not passed_gates:
            fail_gate += 1
            return

        eid += 1

        # 强度统计：用中心 OLR（更稳定）
        act_idxs = np.where(seg_active)[0] + s0
        seg_olr = olr_center[act_idxs] if act_idxs.size > 0 else np.array([np.nan])
        seg_amp = (amp[act_idxs] if (amp is not None and act_idxs.size > 0) else None)

        # lon_start/lon_end 用段内首尾“有效轨迹点”
        idxs = np.arange(s0, s1+1)
        valid_idxs = idxs[np.isfinite(center_track[s0:s1+1])]
        lon_start = float(center_track[valid_idxs[0]])
        lon_end = float(center_track[valid_idxs[-1]])

        events.append({
            "event_id": eid,
            "start_date": time[s0].strftime("%Y-%m-%d"),
            "end_date": time[s1].strftime("%Y-%m-%d"),
            "duration_days": int(s1 - s0 + 1),
            "lon_start": lon_start,
            "lon_end": lon_end,
            "east_displacement_deg": float(disp),
            "min_olr_min": float(np.nanmin(seg_olr)),
            "min_olr_mean": float(np.nanmean(seg_olr)),
            "amp_mean": float(np.nanmean(seg_amp)) if seg_amp is not None else np.nan,
            "active_days": int(n_active),
        })

    for t in range(n):
        if not winter_mask[t]:
            if in_seg:
                finalize_segment(seg_start, seg_end)
            in_seg = False
            seg_start = seg_end = -1
            nan_gap = 0
            prev_valid_lon = np.nan
            continue

        lon_t = center_track[t]
        if not np.isfinite(lon_t):
            if in_seg:
                nan_gap += 1
                if nan_gap > TRACK_GAP_DAYS:
                    finalize_segment(seg_start, seg_end)
                    in_seg = False
                    seg_start = seg_end = -1
                    nan_gap = 0
                    prev_valid_lon = np.nan
            continue

        lon_t = float(lon_t)

        if not in_seg:
            in_seg = True
            seg_start = t
            seg_end = t
            nan_gap = 0
            prev_valid_lon = lon_t
            continue

        # 有效点：先检查“相邻有效点”的 west-jump
        # 注意：这里不是相邻日，而是相邻“有效轨迹点”（中间 NaN 不算）
        jump = lon_t - float(prev_valid_lon)
        if jump < -MAX_WEST_JUMP_DEG:
            fail_west += 1
            finalize_segment(seg_start, seg_end)
            # 以当前有效点开启新段
            in_seg = True
            seg_start = t
            seg_end = t
            nan_gap = 0
            prev_valid_lon = lon_t
            continue

        # 连续延伸段
        seg_end = t
        nan_gap = 0
        prev_valid_lon = lon_t

    if in_seg:
        finalize_segment(seg_start, seg_end)

    print("[Event diagnostics - track-first]")
    print("  fail_west (cut by west-jump):", int(fail_west))
    print("  fail_short (no valid track in seg):", int(fail_short))
    print("  fail_active (active days < MIN_ACTIVE_DAYS_IN_EVENT):", int(fail_active))
    print("  fail_gate (did not pass IO/MC):", int(fail_gate))
    print("  fail_disp (disp < MIN_EAST_DISP):", int(fail_disp))
    print("  pass_events:", int(eid))

    return pd.DataFrame(events)


# ======================
# MAIN
# ======================
def main():
    # ======================
    # 缓存检测：若已有 EOF 结果，可直接加载跳过计算
    # ======================
    if SKIP_EOF_IF_EXISTS and EOF_CACHE_NC.exists():
        print(f"[CACHE] 检测到已有 EOF 结果，跳过计算直接加载: {EOF_CACHE_NC}")
        ds_cache = xr.open_dataset(EOF_CACHE_NC)
        
        # 提取所需变量
        pc1 = ds_cache["pc1"].values.astype(np.float32)
        pc2 = ds_cache["pc2"].values.astype(np.float32)
        amp = ds_cache["amp"].values.astype(np.float32)
        phase = ds_cache["phase"].values.astype(np.int16)
        recon_olr_latmean = ds_cache["olr_recon"]
        center_track_np_all = ds_cache["center_lon_track"].values.astype(float)
        olr_center_all = ds_cache["olr_center_track"].values.astype(float)
        time_index = pd.to_datetime(ds_cache["time"].values)
        
        # 冬季掩码
        winter_np = np.isin(time_index.month, list(WINTER_MONTHS)).astype(bool)
        
        # 准备事件识别所需数组
        center_track_np = center_track_np_all.copy()
        amp_np = amp.copy()
        olr_center_np = olr_center_all.copy()
        
        center_track_np[~winter_np] = np.nan
        amp_np[~winter_np] = np.nan
        olr_center_np[~winter_np] = np.nan
        
        print("冬季天数:", int(winter_np.sum()))
        active_center = np.isfinite(olr_center_np) & (olr_center_np <= OLR_MIN_THRESH)
        print("冬季满足强度阈值的天数:", int(np.nansum(active_center.astype(int))))
        
        # 直接跳到事件识别
        df_events = _build_events_track_first(
            time=time_index,
            center_track=center_track_np,
            olr_center=olr_center_np,
            amp=amp_np,
            winter_mask=winter_np
        )
        
        # 输出事件 CSV（不重写 NC，因为 EOF 结果不变）
        out_csv = OUT_DIR / f"mjo_events_step3_{START_DATE[:4]}-{END_DATE[:4]}.csv"
        
        if df_events.empty:
            df_events = pd.DataFrame(columns=[
                "event_id", "start_date", "end_date", "duration_days",
                "lon_start", "lon_end", "east_displacement_deg",
                "min_olr_min", "min_olr_mean", "amp_mean", "active_days",
            ])
        
        df_events.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[CACHE] 保存事件列表: {out_csv}")
        print("识别到的事件数量:", int(len(df_events)))
        if len(df_events) > 0:
            print(df_events.head(10).to_string(index=False))
        
        ds_cache.close()
        return
    
    # ======================
    # 完整 EOF 计算流程（首次运行或强制重算）
    # ======================
    print("[COMPUTE] 开始完整 EOF 计算...")
    # ---- load step2 outputs ----
    ds_olr = _to_lon_180(_rename_latlon_if_needed(_open_ds(OLR_BP_PATH)))
    ds_u   = _to_lon_180(_rename_latlon_if_needed(_open_ds(U_BP_PATH)))

    # ---- time subset ----
    ds_olr = ds_olr.sel(time=slice(START_DATE, END_DATE))
    ds_u   = ds_u.sel(time=slice(START_DATE, END_DATE))

    # ---- align time (inner join) ----
    ds_olr, ds_u = xr.align(ds_olr, ds_u, join="inner")
    ds_olr = ds_olr.chunk({"time": 90})
    ds_u   = ds_u.chunk({"time": 90})

    # ===== INSERT HERE: time consistency check + intersection =====
    t_olr = pd.to_datetime(ds_olr["time"].values)
    t_u   = pd.to_datetime(ds_u["time"].values)

    print("OLR time[0], time[-1]:", t_olr[0], t_olr[-1], "n=", len(t_olr))
    print("U time[0], time[-1]:", t_u[0], t_u[-1], "n=", len(t_u))
    print("Same length?", len(t_olr) == len(t_u))
    print("Exact same times?", np.array_equal(t_olr.values, t_u.values))

    # ---- keep 2D space (lat, lon) for MV-EOF ----
    ds_olr = _sel_lat_band(ds_olr, LAT_BAND)

    olr_bp_2d  = ds_olr["olr_bp"].transpose("time", "lat", "lon")
    u850_bp_2d = ds_u["u850_bp"].transpose("time", "lat", "lon")
    u200_bp_2d = ds_u["u200_bp"].transpose("time", "lat", "lon")

    # ---- OPTIONAL: restrict lon to tracking window for OLR (OK) ----
    olr_bp_2d = olr_bp_2d.sel(lon=slice(TRACK_LON_MIN, TRACK_LON_MAX))

    # 对 ERA5：先用稍大窗口，避免 coarsen(trim) 砍掉边界导致后续插值 NaN
    PAD_DEG = 10.0  # 5~10 都行，建议 10 更稳
    lat0 = float(LAT_BAND[0] - PAD_DEG)
    lat1 = float(LAT_BAND[1] + PAD_DEG)

    # 关键：根据 lat 方向决定 slice 顺序
    lat_vals = u850_bp_2d["lat"].values
    is_desc = (lat_vals.size >= 2) and ((lat_vals[1] - lat_vals[0]) < 0)
    lat_slice = slice(lat1, lat0) if is_desc else slice(lat0, lat1)

    lon_slice = slice(TRACK_LON_MIN - PAD_DEG, TRACK_LON_MAX + PAD_DEG)

    u850_bp_2d = u850_bp_2d.sel(lat=lat_slice, lon=lon_slice)
    u200_bp_2d = u200_bp_2d.sel(lat=lat_slice, lon=lon_slice)

    print(
        "ERA5 lat head/tail:",
        float(u850_bp_2d.lat.values[0]),
        float(u850_bp_2d.lat.values[-1]),
        "nlat=",
        u850_bp_2d.sizes["lat"]
    )

    def _guess_res(vals: np.ndarray) -> float:
        v = np.asarray(vals, dtype=float)
        dv = np.diff(v)
        dv = dv[np.isfinite(dv)]
        if dv.size == 0:
            return np.nan
        return float(np.nanmedian(np.abs(dv)))

    target = 2.5

    lat_res = _guess_res(u850_bp_2d["lat"].values)
    lon_res = _guess_res(u850_bp_2d["lon"].values)

    if (not np.isfinite(lat_res)) or lat_res <= 0:
        lat_res = target
    if (not np.isfinite(lon_res)) or lon_res <= 0:
        lon_res = target

    r_lat = target / lat_res
    r_lon = target / lon_res

    fac_lat = int(round(r_lat))
    fac_lon = int(round(r_lon))

    # 只有当比例足够接近整数才 coarsen；否则跳过
    if abs(r_lat - fac_lat) > 0.1:
        fac_lat = 1
    if abs(r_lon - fac_lon) > 0.1:
        fac_lon = 1

    fac_lat = max(1, fac_lat)
    fac_lon = max(1, fac_lon)

    print(f"Detected U grid: lat_res≈{lat_res}, lon_res≈{lon_res}, coarsen factors: lat={fac_lat}, lon={fac_lon}")

    if fac_lat > 1 or fac_lon > 1:
        u850_bp_2d = u850_bp_2d.coarsen(lat=fac_lat, lon=fac_lon, boundary="trim").mean(skipna=True)
        u200_bp_2d = u200_bp_2d.coarsen(lat=fac_lat, lon=fac_lon, boundary="trim").mean(skipna=True)
    else:
        print("Skip coarsen: U grid is ~2.5deg OR not an integer-factor to 2.5deg.")

    # ==================================
    # 对齐到 OLR 网格（必须做，最稳）
    # ==================================
    u850_bp_2d = u850_bp_2d.interp_like(olr_bp_2d, method="linear")
    u200_bp_2d = u200_bp_2d.interp_like(olr_bp_2d, method="linear")

    print(
        "OLR grid lat/lon:",
        float(olr_bp_2d.lat.min()), float(olr_bp_2d.lat.max()),
        float(olr_bp_2d.lon.min()), float(olr_bp_2d.lon.max())
    )
    print(
        "U850 grid lat/lon:",
        float(u850_bp_2d.lat.min()), float(u850_bp_2d.lat.max()),
        float(u850_bp_2d.lon.min()), float(u850_bp_2d.lon.max())
    )

    wmask = _winter_mask(olr_bp_2d["time"])
    olr_tr  = olr_bp_2d.where(wmask, drop=True)
    u850_tr = u850_bp_2d.where(wmask, drop=True)
    u200_tr = u200_bp_2d.where(wmask, drop=True)

    # ---- CRITICAL FIX: drop NaN training days ----
    olr_tr, u850_tr, u200_tr, _good = _drop_nan_training_days(olr_tr, u850_tr, u200_tr)
    if olr_tr.sizes["time"] < 30:
        raise RuntimeError(f"Too few valid winter training days after dropping NaNs: {olr_tr.sizes['time']}")

    # ---- training means (winter, after dropping NaNs) ----
    mu_olr  = olr_tr.mean("time", skipna=True)
    mu_u850 = u850_tr.mean("time", skipna=True)
    mu_u200 = u200_tr.mean("time", skipna=True)

    # ---- build training matrix (E: 2D space -> flatten) ----
    olr_tr_anom  = (olr_tr  - mu_olr ).values.astype(np.float64)  # (T,lat,lon)
    u850_tr_anom = (u850_tr - mu_u850).values.astype(np.float64)  # (T,lat,lon)
    u200_tr_anom = (u200_tr - mu_u200).values.astype(np.float64)  # (T,lat,lon)

    olr_tr_std,  s_olr  = _standardize_by_space_time_std(olr_tr_anom)
    u850_tr_std, s_u850 = _standardize_by_space_time_std(u850_tr_anom)
    u200_tr_std, s_u200 = _standardize_by_space_time_std(u200_tr_anom)

    # flatten (T,lat,lon) -> (T,P)
    Ttr, Y, X = olr_tr_std.shape
    P = Y * X
    olr_tr_2d  = olr_tr_std.reshape(Ttr, P)
    u850_tr_2d = u850_tr_std.reshape(Ttr, P)
    u200_tr_2d = u200_tr_std.reshape(Ttr, P)

    # MV-EOF input: (T, 3P)
    X_tr = np.concatenate([olr_tr_2d, u850_tr_2d, u200_tr_2d], axis=1)
    mu_X = np.mean(X_tr, axis=0, keepdims=True)
    X_tr = X_tr - mu_X

    X_tr = np.asarray(X_tr, dtype=np.float64)
    if not np.isfinite(X_tr).all():
        bad = np.where(~np.isfinite(X_tr))
        raise RuntimeError(
            f"Non-finite values remain in X_tr after dropping NaN days. Example index={bad[0][0]},{bad[1][0]}"
        )

    # ---- EOF fitting ----
    EOFs, PCs_tr, _svals = _svd_eof(X_tr, nmode=2)

    # ---- normalize PCs by training std ----
    pc1_tr = PCs_tr[:, 0]
    pc2_tr = PCs_tr[:, 1]
    pc1_scale = np.std(pc1_tr) if np.std(pc1_tr) != 0 else 1.0
    pc2_scale = np.std(pc2_tr) if np.std(pc2_tr) != 0 else 1.0

    # ---- project FULL period onto EOFs (F: 2D space -> flatten) ----
    olr_full_anom  = ((olr_bp_2d  - mu_olr ) / s_olr ).values.astype(np.float64)   # (T,lat,lon)
    u850_full_anom = ((u850_bp_2d - mu_u850) / s_u850).values.astype(np.float64)  # (T,lat,lon)
    u200_full_anom = ((u200_bp_2d - mu_u200) / s_u200).values.astype(np.float64)  # (T,lat,lon)

    Tfull = olr_full_anom.shape[0]

    # ---- CRITICAL: handle edge-NaN days in FULL period (from filtering) ----
    good_full = (
        np.isfinite(olr_full_anom).all(axis=(1, 2))
        & np.isfinite(u850_full_anom).all(axis=(1, 2))
        & np.isfinite(u200_full_anom).all(axis=(1, 2))
    )

    good_idx = np.nonzero(good_full)[0]
    if good_idx.size < 10:
        raise RuntimeError(f"Too few valid full days: {good_idx.size}. Check filtering edge NaNs.")

    print("Full days:", Tfull, "valid full days:", int(good_full.sum()), "dropped:", int((~good_full).sum()))

    # pre-allocate outputs as NaN (full length)
    pc1 = np.full(Tfull, np.nan, dtype=np.float32)
    pc2 = np.full(Tfull, np.nan, dtype=np.float32)

    # only compute on valid days
    olr_full_2d  = olr_full_anom[good_idx].reshape(len(good_idx), P)
    u850_full_2d = u850_full_anom[good_idx].reshape(len(good_idx), P)
    u200_full_2d = u200_full_anom[good_idx].reshape(len(good_idx), P)

    X_full = np.concatenate([olr_full_2d, u850_full_2d, u200_full_2d], axis=1)
    X_full = X_full - mu_X
    X_full = np.asarray(X_full, dtype=np.float64)
    if not np.isfinite(X_full).all():
        bad = np.where(~np.isfinite(X_full))
        raise RuntimeError(f"Non-finite in X_full(valid days). Example={bad[0][0]},{bad[1][0]}")

    PCs_full = X_full @ EOFs
    pc1[good_idx] = (PCs_full[:, 0] / pc1_scale).astype(np.float32)
    pc2[good_idx] = (PCs_full[:, 1] / pc2_scale).astype(np.float32)

    amp = np.sqrt(pc1**2 + pc2**2).astype(np.float32)
    phase = _phase8(pc1, pc2).astype(np.int16)
    phase[~np.isfinite(pc1) | ~np.isfinite(pc2)] = 0  # optional: invalid day phase=0

    # ---- reconstructed OLR from first 2 EOFs (OLR block only) (G: reconstruct 2D) ----
    # OLR block length = P (lat*lon)
    EOFs_olr = EOFs[:P, :]  # (P,2)

    # recon in standardized space: (T,P) 先全 NaN
    recon_std_flat = np.full((Tfull, P), np.nan, dtype=np.float64)

    # 只对 valid days 重构（用 pc1/pc2 的 valid 子集）
    recon_std_flat[good_idx, :] = (
        (pc1[good_idx, None].astype(np.float64) * pc1_scale) * EOFs_olr[:, 0][None, :]
        + (pc2[good_idx, None].astype(np.float64) * pc2_scale) * EOFs_olr[:, 1][None, :]
    )

    # add back std + mean (mu_olr is (lat,lon))
    mu_olr_flat = mu_olr.values.astype(np.float64).reshape(P)  # consistent with (lat,lon) flatten
    recon_olr_flat = recon_std_flat * s_olr + mu_olr_flat[None, :]  # (T,P)

    # reshape to (T,lat,lon)
    recon_olr_3d = recon_olr_flat.reshape(Tfull, Y, X)

    recon_olr_da_2d = xr.DataArray(
        recon_olr_3d.astype(np.float32),
        coords={"time": olr_bp_2d["time"], "lat": olr_bp_2d["lat"], "lon": olr_bp_2d["lon"]},
        dims=("time", "lat", "lon"),
        name="olr_recon_2d",
        attrs={"desc": "Reconstructed OLR from 2 leading MV-EOF modes (2D lat-lon)."}
    )

    # IMPORTANT: tracking/patch still uses (time,lon), so take 15S–15N mean AFTER reconstruction
    recon_olr_latmean = recon_olr_da_2d.mean("lat", skipna=True).transpose("time", "lon").rename("olr_recon")

    # ---- winter-only mask for event building ----
    time_index = pd.to_datetime(olr_bp_2d["time"].values)
    winter_np = np.isin(time_index.month, list(WINTER_MONTHS)).astype(bool)

    # ---- continuous tracking center (use lat-mean recon) ----
    center_track_da = _track_center_with_candidates(
        recon_olr_latmean,
        lon_min=TRACK_LON_MIN,
        lon_max=TRACK_LON_MAX,
        max_west_jump=MAX_WEST_JUMP_DEG
    )
    center_track_np_all = center_track_da.values.astype(float)

    # ---- OLR at tracked center (for intensity flag, avoid daily_min nanmin warning) ----
    olr_center_all = _sample_olr_at_track_center(recon_olr_latmean, center_track_np_all)

    center_track_out_da = xr.DataArray(
        center_track_np_all,
        coords={"time": recon_olr_latmean["time"]},
        dims=("time",),
        name="center_lon_track"
    )

    # arrays for events
    center_track_np = center_track_np_all.copy()
    amp_np = amp.copy()

    # enforce non-winter = inactive/ignored
    center_track_np[~winter_np] = np.nan
    amp_np[~winter_np] = np.nan

    olr_center_np = olr_center_all.copy()
    olr_center_np[~winter_np] = np.nan

    print("Winter days total:", int(winter_np.sum()))
    active_center = np.isfinite(olr_center_np) & (olr_center_np <= OLR_MIN_THRESH)
    print("Winter days with center_olr<=thr:", int(np.nansum(active_center.astype(int))))
    print("Fraction:", float(np.nansum(active_center.astype(int))) / float(winter_np.sum()))

    # ---- build events (paper-style) ----
    df_events = _build_events_track_first(
        time=time_index,
        center_track=center_track_np,
        olr_center=olr_center_np,
        amp=amp_np,
        winter_mask=winter_np
    )

    # ---- outputs ----
    out_nc = OUT_DIR / f"mjo_mvEOF_step3_{START_DATE[:4]}-{END_DATE[:4]}.nc"
    out_csv = OUT_DIR / f"mjo_events_step3_{START_DATE[:4]}-{END_DATE[:4]}.csv"

    ds_out = xr.Dataset(
        {
            "pc1": xr.DataArray(pc1, coords={"time": olr_bp_2d["time"]}, dims=("time",)),
            "pc2": xr.DataArray(pc2, coords={"time": olr_bp_2d["time"]}, dims=("time",)),
            "amp": xr.DataArray(amp, coords={"time": olr_bp_2d["time"]}, dims=("time",)),
            "phase": xr.DataArray(phase, coords={"time": olr_bp_2d["time"]}, dims=("time",)),
            "olr_recon": recon_olr_latmean,  # (time,lon)
            # add continuous track center (needed by paper-style event logic)
            "center_lon_track": center_track_out_da.astype(np.float32),
            "olr_center_track": xr.DataArray(
                olr_center_all.astype(np.float32),
                coords={"time": olr_bp_2d["time"]},
                dims=("time",)
            ),
        },
        attrs={
            "step": "Step3 MV-EOF + reconstructed OLR + DP-tracked convective center + track-first MJO event identification",
            "lat_band": f"{LAT_BAND[0]} to {LAT_BAND[1]}",
            "winter_months": ",".join(map(str, sorted(WINTER_MONTHS))),
            "track_lon_window": f"{TRACK_LON_MIN}E to {TRACK_LON_MAX}E",
            "max_west_jump_deg": str(MAX_WEST_JUMP_DEG),
            "olr_min_thresh": str(OLR_MIN_THRESH),
            "min_east_disp_deg": str(MIN_EAST_DISP_DEG),
            "io_gate_lon": str(IO_GATE_LON),
            "mc_gate_lon": str(MC_GATE_LON),
            "amp_thresh": str(AMP_THRESH),
            "dp_jump_weight": str(DP_JUMP_WEIGHT),
            "dp_near_min_delta": str(DP_NEAR_MIN_DELTA),
            "dp_max_cands": str(DP_MAX_CANDS),
            "scales": f"s_olr={s_olr}, s_u850={s_u850}, s_u200={s_u200}, pc1_scale={pc1_scale}, pc2_scale={pc2_scale}",
            "training_days_used": str(int(olr_tr.sizes["time"])),
        }
    )

    enc = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    for c in ds_out.coords:
        enc[c] = {"zlib": False, "_FillValue": None}
    ds_out.to_netcdf(out_nc, engine="netcdf4", encoding=enc)

    if df_events.empty:
        df_events = pd.DataFrame(columns=[
            "event_id", "start_date", "end_date", "duration_days",
            "lon_start", "lon_end", "east_displacement_deg",
            "min_olr_min", "min_olr_mean", "amp_mean",
            "active_days",
        ])

    df_events.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("Saved:", str(out_nc))
    print("Saved:", str(out_csv))
    print("Winter training days used (after dropping NaNs):", int(olr_tr.sizes["time"]))
    print("Events found:", int(len(df_events)))
    if len(df_events) > 0:
        print(df_events.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
