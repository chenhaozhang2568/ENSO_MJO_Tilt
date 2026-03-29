# -*- coding: utf-8 -*-
"""
03_compute_tilt_daily.py: Step4 - 逐日 MJO Tilt 指数计算

================================================================================
功能描述：
    本脚本基于 ERA5 带通滤波 omega（w'）和 Step3 对流中心轨迹，
    计算逐日 MJO 垂直倾斜（Tilt）指数。

Tilt 定义：
    Tilt = 低层上升区西边界 - 高层上升区西边界（相对经度，单位：°）
    
    正值表示"低层偏东、高层偏西"的典型 MJO 后倾结构。

层次定义：
    低层：1000-600 hPa（层平均）
    高层：400-200 hPa（层平均）

输入数据：
    - Step3 输出：center_lon_track（对流中心轨迹）
    - ERA5 带通 omega：w_bp（纬向平均）

输出：
    - tilt_daily_step4_layermean_1979-2022.nc
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.interpolate import BSpline

# ======================
# USER PATHS
# ======================
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
# Use normalized omega from 02_mvEOF.py output (already MJO reconstructed + amp normalized)
W_NORM_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
OUT_NC = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"


# ======================
# SETTINGS (match Step3)
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
TRACK_LON_MIN = 0.0
TRACK_LON_MAX = 240.0   # 东端扩展到 240°E（覆盖西太平洋以东）

# --- layer-mean definition (hPa) ---
LOW_LAYER = (1000.0, 600.0)   # inclusive slice: 1000..600
UP_LAYER  = (400.0, 200.0)    # inclusive slice: 400..200
# 备选：UP_LAYER = (500.0, 200.0)
PRESSURE_WEIGHTED = False       # True=气压厚度加权层平均, False=等权重平均


# ---------- HALF-MAX (FWHM-like) boundary ----------
# 边界定义：ω 达到 HALF_MAX_FRACTION × ω_min 的位置
# 类似光谱分析的半高宽 (Full Width at Half Maximum)
# HALF_MAX_FRACTION = 0.5 表示边界处 ω = 50% × ω_min
HALF_MAX_FRACTION = 0.0  # 50% = half-max (FWHM-like)
EDGE_N_CONSEC = 1        # 连续 N 个点都满足阈值才算出边界
SMOOTH_WINDOW = 10        # 边界检测前滑动平均窗口（1=不平滑）
PIVOT_DELTA_DEG = 10.0   # pivot 搜索范围

# --- NCL-matching preprocessing (csa1 + smth9) ---
CSA_KNOTS = 9            # NCL csa1 knots 参数
CSA_TARGET_DLON = 0.25   # 目标经度分辨率（°）
SMTH9_P = 0.50           # NCL smth9 p 参数
SMTH9_Q = 0.25           # NCL smth9 q 参数
MIN_VALID_POINTS = 7
OLR_MIN_THRESH = -15.0
ACTIVE_ONLY = False


# --- amp normalize safety (minimal) ---
AMP_EPS = 1e-6  # avoid divide-by-zero / blow-up
AMP_FLOOR = 1.0  # NEW: clamp amp for normalization (prevents small-amp blow-up)

# ======================
# helpers
# ======================
def _pressure_weighted_mean(da: xr.DataArray, layer_bounds: tuple) -> xr.DataArray:
    """
    气压厚度加权层平均。

    每层权重 = 该层所代表的气压厚度 Δp（相邻层中点之差）。
    最外两层的外侧边界取 layer_bounds 限定。

    Parameters
    ----------
    da : xr.DataArray  含 'level' 维
    layer_bounds : (p_top, p_bot)  层组的上下界 hPa，顺序不限
    """
    levels = da["level"].values.astype(float)
    p_top = min(layer_bounds)
    p_bot = max(layer_bounds)
    n = len(levels)
    dp = np.empty(n, dtype=float)
    for k in range(n):
        upper = 0.5 * (levels[k] + levels[k - 1]) if k > 0     else p_bot
        lower = 0.5 * (levels[k] + levels[k + 1]) if k < n - 1 else p_top
        dp[k] = abs(upper - lower)
    weights = xr.DataArray(dp, dims=["level"], coords={"level": levels})
    return da.weighted(weights).mean("level")

def _winter_np(time_index: pd.DatetimeIndex) -> np.ndarray:
    return np.isin(time_index.month, list(WINTER_MONTHS)).astype(bool)

def _mask_event_days(time: pd.DatetimeIndex, events_csv: str) -> np.ndarray:
    ev = pd.read_csv(events_csv, parse_dates=["start_date", "end_date"])
    m = np.zeros(len(time), dtype=bool)
    if ev.empty:
        return m
    tv = time.values.astype("datetime64[ns]")
    for _, r in ev.iterrows():
        s = np.datetime64(pd.Timestamp(r["start_date"]).normalize().to_datetime64())
        e = np.datetime64(pd.Timestamp(r["end_date"]).normalize().to_datetime64())
        i0 = int(np.searchsorted(tv, s, side="left"))
        i1 = int(np.searchsorted(tv, e, side="right")) - 1
        if i1 >= i0:
            m[i0:i1+1] = True
    return m


def _cubic_spline_approx_lon(da: xr.DataArray, n_knots: int,
                             target_dlon: float) -> xr.DataArray:
    """
    沿经度维做 Akima 三次样条插值，匹配 NCL csa1。

    使用 scipy Akima1DInterpolator，通过预计算插值矩阵实现向量化。
    """
    from scipy.interpolate import Akima1DInterpolator

    src_lon = da["lon"].values.astype(float)
    lon_min, lon_max = float(src_lon.min()), float(src_lon.max())
    n_target = int(round((lon_max - lon_min) / target_dlon)) + 1
    target_lon = np.linspace(lon_min, lon_max, n_target)

    # 预计算插值矩阵：逐基向量构建 Akima 插值并采样
    n_src = len(src_lon)
    M = np.zeros((len(target_lon), n_src))
    for j in range(n_src):
        e_j = np.zeros(n_src)
        e_j[j] = 1.0
        M[:, j] = Akima1DInterpolator(src_lon, e_j)(target_lon)

    data = da.values
    orig_shape = data.shape
    flat = data.reshape(-1, orig_shape[-1])  # (N, n_src)
    out_flat = flat @ M.T                    # (N, n_tgt)
    out = out_flat.reshape(*orig_shape[:-1], len(target_lon))

    new_coords = {d: (target_lon if d == "lon" else da[d].values)
                  for d in da.dims}
    result = xr.DataArray(out, dims=da.dims, coords=new_coords)
    result["lon"].attrs["units"] = "degrees_east"
    return result


def _smth9(data2d: np.ndarray, p: float, q: float, cyclic: bool) -> np.ndarray:
    """
    NCL smth9 的 Python 实现（向量化）。

    公式: f' = f + (p/4)*(cross - 4f) + (q/4)*(diag - 4f)
    边缘行（第一行/最后一行）保持不变，与 NCL 行为一致。
    cyclic=True 时第二维（经度）做周期边界处理。
    """
    ny, nx = data2d.shape
    if ny < 3:
        return data2d.copy()

    # 经度维 padding
    if cyclic:
        ext = np.concatenate([data2d[:, -1:], data2d, data2d[:, :1]], axis=1)
    else:
        ext = np.pad(data2d, ((0, 0), (1, 1)), mode='edge')

    # 向量化计算内部行 [1:-1]
    c     = ext[1:-1, 1:-1]
    cross = ext[:-2, 1:-1] + ext[2:, 1:-1] + ext[1:-1, :-2] + ext[1:-1, 2:]
    diag  = ext[:-2, :-2] + ext[:-2, 2:] + ext[2:, :-2] + ext[2:, 2:]

    result = data2d.copy()
    result[1:-1, :] = c + (p / 4.0) * (cross - 4.0 * c) + (q / 4.0) * (diag - 4.0 * c)
    return result


def _ascent_boundary_by_half_max(
    rel_lon: np.ndarray,
    w: np.ndarray,
    half_max_fraction: float = 0.5,
    pivot_delta: float = 10.0,
    n_consec: int = 1
):
    """
    使用半高宽法 (FWHM-like) 定义边界，解决长尾效应：
    
    原理：
      - 找到上升核心 (pivot)：ω 最小值位置
      - 边界阈值 thr = half_max_fraction × ω_min（如 50% × ω_min）
      - 从 pivot 向西/东扫描，第一次 ω >= thr 的位置即为边界
    
    类似光谱分析的 FWHM（Full Width at Half Maximum）：
      - half_max_fraction = 0.5 对应半高宽
      - 物理意义：边界处上升运动衰减到峰值的 50%
    
    Returns: (west_rel, east_rel, center_rel, wmin)
    """
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return (np.nan, np.nan, np.nan, np.nan)
    
    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)
    
    # 注意：滑动平均已在 main() 中逐层预处理完成，此处不再重复平滑
    
    # --- 直接以 OLR 对流中心 (rel_lon=0) 为参考点，匹配 NCL ---
    pivot_idx = int(np.argmin(np.abs(rr)))
    
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        # 没有上升运动
        return (np.nan, np.nan, np.nan, wmin)
    
    # 半高宽阈值：边界处 ω = half_max_fraction × ω_min
    # 例如 half_max_fraction=0.5, ω_min=-0.02 => thr = -0.01
    thr = float(half_max_fraction) * wmin  # wmin<0, 所以 thr<0
    
    # ============ west edge ============
    outside = 0
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:  # 上升强度衰减到阈值以下
            outside += 1
        else:
            outside = 0
        if outside >= n_consec:
            cand = i + n_consec
            cand = min(cand, pivot_idx)
            west_idx = cand
            break
    
    if west_idx is None:
        return (np.nan, np.nan, np.nan, wmin)  # 没找到边界 → 无效
    
    # ============ east edge ============
    outside = 0
    east_idx = None
    for i in range(pivot_idx, len(ww)):
        if ww[i] >= thr:
            outside += 1
        else:
            outside = 0
        if outside >= n_consec:
            cand = i - n_consec
            cand = max(cand, pivot_idx)
            east_idx = cand
            break
    
    if east_idx is None:
        return (np.nan, np.nan, np.nan, wmin)  # 没找到边界 → 无效
    
    west = float(rr[west_idx])
    east = float(rr[east_idx])
    
    # 防御：噪声导致 west>east 直接判无效
    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return (np.nan, np.nan, np.nan, wmin)
    
    center = 0.5 * (west + east)
    return (west, east, center, wmin)




def main():
    out_path = Path(OUT_NC)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load Step3 (track) ---
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    if "center_lon_track" not in ds3:
        raise RuntimeError("Step3 nc missing variable: center_lon_track")
    for v in ["pc1", "pc2", "amp", "olr_center_track"]:
        if v not in ds3:
            raise RuntimeError(f"Step3 nc missing variable: {v}")

    t3 = pd.to_datetime(ds3["time"].values)
    center = ds3["center_lon_track"].astype(float)
    olr_center = ds3["olr_center_track"].astype(float)
    amp = ds3["amp"].astype(float)  # Only need amp for filtering, not for reconstruction

    # --- load normalized omega from 02_mvEOF.py output ---
    print("Loading normalized omega from 02_mvEOF output...")
    dsw = xr.open_dataset(W_NORM_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    
    w_var = "w_mjo_recon_norm"
    if w_var not in dsw:
        raise RuntimeError(f"Normalized omega file missing variable: {w_var}")
    
    w = dsw[w_var]  # (time, pressure_level, lon)
    
    # Rename pressure_level to level if needed
    if "pressure_level" in w.dims:
        w = w.rename({"pressure_level": "level"})
    
    # 将经度从 -180~180 转为 0~360，以支持 TRACK_LON_MAX > 180
    lon_vals = w["lon"].values
    if lon_vals.min() < 0:
        new_lon = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
        w = w.assign_coords(lon=new_lon).sortby("lon")

    # --- 三次样条逼近到 0.25° 分辨率 (匹配 NCL csa1) ---
    print(f"  Cubic spline approximation (knots={CSA_KNOTS}, "
          f"target dlon={CSA_TARGET_DLON}°)...")
    w = _cubic_spline_approx_lon(w, CSA_KNOTS, CSA_TARGET_DLON)
    print(f"  After interpolation: lon {w['lon'].shape}")

    # --- 逐层滑动平均平滑 (每层独立, window={SMOOTH_WINDOW}) ---
    if SMOOTH_WINDOW > 1:
        print(f"  Applying per-level sliding average (window={SMOOTH_WINDOW})...")
        w_vals = w.values  # (time, level, lon)
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        for t in range(w_vals.shape[0]):
            for k in range(w_vals.shape[1]):
                profile = w_vals[t, k, :]
                valid = np.isfinite(profile).astype(float)
                filled = np.where(np.isfinite(profile), profile, 0.0)
                smoothed = np.convolve(filled, kernel, mode='same')
                count = np.convolve(valid, kernel, mode='same')
                count[count < 1e-10] = np.nan
                w_vals[t, k, :] = smoothed / count
        w = xr.DataArray(w_vals, dims=w.dims, coords=w.coords)

    # layer-mean: low (1000-600 hPa), up (400-200 hPa)
    w_low_sel = w.sel(level=slice(LOW_LAYER[0], LOW_LAYER[1]))
    w_up_sel  = w.sel(level=slice(UP_LAYER[0],  UP_LAYER[1]))
    if PRESSURE_WEIGHTED:
        print("  Using pressure-thickness weighted layer mean")
        w_low = _pressure_weighted_mean(w_low_sel, LOW_LAYER)
        w_up  = _pressure_weighted_mean(w_up_sel,  UP_LAYER)
    else:
        print("  Using equal-weight layer mean")
        w_low = w_low_sel.mean("level", skipna=True)
        w_up  = w_up_sel.mean("level", skipna=True)

    # make sure dims are (time, lon)
    w_low = w_low.transpose("time", "lon")
    w_up  = w_up.transpose("time", "lon")

    # subset lon to tracking window
    w_low = w_low.sel(lon=slice(TRACK_LON_MIN, TRACK_LON_MAX))
    w_up  = w_up.sel(lon=slice(TRACK_LON_MIN, TRACK_LON_MAX))
    
    print(f"  w_low shape: {w_low.shape}, w_up shape: {w_up.shape}")

    # align time with Step3 (inner intersection)
    center_a, olr_center_a, amp_a, w_low_a, w_up_a = xr.align(
        center, olr_center, amp, w_low, w_up, join="inner"
    )
    time = pd.to_datetime(center_a["time"].values)
    winter = _winter_np(time)
    active = (olr_center_a.values.astype(float) <= OLR_MIN_THRESH) & np.isfinite(olr_center_a.values.astype(float))
    eventmask = _mask_event_days(time, EVENTS_CSV)

    lon = w_low_a["lon"].values.astype(float)

    # Data is already normalized by amp in 02_mvEOF.py, use directly
    w_low_norm = w_low_a.values.astype(float)
    w_up_norm = w_up_a.values.astype(float)
    
    amp_np_all = amp_a.values.astype(float)
    amp_ok = np.isfinite(amp_np_all) & (amp_np_all > AMP_EPS)

    # pre-allocate
    n = time.size
    low_west = np.full(n, np.nan, np.float32)
    low_east = np.full(n, np.nan, np.float32)
    low_ctr  = np.full(n, np.nan, np.float32)
    low_wmin = np.full(n, np.nan, np.float32)

    up_west = np.full(n, np.nan, np.float32)
    up_east = np.full(n, np.nan, np.float32)
    up_ctr  = np.full(n, np.nan, np.float32)
    up_wmin = np.full(n, np.nan, np.float32)

    tilt = np.full(n, np.nan, np.float32)
    tilt_east = np.full(n, np.nan, np.float32)

    # load arrays (faster than per-day isel)
    c_np = center_a.values.astype(float)
    # wl_np = w_low_a.values.astype(float)  # (time, lon)
    # wu_np = w_up_a.values.astype(float)

    for i in range(n):
        if not winter[i]:
            continue
        if ACTIVE_ONLY and (not active[i]):
            continue
        if not eventmask[i]:
            continue

        c = c_np[i]
        if not np.isfinite(c):
            continue

        # (振幅筛选已去除，与 NCL 一致)

        # build relative lon axis in tracking window (no wrap needed)
        rel = lon - float(c)

        # use reconstructed+normalized profiles
        wl = w_low_norm[i, :]
        wu = w_up_norm[i, :]

        lw, le, lc, lmin = _ascent_boundary_by_half_max(
            rel, wl, HALF_MAX_FRACTION, PIVOT_DELTA_DEG, EDGE_N_CONSEC
        )
        uw, ue, uc, umin = _ascent_boundary_by_half_max(
            rel, wu, HALF_MAX_FRACTION, PIVOT_DELTA_DEG, EDGE_N_CONSEC
        )

        low_west[i], low_east[i], low_ctr[i], low_wmin[i] = lw, le, lc, lmin
        up_west[i],  up_east[i],  up_ctr[i],  up_wmin[i]  = uw, ue, uc, umin

        # 西侧 tilt：low_west - up_west
        if np.isfinite(lw) and np.isfinite(uw):
            tilt[i] = float(lw - uw)
        else:
            tilt[i] = np.nan
        
        # 东侧 tilt：low_east - up_east
        if np.isfinite(le) and np.isfinite(ue):
            tilt_east[i] = float(le - ue)
        else:
            tilt_east[i] = np.nan

    ds_out = xr.Dataset(
        {
            "low_west_rel": xr.DataArray(low_west, coords={"time": time}, dims=("time",)),
            "low_east_rel": xr.DataArray(low_east, coords={"time": time}, dims=("time",)),
            "low_center_rel": xr.DataArray(low_ctr, coords={"time": time}, dims=("time",)),
            "low_wmin": xr.DataArray(low_wmin, coords={"time": time}, dims=("time",)),
            "up_west_rel": xr.DataArray(up_west, coords={"time": time}, dims=("time",)),
            "up_east_rel": xr.DataArray(up_east, coords={"time": time}, dims=("time",)),
            "up_center_rel": xr.DataArray(up_ctr, coords={"time": time}, dims=("time",)),
            "up_wmin": xr.DataArray(up_wmin, coords={"time": time}, dims=("time",)),
            "tilt": xr.DataArray(tilt, coords={"time": time}, dims=("time",),
                         attrs={"desc": "tilt = low_west_rel - up_west_rel (deg), 西侧倾斜"}),
            "tilt_east": xr.DataArray(tilt_east, coords={"time": time}, dims=("time",),
                         attrs={"desc": "tilt_east = low_east_rel - up_east_rel (deg), 东侧倾斜"}),
            "active_mask": xr.DataArray(active.astype(np.int8), coords={"time": time}, dims=("time",),
                            attrs={"desc": f"1 if olr_center_track <= {OLR_MIN_THRESH} else 0"}),
            "event_mask": xr.DataArray(eventmask.astype(np.int8), coords={"time": time}, dims=("time",),
                           attrs={"desc": "1 if within any Step3 event [start,end] else 0"}),
        },
        attrs={
            "source_step3": STEP3_NC,
            "source_w": W_NORM_NC,
            "levels": f"low_layer={LOW_LAYER[0]}..{LOW_LAYER[1]}hPa, up_layer={UP_LAYER[0]}..{UP_LAYER[1]}hPa",
            "lon_window": f"{TRACK_LON_MIN}..{TRACK_LON_MAX}",
            "layer_mean_method": "pressure_weighted" if PRESSURE_WEIGHTED else "equal_weight",
            "boundary_method": "half_max_fwhm",
            "half_max_fraction": str(HALF_MAX_FRACTION),
            "winter_months": ",".join(map(str, sorted(WINTER_MONTHS))),
            "time_range": f"{START_DATE}..{END_DATE}",
            "active_only": str(ACTIVE_ONLY),
            "olr_min_thresh_active": str(OLR_MIN_THRESH),
            "events_csv": EVENTS_CSV,
            "mjo_recon": "w reconstructed by regression on Step3 pc1/pc2 (with intercept), then normalized by amp",
            "amp_eps": str(AMP_EPS),
        }
    )

    enc = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    enc["time"] = {"zlib": False, "_FillValue": None}
    ds_out.to_netcdf(out_path, engine="netcdf4", encoding=enc)

    # quick console summary
    tv = ds_out["tilt"].values.astype(float)
    ok = np.isfinite(tv)
    print("Saved:", str(out_path))
    print("tilt finite days:", int(ok.sum()), "/", int(tv.size),
          "winter:", int(winter.sum()))
    if ok.any():
        print("tilt stats (deg): min", float(np.nanmin(tv)), "p5", float(np.nanpercentile(tv, 5)),
              "mean", float(np.nanmean(tv)), "median", float(np.nanpercentile(tv, 50)),
              "p95", float(np.nanpercentile(tv, 95)), "max", float(np.nanmax(tv)))

if __name__ == "__main__":
    main()
