# -*- coding: utf-8 -*-
"""
Step4: compute daily tilt(t) from ERA5 w' (bandpass) using Step3 convective center track.

Inputs:
- Step3 NC: center_lon_track(time)
- ERA5 w bandpass latmean: w_bp(level, lon, time)

Outputs:
- tilt_daily_step4_1979-2022.nc

Run:
cd /d E:\Projects\ENSO_MJO_Tilt
python src\04_compute_tilt_daily_step4.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ======================
# USER PATHS
# ======================
STEP3_NC = r"E:\Datas\ClimateIndex\processed\mjo_mvEOF_step3_1979-2022.nc"
ERA5_W_LATMEAN = r"E:\Datas\ERA5\processed\era5_w_bp_latmean_1979-2022.nc"
OUT_NC = r"E:\Datas\ClimateIndex\processed\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\ClimateIndex\processed\mjo_events_step3_1979-2022.csv"

# ======================
# SETTINGS (match Step3)
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
TRACK_LON_MIN = 0.0
TRACK_LON_MAX = 180.0

# --- layer-mean definition (hPa) ---
LOW_LAYER = (1000.0, 700.0)   # inclusive slice: 1000..700
UP_LAYER  = (300.0, 200.0)    # inclusive slice: 300..200
# 备选：UP_LAYER = (500.0, 200.0)


# ---------- ZERO-crossing boundary ----------
ZERO_TOL = 0.0      # Pa/s，容忍量：认为 w >= -ZERO_TOL 就“到 0 了”
EDGE_N_CONSEC = 1    # 连续 N 个点都满足“到 0”才算出边界
EDGE_PAD_DEG  = 2.5  # 边界留余量（度）

# pivot 仍可保留：用来锁定“从哪里向外扫”
PIVOT_DELTA_DEG = 10.0
MIN_VALID_POINTS = 7
OLR_MIN_THRESH = -15.0
ACTIVE_ONLY = False


# --- amp normalize safety (minimal) ---
AMP_EPS = 1e-6  # avoid divide-by-zero / blow-up
AMP_FLOOR = 1.0  # NEW: clamp amp for normalization (prevents small-amp blow-up)

# ======================
# helpers
# ======================
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

def _ascent_center_from_profile(
    rel_lon: np.ndarray,
    w: np.ndarray,
    zero_tol: float,
    pivot_delta: float,
    n_consec: int,
    pad: float
):
    """
    以 omega=0 作为东西边界（允许容忍量 zero_tol）：

    定义：
      inside(上升区)  : w < -zero_tol
      outside(到0/非上升): w >= -zero_tol   （注意仍可能是微弱负值，但当作“到0”）

    步骤：
      1) pivot：在 [-pivot_delta,+pivot_delta] 内找 w 最小值（最强上升核）
      2) 从 pivot 向西/东扫描：第一次出现 连续 n_consec 个点 outside => 边界取前一格（更靠近 pivot）
      3) 若一侧始终找不到 outside（无 0 交点），边界取窗口边缘（表示上升区覆盖到窗口边界）
    Returns: (west_rel, east_rel, center_rel, wmin)
    """
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return (np.nan, np.nan, np.nan, np.nan)

    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)

    # --- pivot：0附近最强上升点 ---
    win = (rr >= -pivot_delta) & (rr <= pivot_delta)
    if win.any():
        j0 = int(np.nanargmin(ww[win]))
        pivot_idx = np.where(win)[0][j0]
    else:
        pivot_idx = int(np.nanargmin(ww))

    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= -abs(zero_tol)):
        # pivot 都不明显为负（接近0/正），说明没有可靠上升核
        return (np.nan, np.nan, np.nan, wmin)

    thr0 = -abs(float(zero_tol))   # “到0”的判据：w >= thr0

    # ============ west edge ============
    outside = 0
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr0:
            outside += 1
        else:
            outside = 0
        if outside >= n_consec:
            cand = i + n_consec
            cand = min(cand, pivot_idx)
            west_idx = cand
            break

    if west_idx is None:
        west_idx = 0

    # ============ east edge ============
    outside = 0
    east_idx = None
    for i in range(pivot_idx, len(ww)):
        if ww[i] >= thr0:
            outside += 1
        else:
            outside = 0
        if outside >= n_consec:
            cand = i - n_consec
            cand = max(cand, pivot_idx)
            east_idx = cand
            break

    if east_idx is None:
        east_idx = len(ww) - 1

    west = float(rr[west_idx])
    east = float(rr[east_idx])

    # 防御：噪声导致 west>east 直接判无效
    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return (np.nan, np.nan, np.nan, wmin)

    # padding + clip
    rr_min = float(np.nanmin(rr))
    rr_max = float(np.nanmax(rr))
    west = max(rr_min, west - float(pad))
    east = min(rr_max, east + float(pad))

    center = 0.5 * (west + east)
    return (west, east, center, wmin)


def _reconstruct_by_pc12_regression(
    w_time_lon: xr.DataArray,
    pc1: xr.DataArray,
    pc2: xr.DataArray,
    winter_mask: np.ndarray
) -> xr.DataArray:
    """
    Reconstruct w(time,lon) using linear regression on PC1/PC2:
      w_hat(t,lon) = a(lon) + b1(lon)*pc1(t) + b2(lon)*pc2(t)

    Minimal implementation: loop over lon (typically small), robust to NaNs.
    """
    # ensure aligned dims
    w_time_lon = w_time_lon.transpose("time", "lon")
    pc1 = pc1.transpose("time")
    pc2 = pc2.transpose("time")

    time = w_time_lon["time"].values
    # lon = w_time_lon["lon"].values.astype(float)

    w_np = w_time_lon.values.astype(float)   # (T,L)
    pc1_np = pc1.values.astype(float)        # (T,)
    pc2_np = pc2.values.astype(float)        # (T,)
    T, L = w_np.shape

    out = np.full((T, L), np.nan, dtype=float)

    # precompute pc finite + winter
    base_mask = winter_mask & np.isfinite(pc1_np) & np.isfinite(pc2_np)

    for j in range(L):
        y = w_np[:, j]
        m = base_mask & np.isfinite(y)
        if m.sum() < 10:
            continue

        # X = [1, pc1, pc2]
        X = np.column_stack([np.ones(m.sum(), dtype=float), pc1_np[m], pc2_np[m]])
        yy = y[m]

        # least squares
        beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)  # (3,)

        # reconstruct for all times where pc1/pc2 finite
        mt = np.isfinite(pc1_np) & np.isfinite(pc2_np)
        Xall = np.column_stack([np.ones(mt.sum(), dtype=float), pc1_np[mt], pc2_np[mt]])
        out[mt, j] = beta[1] * pc1_np[mt] + beta[2] * pc2_np[mt]

    return xr.DataArray(
        out.astype(np.float32),
        coords={"time": w_time_lon["time"], "lon": w_time_lon["lon"]},
        dims=("time", "lon"),
        name=f"{w_time_lon.name}_mjo_recon"
    )



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
    pc1 = ds3["pc1"].astype(float)
    pc2 = ds3["pc2"].astype(float)
    amp = ds3["amp"].astype(float)

    # --- load ERA5 w (latmean) ---
    dsw = xr.open_dataset(ERA5_W_LATMEAN, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))

    if "w_bp" not in dsw:
        raise RuntimeError("ERA5 w file missing variable: w_bp")
    if "level" not in dsw["w_bp"].dims:
        raise RuntimeError("w_bp has no 'level' dimension")
    if "lon" not in dsw["w_bp"].dims:
        raise RuntimeError("w_bp has no 'lon' dimension")
    if "time" not in dsw["w_bp"].dims:
        raise RuntimeError("w_bp has no 'time' dimension")

    # subset lon to tracking window and select two levels
    w = dsw["w_bp"].sel(lon=slice(TRACK_LON_MIN, TRACK_LON_MAX))
    # level 在 ERA5 是从 1000 -> 200 递减，slice(1000,700) 在 xarray 是 OK 的（按坐标选）
    w_low = w.sel(level=slice(LOW_LAYER[0], LOW_LAYER[1])).mean("level", skipna=True)
    w_up  = w.sel(level=slice(UP_LAYER[0],  UP_LAYER[1])).mean("level", skipna=True)

    # make sure dims are (time, lon)
    w_low = w_low.transpose("time", "lon")
    w_up  = w_up.transpose("time", "lon")

    # align time with Step3 (inner intersection)
    center_a, olr_center_a, pc1_a, pc2_a, amp_a, w_low_a, w_up_a = xr.align(
        center, olr_center, pc1, pc2, amp, w_low, w_up, join="inner"
    )
    time = pd.to_datetime(center_a["time"].values)
    winter = _winter_np(time)
    active = (olr_center_a.values.astype(float) <= OLR_MIN_THRESH) & np.isfinite(olr_center_a.values.astype(float))
    eventmask = _mask_event_days(time, EVENTS_CSV)

    lon = w_low_a["lon"].values.astype(float)

    # ==========================================================
    # NEW (requested): reconstruct w using pc1/pc2, then normalize by amp
    # ==========================================================
    w_low_mjo = _reconstruct_by_pc12_regression(w_low_a, pc1_a, pc2_a, winter)
    w_up_mjo  = _reconstruct_by_pc12_regression(w_up_a,  pc1_a, pc2_a, winter)

    amp_np_all = amp_a.values.astype(float)
    amp_ok = np.isfinite(amp_np_all) & (amp_np_all > AMP_EPS)

    # NEW: amp floor to prevent small-amp amplification
    amp_den = np.maximum(amp_np_all, float(AMP_FLOOR))

    # normalize (time,lon): w_mjo_norm = w_mjo / max(amp, AMP_FLOOR)
    w_low_mjo_np = w_low_mjo.values.astype(float)
    w_up_mjo_np  = w_up_mjo.values.astype(float)

    w_low_norm = np.full_like(w_low_mjo_np, np.nan, dtype=float)
    w_up_norm  = np.full_like(w_up_mjo_np,  np.nan, dtype=float)

    idx_amp = np.where(amp_ok)[0]
    if idx_amp.size > 0:
        w_low_norm[idx_amp, :] = w_low_mjo_np[idx_amp, :] / amp_den[idx_amp, None]
        w_up_norm[idx_amp, :]  = w_up_mjo_np[idx_amp, :]  / amp_den[idx_amp, None]

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

        # require amp-valid for normalization result
        if not amp_ok[i]:
            continue

        # build relative lon axis in tracking window (no wrap needed)
        rel = lon - float(c)

        # use reconstructed+normalized profiles
        wl = w_low_norm[i, :]
        wu = w_up_norm[i, :]

        lw, le, lc, lmin = _ascent_center_from_profile(
            rel, wl, ZERO_TOL, PIVOT_DELTA_DEG, EDGE_N_CONSEC, EDGE_PAD_DEG
        )
        uw, ue, uc, umin = _ascent_center_from_profile(
            rel, wu, ZERO_TOL, PIVOT_DELTA_DEG, EDGE_N_CONSEC, EDGE_PAD_DEG
        )

        low_west[i], low_east[i], low_ctr[i], low_wmin[i] = lw, le, lc, lmin
        up_west[i],  up_east[i],  up_ctr[i],  up_wmin[i]  = uw, ue, uc, umin

        # 用西边界定义 tilt：low_west - up_west
        if np.isfinite(lw) and np.isfinite(uw):
            tilt[i] = float(lw - uw)
        else:
            tilt[i] = np.nan

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
                         attrs={"desc": "tilt = low_west_rel - up_west_rel (deg)"}),
            "active_mask": xr.DataArray(active.astype(np.int8), coords={"time": time}, dims=("time",),
                            attrs={"desc": f"1 if olr_center_track <= {OLR_MIN_THRESH} else 0"}),
            "event_mask": xr.DataArray(eventmask.astype(np.int8), coords={"time": time}, dims=("time",),
                           attrs={"desc": "1 if within any Step3 event [start,end] else 0"}),
        },
        attrs={
            "source_step3": STEP3_NC,
            "source_w": ERA5_W_LATMEAN,
            "levels": f"low_layer={LOW_LAYER[0]}..{LOW_LAYER[1]}hPa, up_layer={UP_LAYER[0]}..{UP_LAYER[1]}hPa",
            "lon_window": f"{TRACK_LON_MIN}..{TRACK_LON_MAX}",
            "zero_tol": str(ZERO_TOL),
            "edge_n_consec": str(EDGE_N_CONSEC),
            "edge_pad_deg": str(EDGE_PAD_DEG),
            "pivot_delta_deg": str(PIVOT_DELTA_DEG),
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
