# -*- coding: utf-8 -*-
"""
03b_compute_tilt_q.py: 逐日 MJO Tilt 指数计算（q 最大值定义下层）

================================================================================
功能描述：
    基于 ERA5 MJO 重构水汽场（q）和 omega 场，计算逐日 MJO 垂直倾斜指数。

新 Tilt 定义：
    Tilt_q = q 低层平均最大值经度 - omega 高层上升区西边界经度（相对经度，单位：°）

    - 上层边界（不变）：omega 400-200 hPa 层平均上升区西边界
    - 下层边界（新）：q 1000-850 hPa 层平均最大值经度位置

输入数据：
    - Step3 输出：center_lon_track（对流中心轨迹）
    - MJO 重构归一化 omega：era5_mjo_recon_w_norm_1979-2022.nc
    - MJO 重构归一化 q：era5_mjo_recon_q_norm_1979-2022.nc

输出：
    - tilt_q_daily_1979-2022.nc
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.interpolate import Akima1DInterpolator

# ======================
# USER PATHS
# ======================
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
W_NORM_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
Q_NORM_NC = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
OUT_NC = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"


# ======================
# SETTINGS
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
TRACK_LON_MIN = 0.0
TRACK_LON_MAX = 240.0

# --- 层次定义 (hPa) ---
Q_LOW_LAYER = (1000.0, 850.0)    # q 低层：1000-850 hPa
UP_LAYER    = (400.0, 200.0)     # omega 高层：400-200 hPa（不变）
PRESSURE_WEIGHTED = False

# --- omega 上层边界检测参数（沿用 03_compute_tilt_daily.py） ---
HALF_MAX_FRACTION = 0.0
EDGE_N_CONSEC = 1
SMOOTH_WINDOW = 10
PIVOT_DELTA_DEG = 10.0

# --- 插值参数 ---
CSA_KNOTS = 9
CSA_TARGET_DLON = 0.25

MIN_VALID_POINTS = 7
OLR_MIN_THRESH = -15.0
ACTIVE_ONLY = False

AMP_EPS = 1e-6
AMP_FLOOR = 1.0


# ======================
# helpers（复用自 03_compute_tilt_daily.py）
# ======================
def _pressure_weighted_mean(da: xr.DataArray, layer_bounds: tuple) -> xr.DataArray:
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
    """沿经度维做 Akima 三次样条插值。"""
    src_lon = da["lon"].values.astype(float)
    lon_min, lon_max = float(src_lon.min()), float(src_lon.max())
    n_target = int(round((lon_max - lon_min) / target_dlon)) + 1
    target_lon = np.linspace(lon_min, lon_max, n_target)

    n_src = len(src_lon)
    M = np.zeros((len(target_lon), n_src))
    for j in range(n_src):
        e_j = np.zeros(n_src)
        e_j[j] = 1.0
        M[:, j] = Akima1DInterpolator(src_lon, e_j)(target_lon)

    data = da.values
    orig_shape = data.shape
    flat = data.reshape(-1, orig_shape[-1])
    out_flat = flat @ M.T
    out = out_flat.reshape(*orig_shape[:-1], len(target_lon))

    new_coords = {d: (target_lon if d == "lon" else da[d].values)
                  for d in da.dims}
    result = xr.DataArray(out, dims=da.dims, coords=new_coords)
    result["lon"].attrs["units"] = "degrees_east"
    return result


def _ascent_boundary_by_half_max(
    rel_lon: np.ndarray,
    w: np.ndarray,
    half_max_fraction: float = 0.5,
    pivot_delta: float = 10.0,
    n_consec: int = 1
):
    """使用半高宽法定义 omega 上升区边界。"""
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return (np.nan, np.nan, np.nan, np.nan)

    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)

    pivot_idx = int(np.argmin(np.abs(rr)))
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return (np.nan, np.nan, np.nan, wmin)

    thr = float(half_max_fraction) * wmin

    # west edge
    outside = 0
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:
            outside += 1
        else:
            outside = 0
        if outside >= n_consec:
            cand = i + n_consec
            cand = min(cand, pivot_idx)
            west_idx = cand
            break

    if west_idx is None:
        return (np.nan, np.nan, np.nan, wmin)

    # east edge
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
        return (np.nan, np.nan, np.nan, wmin)

    west = float(rr[west_idx])
    east = float(rr[east_idx])

    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return (np.nan, np.nan, np.nan, wmin)

    center = 0.5 * (west + east)
    return (west, east, center, wmin)


Q_MAX_SEARCH_MIN = 0.0    # q 最大值搜索范围：相对经度下界（°）
Q_MAX_SEARCH_MAX = 50.0   # q 最大值搜索范围：相对经度上界（°）


def _find_q_max_position(rel_lon: np.ndarray, q_profile: np.ndarray) -> tuple:
    """
    在 q 低层平均剖面中找到最大值位置。
    搜索范围限制在相对经度 [Q_MAX_SEARCH_MIN, Q_MAX_SEARCH_MAX] 内。

    Returns: (q_max_rel_lon, q_max_value)
    """
    m = np.isfinite(q_profile) & np.isfinite(rel_lon)
    # 额外限制搜索范围
    m = m & (rel_lon >= Q_MAX_SEARCH_MIN) & (rel_lon <= Q_MAX_SEARCH_MAX)
    if m.sum() < MIN_VALID_POINTS:
        return (np.nan, np.nan)

    rr = rel_lon[m].astype(float)
    qq = q_profile[m].astype(float)

    max_idx = int(np.argmax(qq))
    q_max_rel = float(rr[max_idx])
    q_max_val = float(qq[max_idx])

    return (q_max_rel, q_max_val)


def main():
    out_path = Path(OUT_NC)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- load Step3 (track) ---
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    if "center_lon_track" not in ds3:
        raise RuntimeError("Step3 nc missing variable: center_lon_track")

    t3 = pd.to_datetime(ds3["time"].values)
    center = ds3["center_lon_track"].astype(float)
    olr_center = ds3["olr_center_track"].astype(float)
    amp = ds3["amp"].astype(float)

    # --- load normalized omega ---
    print("Loading normalized omega...")
    dsw = xr.open_dataset(W_NORM_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    w_var = "w_mjo_recon_norm"
    if w_var not in dsw:
        raise RuntimeError(f"Normalized omega file missing variable: {w_var}")
    w = dsw[w_var]
    if "pressure_level" in w.dims:
        w = w.rename({"pressure_level": "level"})
    lon_vals = w["lon"].values
    if lon_vals.min() < 0:
        new_lon = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
        w = w.assign_coords(lon=new_lon).sortby("lon")

    # --- load normalized q ---
    print("Loading normalized q...")
    dsq = xr.open_dataset(Q_NORM_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    q_var = "q_mjo_recon_norm"
    if q_var not in dsq:
        raise RuntimeError(f"Normalized q file missing variable: {q_var}")
    q = dsq[q_var]
    if "pressure_level" in q.dims:
        q = q.rename({"pressure_level": "level"})
    lon_vals_q = q["lon"].values
    if lon_vals_q.min() < 0:
        new_lon_q = np.where(lon_vals_q < 0, lon_vals_q + 360, lon_vals_q)
        q = q.assign_coords(lon=new_lon_q).sortby("lon")

    # --- 对 omega 做三次样条插值 + 平滑 ---
    print(f"  Omega: cubic spline interpolation (target dlon={CSA_TARGET_DLON}°)...")
    w = _cubic_spline_approx_lon(w, CSA_KNOTS, CSA_TARGET_DLON)
    print(f"  After interpolation: lon {w['lon'].shape}")

    if SMOOTH_WINDOW > 1:
        print(f"  Omega: applying per-level sliding average (window={SMOOTH_WINDOW})...")
        w_vals = w.values
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

    # --- 对 q 做三次样条插值 + 平滑 ---
    print(f"  Q: cubic spline interpolation (target dlon={CSA_TARGET_DLON}°)...")
    q = _cubic_spline_approx_lon(q, CSA_KNOTS, CSA_TARGET_DLON)
    print(f"  After interpolation: lon {q['lon'].shape}")

    if SMOOTH_WINDOW > 1:
        print(f"  Q: applying per-level sliding average (window={SMOOTH_WINDOW})...")
        q_vals = q.values
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        for t in range(q_vals.shape[0]):
            for k in range(q_vals.shape[1]):
                profile = q_vals[t, k, :]
                valid = np.isfinite(profile).astype(float)
                filled = np.where(np.isfinite(profile), profile, 0.0)
                smoothed = np.convolve(filled, kernel, mode='same')
                count = np.convolve(valid, kernel, mode='same')
                count[count < 1e-10] = np.nan
                q_vals[t, k, :] = smoothed / count
        q = xr.DataArray(q_vals, dims=q.dims, coords=q.coords)

    # --- 层平均 ---
    # omega 高层：400-200 hPa
    w_up_sel = w.sel(level=slice(UP_LAYER[0], UP_LAYER[1]))
    if PRESSURE_WEIGHTED:
        w_up = _pressure_weighted_mean(w_up_sel, UP_LAYER)
    else:
        w_up = w_up_sel.mean("level", skipna=True)

    # q 低层：1000-850 hPa
    q_low_sel = q.sel(level=slice(Q_LOW_LAYER[0], Q_LOW_LAYER[1]))
    if PRESSURE_WEIGHTED:
        q_low = _pressure_weighted_mean(q_low_sel, Q_LOW_LAYER)
    else:
        q_low = q_low_sel.mean("level", skipna=True)

    # make sure dims are (time, lon)
    w_up = w_up.transpose("time", "lon")
    q_low = q_low.transpose("time", "lon")

    # subset lon to tracking window
    w_up = w_up.sel(lon=slice(TRACK_LON_MIN, TRACK_LON_MAX))
    q_low = q_low.sel(lon=slice(TRACK_LON_MIN, TRACK_LON_MAX))

    print(f"  w_up shape: {w_up.shape}, q_low shape: {q_low.shape}")

    # align time
    center_a, olr_center_a, amp_a, w_up_a, q_low_a = xr.align(
        center, olr_center, amp, w_up, q_low, join="inner"
    )
    time = pd.to_datetime(center_a["time"].values)
    winter = _winter_np(time)
    active = (olr_center_a.values.astype(float) <= OLR_MIN_THRESH) & np.isfinite(olr_center_a.values.astype(float))
    eventmask = _mask_event_days(time, EVENTS_CSV)

    lon = w_up_a["lon"].values.astype(float)
    w_up_norm = w_up_a.values.astype(float)
    q_low_norm = q_low_a.values.astype(float)

    # pre-allocate
    n = time.size
    q_max_rel = np.full(n, np.nan, np.float32)
    q_max_val = np.full(n, np.nan, np.float32)

    up_west = np.full(n, np.nan, np.float32)
    up_east = np.full(n, np.nan, np.float32)
    up_ctr  = np.full(n, np.nan, np.float32)
    up_wmin = np.full(n, np.nan, np.float32)

    tilt_q = np.full(n, np.nan, np.float32)

    c_np = center_a.values.astype(float)

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

        rel = lon - float(c)

        # omega 高层边界
        wu = w_up_norm[i, :]
        uw, ue, uc, umin = _ascent_boundary_by_half_max(
            rel, wu, HALF_MAX_FRACTION, PIVOT_DELTA_DEG, EDGE_N_CONSEC
        )
        up_west[i], up_east[i], up_ctr[i], up_wmin[i] = uw, ue, uc, umin

        # q 低层最大值位置
        ql = q_low_norm[i, :]
        qm_rel, qm_val = _find_q_max_position(rel, ql)
        q_max_rel[i] = qm_rel
        q_max_val[i] = qm_val

        # 新 tilt：q_max_rel - up_west
        if np.isfinite(qm_rel) and np.isfinite(uw):
            tilt_q[i] = float(qm_rel - uw)
        else:
            tilt_q[i] = np.nan

    # --- save ---
    ds_out = xr.Dataset(
        {
            "q_max_rel": xr.DataArray(q_max_rel, coords={"time": time}, dims=("time",),
                          attrs={"desc": "q 低层平均最大值相对经度 (deg)"}),
            "q_max_value": xr.DataArray(q_max_val, coords={"time": time}, dims=("time",),
                            attrs={"desc": "q 低层平均最大值"}),
            "up_west_rel": xr.DataArray(up_west, coords={"time": time}, dims=("time",),
                            attrs={"desc": "omega 高层上升区西边界相对经度 (deg)"}),
            "up_east_rel": xr.DataArray(up_east, coords={"time": time}, dims=("time",),
                            attrs={"desc": "omega 高层上升区东边界相对经度 (deg)"}),
            "up_center_rel": xr.DataArray(up_ctr, coords={"time": time}, dims=("time",)),
            "up_wmin": xr.DataArray(up_wmin, coords={"time": time}, dims=("time",)),
            "tilt_q": xr.DataArray(tilt_q, coords={"time": time}, dims=("time",),
                       attrs={"desc": "tilt_q = q_max_rel - up_west_rel (deg), 水汽最大值定义下层"}),
            "active_mask": xr.DataArray(active.astype(np.int8), coords={"time": time}, dims=("time",),
                            attrs={"desc": f"1 if olr_center_track <= {OLR_MIN_THRESH} else 0"}),
            "event_mask": xr.DataArray(eventmask.astype(np.int8), coords={"time": time}, dims=("time",),
                           attrs={"desc": "1 if within any Step3 event [start,end] else 0"}),
        },
        attrs={
            "source_step3": STEP3_NC,
            "source_w": W_NORM_NC,
            "source_q": Q_NORM_NC,
            "levels": f"q_low_layer={Q_LOW_LAYER[0]}..{Q_LOW_LAYER[1]}hPa, up_layer={UP_LAYER[0]}..{UP_LAYER[1]}hPa",
            "lon_window": f"{TRACK_LON_MIN}..{TRACK_LON_MAX}",
            "layer_mean_method": "pressure_weighted" if PRESSURE_WEIGHTED else "equal_weight",
            "boundary_method_up": "half_max_fwhm (omega)",
            "boundary_method_low": "q max position",
            "half_max_fraction": str(HALF_MAX_FRACTION),
            "smooth_window": str(SMOOTH_WINDOW),
            "winter_months": ",".join(map(str, sorted(WINTER_MONTHS))),
            "time_range": f"{START_DATE}..{END_DATE}",
            "active_only": str(ACTIVE_ONLY),
        }
    )

    enc = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    enc["time"] = {"zlib": False, "_FillValue": None}
    ds_out.to_netcdf(out_path, engine="netcdf4", encoding=enc)

    # quick console summary
    tv = ds_out["tilt_q"].values.astype(float)
    ok = np.isfinite(tv)
    print("Saved:", str(out_path))
    print("tilt_q finite days:", int(ok.sum()), "/", int(tv.size),
          "winter:", int(winter.sum()))
    if ok.any():
        print("tilt_q stats (deg): min", float(np.nanmin(tv)), "p5", float(np.nanpercentile(tv, 5)),
              "mean", float(np.nanmean(tv)), "median", float(np.nanpercentile(tv, 50)),
              "p95", float(np.nanpercentile(tv, 95)), "max", float(np.nanmax(tv)))

    # --- q_max_rel summary ---
    qr = ds_out["q_max_rel"].values.astype(float)
    qok = np.isfinite(qr)
    if qok.any():
        print("q_max_rel stats (deg): min", float(np.nanmin(qr)), "p5", float(np.nanpercentile(qr, 5)),
              "mean", float(np.nanmean(qr)), "median", float(np.nanpercentile(qr, 50)),
              "p95", float(np.nanpercentile(qr, 95)), "max", float(np.nanmax(qr)))


if __name__ == "__main__":
    main()
