# -*- coding: utf-8 -*-
"""
tilt=中心经度差

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
TRACK_LON_MIN = 60.0
TRACK_LON_MAX = 180.0

# --- layer-mean definition (hPa) ---
LOW_LAYER = (1000.0, 700.0)   # inclusive slice: 1000..700
UP_LAYER  = (300.0, 200.0)    # inclusive slice: 300..200
# 备选：UP_LAYER = (500.0, 200.0)


# ---------- ZERO-crossing boundary ----------
ZERO_TOL = 0.005      # Pa/s（保留参数，不再用于边界判据）
EDGE_N_CONSEC = 2     # （保留参数，不再用于边界判据）
EDGE_PAD_DEG  = 2.5   # 边界留余量（度）

# pivot 仍可保留：用来锁定“从哪里向外扫”
PIVOT_DELTA_DEG = 10.0
MIN_VALID_POINTS = 7
OLR_MIN_THRESH = -15.0
ACTIVE_ONLY = True

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
    【修改点】把“东西边界”改成与你图上的黄/绿框一致的定义：

    - 先确定上升核（pivot）：在 [-pivot_delta, +pivot_delta] 内找 w 最小值（最强上升，w 最负）
    - 定义“上升区”掩膜：w < 0 （即 omega' 为负的区域）
    - 东西边界：取“包含 pivot 的那一段连续 w<0 区域”的左右端点
      （也就是 0 等值线围成的那一圈/那一块在该层平均剖面上的水平范围）

    参数 zero_tol / n_consec 保留以避免改动多余部分，但这里不再用于判据。
    Returns: (west_rel, east_rel, center_rel, wmin)
    """
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return (np.nan, np.nan, np.nan, np.nan)

    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)

    # --- pivot：0附近最强上升点（最小 w） ---
    win = (rr >= -pivot_delta) & (rr <= pivot_delta)
    if win.any():
        j0 = int(np.nanargmin(ww[win]))
        pivot_idx = np.where(win)[0][j0]
    else:
        pivot_idx = int(np.nanargmin(ww))

    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0.0):
        # pivot 都不为负，说明没有可靠“上升区( w<0 )”
        return (np.nan, np.nan, np.nan, wmin)

    # 上升区定义：w < 0 （对应你图上的黄/绿框）
    asc = ww < 0.0
    if not asc[pivot_idx]:
        return (np.nan, np.nan, np.nan, wmin)

    # 向西扩展到连续 asc 的边界
    j0 = pivot_idx
    while (j0 - 1) >= 0 and asc[j0 - 1]:
        j0 -= 1

    # 向东扩展到连续 asc 的边界
    j1 = pivot_idx
    while (j1 + 1) < asc.size and asc[j1 + 1]:
        j1 += 1

    west = float(rr[j0])
    east = float(rr[j1])

    # 防御：异常情况
    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return (np.nan, np.nan, np.nan, wmin)

    # padding + clip（保留你原来的行为）
    rr_min = float(np.nanmin(rr))
    rr_max = float(np.nanmax(rr))
    west = max(rr_min, west - float(pad))
    east = min(rr_max, east + float(pad))

    center = 0.5 * (west + east)
    return (west, east, center, wmin)


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
    center_a, olr_center_a, w_low_a, w_up_a = xr.align(center, olr_center, w_low, w_up, join="inner")
    time = pd.to_datetime(center_a["time"].values)
    winter = _winter_np(time)
    active = (olr_center_a.values.astype(float) <= OLR_MIN_THRESH) & np.isfinite(olr_center_a.values.astype(float))
    eventmask = _mask_event_days(time, EVENTS_CSV)

    lon = w_low_a["lon"].values.astype(float)

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
    east_diff = np.full(n, np.nan, np.float32)  # low_east - up_east
    west_diff = np.full(n, np.nan, np.float32)  # low_west - up_west
    abs_low_east = np.full(n, np.nan, np.float32)  # abs lon of low east boundary
    abs_up_east  = np.full(n, np.nan, np.float32)  # abs lon of up  east boundary
    abs_east_diff = np.full(n, np.nan, np.float32) # abs_low_east - abs_up_east
    abs_low_west = np.full(n, np.nan, np.float32)  # abs lon of low west boundary
    abs_up_west  = np.full(n, np.nan, np.float32)  # abs lon of up  west boundary
    abs_west_diff = np.full(n, np.nan, np.float32) # abs_low_west - abs_up_west

    # load arrays (faster than per-day isel)
    c_np = center_a.values.astype(float)
    wl_np = w_low_a.values.astype(float)  # (time, lon)
    wu_np = w_up_a.values.astype(float)

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

        # build relative lon axis in tracking window (no wrap needed)
        rel = lon - float(c)

        wl = wl_np[i, :]
        wu = wu_np[i, :]

        lw, le, lc, lmin = _ascent_center_from_profile(
            rel, wl, ZERO_TOL, PIVOT_DELTA_DEG, EDGE_N_CONSEC, EDGE_PAD_DEG
        )
        uw, ue, uc, umin = _ascent_center_from_profile(
            rel, wu, ZERO_TOL, PIVOT_DELTA_DEG, EDGE_N_CONSEC, EDGE_PAD_DEG
        )
        # 如果你想要绝对经度（未 wrap）：abs = rel + c
        abs_le = float(le + c)
        abs_ue = float(ue + c)
        # 如果你想要绝对经度（未 wrap）：abs = rel + c
        abs_lw = float(lw + c)
        abs_uw = float(uw + c)
        low_west[i], low_east[i], low_ctr[i], low_wmin[i] = lw, le, lc, lmin
        up_west[i],  up_east[i],  up_ctr[i],  up_wmin[i]  = uw, ue, uc, umin

        if np.isfinite(lc) and np.isfinite(uc):
            tilt[i] = float(lc - uc)  # low_center - up_center (deg)
            
        if np.isfinite(le) and np.isfinite(ue):
            east_diff[i] = float(le - ue)  # rel diff
            # absolute lon (same as rel + c). Note: this is not wrapped to 0..360
            abs_low_east[i] = abs_le
            abs_up_east[i]  = abs_ue
            abs_east_diff[i] = float(abs_le - abs_ue)

        if np.isfinite(lw) and np.isfinite(uw):
            west_diff[i] = float(lw - uw)  # rel diff
            # absolute lon (same as rel + c). Note: this is not wrapped to 0..360
            abs_low_west[i] = abs_lw
            abs_up_west[i]  = abs_uw
            abs_west_diff[i] = float(abs_lw - abs_uw)

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
                                 attrs={"desc": "tilt = low_center_rel - up_center_rel (deg)"}),
            "east_diff": xr.DataArray(east_diff, coords={"time": time}, dims=("time",),
                                      attrs={"desc": "east_diff = low_east_rel - up_east_rel (deg)"}),
            "west_diff": xr.DataArray(west_diff, coords={"time": time}, dims=("time",),
                                      attrs={"desc": "west_diff = low_west_rel - up_west_rel (deg)"}),
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
              "median", float(np.nanpercentile(tv, 50)), "p95", float(np.nanpercentile(tv, 95)),
              "max", float(np.nanmax(tv)))
    # mean of boundary diffs
    ed = ds_out["east_diff"].values.astype(float)
    wd = ds_out["west_diff"].values.astype(float)

    ed_ok = np.isfinite(ed)
    wd_ok = np.isfinite(wd)

    print("east_diff_rel finite days:", int(ed_ok.sum()), "/", int(ed.size))
    if ed_ok.any():
        print("east_diff_rel mean (deg):", float(np.nanmean(ed)))

    print("west_diff_rel finite days:", int(wd_ok.sum()), "/", int(wd.size))
    if wd_ok.any():
        print("west_diff_rel mean (deg):", float(np.nanmean(wd)))
        
    # mean absolute east boundaries (and their diff)
    ale = abs_low_east.astype(float)
    aue = abs_up_east.astype(float)
    aed = abs_east_diff.astype(float)

    print("abs_low_east finite days:", int(np.isfinite(ale).sum()), "/", int(ale.size))
    if np.isfinite(ale).any():
        print("abs_low_east mean (degE):", float(np.nanmean(ale)))

    print("abs_up_east finite days:", int(np.isfinite(aue).sum()), "/", int(aue.size))
    if np.isfinite(aue).any():
        print("abs_up_east mean (degE):", float(np.nanmean(aue)))

    print("abs_east_diff finite days:", int(np.isfinite(aed).sum()), "/", int(aed.size))
    if np.isfinite(aed).any():
        print("abs_low_east - abs_up_east mean (deg):", float(np.nanmean(aed)))
    
    # mean absolute west boundaries (and their diff)
    ale = abs_low_west.astype(float)
    aue = abs_up_west.astype(float)
    aed = abs_west_diff.astype(float)

    print("abs_low_west finite days:", int(np.isfinite(ale).sum()), "/", int(ale.size))
    if np.isfinite(ale).any():
        print("abs_low_west mean (degE):", float(np.nanmean(ale)))

    print("abs_up_west finite days:", int(np.isfinite(aue).sum()), "/", int(aue.size))
    if np.isfinite(aue).any():
        print("abs_up_west mean (degE):", float(np.nanmean(aue)))

    print("abs_west_diff finite days:", int(np.isfinite(aed).sum()), "/", int(aed.size))
    if np.isfinite(aed).any():
        print("abs_low_west - abs_up_west mean (deg):", float(np.nanmean(aed)))

if __name__ == "__main__":
    main()
