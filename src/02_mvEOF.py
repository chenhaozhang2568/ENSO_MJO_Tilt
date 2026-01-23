# -*- coding: utf-8 -*-
"""
E:\Projects\ENSO_MJO_Tilt\src\02_mvEOF_new2.py

FIXES from 02_mvEOF_new.py:
1. Latitudinal Mean BEFORE EOF: Apply meridional mean (15S-15N) to get (Time, Lon)
   before MV-EOF, following Wheeler & Hendon (2004) methodology. This improves
   signal-to-noise ratio and focuses on zonal propagation.
2. Filter Edge Trimming: Remove first/last 60 days (Lanczos edge NaNs).
3. Smart Bad Longitude Detection: Only remove longitudes with >50% NaN fraction.
4. OLR Reconstruction: Use raw anomaly regression for proper amplitude.
5. Restored threshold: OLR_MIN_THRESH = -15.0 (for raw anomaly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ======================
# INPUTS (from Step2)
# ======================
OLR_BP_PATH = r"E:\Datas\ClimateIndex\processed\olr_bp_1979-2022.nc"
U_BP_PATH   = r"E:\Datas\ERA5\processed\pressure_level\era5_u850_u200_bp_1979-2022.nc"

# ======================
# OUTPUTS
# ======================
OUT_DIR = Path(r"E:\Datas\Derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# TIME RANGE
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"

# ======================
# PAPER-STYLE SETTINGS
# ======================
LAT_BAND = (-15.0, 15.0)
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
TRACK_LON_MIN = 60.0
TRACK_LON_MAX = 180.0
MAX_WEST_JUMP_DEG = 5.0

# ======================
# EVENT CRITERIA
# ======================
OLR_MIN_THRESH = -15.0
MIN_EAST_DISP_DEG = 50.0
AMP_THRESH = None

IO_GATE_LON = 90.0
MC_GATE_LON = 110.0

DP_JUMP_WEIGHT = 0.008
DP_NEAR_MIN_DELTA = 5.0
DP_MAX_CANDS = 10
TRACK_GAP_DAYS = 2
MIN_ACTIVE_DAYS_IN_EVENT = 5

# Filter edge trim (Lanczos 121-day window = 60 samples NaN at each edge)
EDGE_TRIM = 60

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
    if lat.size >= 2 and (lat[1] - lat[0]) < 0:
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


def _drop_bad_longitudes(da_list: list, nan_threshold: float = 0.5) -> list:
    """
    Remove longitude columns where NaN fraction > threshold across time in ANY variable.
    """
    if not da_list:
        return da_list
    
    ref = da_list[0]
    T = ref.sizes["time"]
    print(f"[Clean] Checking for bad longitudes (NaN threshold > {nan_threshold*100:.0f}%)...")
    
    bad_mask = xr.zeros_like(ref.isel(time=0), dtype=bool)
    
    for da in da_list:
        nan_frac = np.isnan(da).sum("time") / T
        if hasattr(nan_frac.data, "compute"):
            nan_frac = nan_frac.compute()
        bad_mask = bad_mask | (nan_frac > nan_threshold)
    
    if hasattr(bad_mask.data, "compute"):
        bad_mask = bad_mask.compute()
    
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        print(f"[WARNING] Dropping {n_bad} bad longitudes (NaN fraction > {nan_threshold*100:.0f}%)")
        good_mask = ~bad_mask
        return [da.isel(lon=good_mask.values) for da in da_list]
    else:
        print("  No bad longitudes found.")
        return da_list


def _drop_nan_training_days(olr_tr: xr.DataArray, u850_tr: xr.DataArray, u200_tr: xr.DataArray):
    """Drop days with any NaN across longitude (after lat averaging, dims are (time, lon))."""
    good = (
        np.isfinite(olr_tr).all("lon")
        & np.isfinite(u850_tr).all("lon")
        & np.isfinite(u200_tr).all("lon")
    ).fillna(False)

    if hasattr(good.data, "compute"):
        good = good.compute()

    idx = np.nonzero(good.values)[0]
    return olr_tr.isel(time=idx), u850_tr.isel(time=idx), u200_tr.isel(time=idx), good


# ======================
# TRACKING HELPERS
# ======================
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
    """DP tracking to obtain continuous daily center trajectory."""
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

        t0 = t
        t1 = t
        while t1 + 1 < T:
            yy = dom.isel(time=t1 + 1).values.astype(float)
            if not np.isfinite(yy).any():
                break
            t1 += 1

        cand_lon_list = []
        cand_val_list = []

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

        D = t1 - t0 + 1
        dp = [np.full(len(cand_val_list[d]), np.inf, dtype=float) for d in range(D)]
        bp = [np.full(len(cand_val_list[d]), -1, dtype=int) for d in range(D)]
        dp[0] = cand_val_list[0].copy()

        for d in range(1, D):
            vals = cand_val_list[d]
            lls = cand_lon_list[d]
            prev_lls = cand_lon_list[d - 1]
            for j in range(len(vals)):
                best_cost = np.inf
                best_k = -1
                for k in range(len(cand_val_list[d-1])):
                    jump = lls[j] - prev_lls[k]
                    if jump < -max_west_jump:
                        continue
                    cost = dp[d-1][k] + vals[j] + DP_JUMP_WEIGHT * abs(jump)
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


def _sample_olr_at_track_center(recon_latmean: xr.DataArray, center_track: np.ndarray) -> np.ndarray:
    lons = recon_latmean["lon"].values.astype(float)
    y2d = recon_latmean.values.astype(float)
    T = y2d.shape[0]
    out = np.full(T, np.nan, dtype=float)
    for t in range(T):
        c = center_track[t]
        if np.isfinite(c):
            j = int(np.argmin(np.abs(lons - float(c))))
            out[t] = y2d[t, j]
    return out


def _contour_track_from_threshold(recon_latmean: xr.DataArray, thr: float):
    rr = recon_latmean.values.astype(float)
    lon = recon_latmean["lon"].values.astype(float)
    T, L = rr.shape
    west = np.full(T, np.nan, dtype=float)
    east = np.full(T, np.nan, dtype=float)
    cent = np.full(T, np.nan, dtype=float)

    for t in range(T):
        y = rr[t, :]
        if not np.isfinite(y).any():
            continue
        active = np.isfinite(y) & (y <= thr)
        if not active.any():
            continue
        idx = np.where(active)[0]
        splits = np.where(np.diff(idx) > 1)[0]
        seg_starts = np.r_[0, splits + 1]
        seg_ends = np.r_[splits, len(idx) - 1]
        best_seg = None
        best_min = np.inf
        for s0, s1 in zip(seg_starts, seg_ends):
            seg_idx = idx[s0:s1+1]
            seg_min = float(np.nanmin(y[seg_idx]))
            if seg_min < best_min:
                best_min = seg_min
                best_seg = seg_idx
        if best_seg is not None:
            west[t] = float(lon[int(best_seg[0])])
            east[t] = float(lon[int(best_seg[-1])])
            cent[t] = float(np.nanmean(lon[best_seg]))
    return west, east, cent


def _build_events_by_contour_track(time, contour_track, active_flag, amp, winter_mask):
    n = len(time)
    active_day = active_flag.astype(bool) & winter_mask.astype(bool)
    if AMP_THRESH is not None and amp is not None:
        active_day = active_day & np.isfinite(amp) & (amp >= AMP_THRESH)

    events = []
    failed_events = []
    eid = 0
    fid = 0
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

    def finalize_segment(s0, s1):
        nonlocal eid, fid, events, failed_events
        nonlocal fail_disp, fail_gate, fail_active, fail_short
        if s0 < 0 or s1 < s0:
            return
        seg_lons = contour_track[s0:s1+1].astype(float)
        seg_ok = np.isfinite(seg_lons)
        if not seg_ok.any():
            fail_short += 1
            return
        lons_valid = seg_lons[seg_ok]
        lon_min = float(np.nanmin(lons_valid))
        lon_max = float(np.nanmax(lons_valid))
        disp = lon_max - lon_min
        passed_gates = (lon_min <= IO_GATE_LON) and (lon_max >= MC_GATE_LON)
        seg_active = active_day[s0:s1+1]
        n_active = int(np.nansum(seg_active.astype(int)))

        idxs = np.arange(s0, s1+1)
        valid_idxs = idxs[np.isfinite(contour_track[s0:s1+1])]
        lon_start = float(contour_track[valid_idxs[0]])
        lon_end = float(contour_track[valid_idxs[-1]])

        event_info = {
            "start_date": time[s0].strftime("%Y-%m-%d"),
            "end_date": time[s1].strftime("%Y-%m-%d"),
            "duration_days": int(s1 - s0 + 1),
            "lon_start": lon_start,
            "lon_end": lon_end,
            "east_displacement_deg": float(disp),
            "active_days": int(n_active),
            "amp_mean": float(np.nanmean(amp[valid_idxs])) if (amp is not None) else np.nan,
        }

        failure_reason = None
        if n_active < MIN_ACTIVE_DAYS_IN_EVENT:
            fail_active += 1
            failure_reason = f"insufficient_active_days (n={n_active})"
        elif disp < MIN_EAST_DISP_DEG:
            fail_disp += 1
            failure_reason = f"insufficient_displacement (disp={disp:.1f})"
        elif not passed_gates:
            fail_gate += 1
            failure_reason = "failed_gate_passage"

        if failure_reason is not None:
            fid += 1
            failed_events.append({"failed_event_id": fid, **event_info, "failure_reason": failure_reason})
        else:
            eid += 1
            events.append({"event_id": eid, **event_info})

    for t in range(n):
        if not winter_mask[t]:
            if in_seg:
                finalize_segment(seg_start, seg_end)
            in_seg = False
            seg_start = seg_end = -1
            nan_gap = 0
            prev_valid_lon = np.nan
            continue

        lon_t = contour_track[t]
        if (not np.isfinite(lon_t)) or (not active_day[t]):
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

        jump = lon_t - float(prev_valid_lon)
        if jump < -MAX_WEST_JUMP_DEG:
            fail_west += 1
            finalize_segment(seg_start, seg_end)
            in_seg = True
            seg_start = t
            seg_end = t
            nan_gap = 0
            prev_valid_lon = lon_t
            continue

        seg_end = t
        nan_gap = 0
        prev_valid_lon = lon_t

    if in_seg:
        finalize_segment(seg_start, seg_end)

    print("[Event diagnostics - contour-based]")
    print(f"  fail_west: {fail_west}, fail_short: {fail_short}, fail_active: {fail_active}")
    print(f"  fail_gate: {fail_gate}, fail_disp: {fail_disp}")
    print(f"  pass_events: {eid}, failed_events: {fid}")
    return pd.DataFrame(events), pd.DataFrame(failed_events)


# ======================
# MAIN
# ======================
def main():
    print("[COMPUTE] Starting EOF calculation (v2 - Latitudinal Mean Fix)...")

    # ---- 1. Load Data ----
    ds_olr = _to_lon_180(_rename_latlon_if_needed(_open_ds(OLR_BP_PATH)))
    ds_u   = _to_lon_180(_rename_latlon_if_needed(_open_ds(U_BP_PATH)))

    # ---- 2. Interp ERA5 -> OLR grid ----
    print(f"[PRE-PROCESS] Interpolating ERA5 -> OLR (2.5°)...")
    print(f"  Original ERA5 shape: {ds_u['u850_bp'].shape}")
    print(f"  Target OLR shape: {ds_olr['olr_bp'].shape}")
    ds_u = ds_u.interp(lat=ds_olr["lat"], lon=ds_olr["lon"], method="linear")
    print(f"  Interpolated ERA5 shape: {ds_u['u850_bp'].shape}")

    # ---- 3. Time Subset & Align ----
    ds_olr = ds_olr.sel(time=slice(START_DATE, END_DATE))
    ds_u   = ds_u.sel(time=slice(START_DATE, END_DATE))
    ds_olr, ds_u = xr.align(ds_olr, ds_u, join="inner")

    print(f"OLR time: {ds_olr['time'].values[0]} to {ds_olr['time'].values[-1]}, n={ds_olr.sizes['time']}")

    # ---- 4. Select Lat Band ----
    ds_olr = _sel_lat_band(ds_olr, LAT_BAND)
    ds_u   = _sel_lat_band(ds_u, LAT_BAND)

    # ============================================================
    # CRITICAL FIX: Meridional Mean BEFORE EOF (Wheeler & Hendon 2004)
    # This creates (Time, Lon) 2D matrices for MV-EOF
    # ============================================================
    print("[MOD] Applying meridional mean (15S-15N) BEFORE EOF...")
    
    olr_bp_2d = ds_olr["olr_bp"].mean("lat", skipna=True).transpose("time", "lon")
    u850_bp_2d = ds_u["u850_bp"].mean("lat", skipna=True).transpose("time", "lon")
    u200_bp_2d = ds_u["u200_bp"].mean("lat", skipna=True).transpose("time", "lon")
    
    # Load raw anomaly for regression reconstruction
    if "olr_anom" in ds_olr:
        olr_raw_anom = ds_olr["olr_anom"].mean("lat", skipna=True).transpose("time", "lon")
    else:
        raise KeyError("Input file missing 'olr_anom'. Please re-run Step 1 (01_lanczos_bandpass_new2.py).")
    
    print(f"  After lat-mean: OLR shape = {olr_bp_2d.shape}, U850 shape = {u850_bp_2d.shape}")

    # ============================================================
    # CRITICAL FIX 1: Remove filter edge times (first/last 60 days)
    # ============================================================
    print(f"[Clean] Removing filter edge times (first/last {EDGE_TRIM} days)...")
    olr_bp_2d = olr_bp_2d.isel(time=slice(EDGE_TRIM, -EDGE_TRIM))
    u850_bp_2d = u850_bp_2d.isel(time=slice(EDGE_TRIM, -EDGE_TRIM))
    u200_bp_2d = u200_bp_2d.isel(time=slice(EDGE_TRIM, -EDGE_TRIM))
    olr_raw_anom = olr_raw_anom.isel(time=slice(EDGE_TRIM, -EDGE_TRIM))
    print(f"  Time range after trimming: {olr_bp_2d.sizes['time']} days")

    # ============================================================
    # CRITICAL FIX 2: Remove bad longitudes BEFORE dropping days
    # ============================================================
    vars_to_clean = [olr_bp_2d, u850_bp_2d, u200_bp_2d, olr_raw_anom]
    cleaned_vars = _drop_bad_longitudes(vars_to_clean)
    olr_bp_2d = cleaned_vars[0]
    u850_bp_2d = cleaned_vars[1]
    u200_bp_2d = cleaned_vars[2]
    olr_raw_anom = cleaned_vars[3]
    print(f"  Grid size after cleaning: {olr_bp_2d.sizes['lon']} longitudes")

    # ---- 5. Winter mask & training subset ----
    wmask = _winter_mask(olr_bp_2d["time"])
    olr_tr  = olr_bp_2d.where(wmask, drop=True)
    u850_tr = u850_bp_2d.where(wmask, drop=True)
    u200_tr = u200_bp_2d.where(wmask, drop=True)

    # ---- 6. Drop NaN training days ----
    olr_tr, u850_tr, u200_tr, _good = _drop_nan_training_days(olr_tr, u850_tr, u200_tr)
    
    if olr_tr.sizes["time"] < 30:
        raise RuntimeError(f"Too few valid winter training days after dropping NaNs: {olr_tr.sizes['time']}")
    
    print(f"  Training days retained: {olr_tr.sizes['time']}")

    # ---- 7. Compute training means ----
    mu_olr  = olr_tr.mean("time", skipna=True)
    mu_u850 = u850_tr.mean("time", skipna=True)
    mu_u200 = u200_tr.mean("time", skipna=True)

    # ---- 8. Build training matrix (now 2D: time x lon) ----
    olr_tr_anom  = (olr_tr  - mu_olr).values.astype(np.float64)
    u850_tr_anom = (u850_tr - mu_u850).values.astype(np.float64)
    u200_tr_anom = (u200_tr - mu_u200).values.astype(np.float64)

    olr_tr_std,  s_olr  = _standardize_by_space_time_std(olr_tr_anom)
    u850_tr_std, s_u850 = _standardize_by_space_time_std(u850_tr_anom)
    u200_tr_std, s_u200 = _standardize_by_space_time_std(u200_tr_anom)

    Ttr, P = olr_tr_std.shape  # P = number of longitudes
    print(f"  Training matrix: {Ttr} days x {P} longitudes")

    X_tr = np.concatenate([olr_tr_std, u850_tr_std, u200_tr_std], axis=1)
    mu_X = np.mean(X_tr, axis=0, keepdims=True)
    X_tr = X_tr - mu_X

    if not np.isfinite(X_tr).all():
        bad = np.where(~np.isfinite(X_tr))
        raise RuntimeError(f"Non-finite values in X_tr. Example: {bad[0][0]},{bad[1][0]}")

    # ---- 9. SVD / EOF ----
    EOFs, PCs_tr, _svals = _svd_eof(X_tr, nmode=2)
    pc1_scale = np.std(PCs_tr[:, 0]) or 1.0
    pc2_scale = np.std(PCs_tr[:, 1]) or 1.0
    print(f"  EOF computed. PC1 scale: {pc1_scale:.3f}, PC2 scale: {pc2_scale:.3f}")

    # ---- 10. Project full period ----
    olr_full_anom  = ((olr_bp_2d  - mu_olr ) / s_olr ).values.astype(np.float64)
    u850_full_anom = ((u850_bp_2d - mu_u850) / s_u850).values.astype(np.float64)
    u200_full_anom = ((u200_bp_2d - mu_u200) / s_u200).values.astype(np.float64)

    Tfull = olr_full_anom.shape[0]

    good_full = (
        np.isfinite(olr_full_anom).all(axis=1)
        & np.isfinite(u850_full_anom).all(axis=1)
        & np.isfinite(u200_full_anom).all(axis=1)
    )
    good_idx = np.nonzero(good_full)[0]
    
    if good_idx.size < 10:
        raise RuntimeError(f"Too few valid full days: {good_idx.size}")

    print(f"  Full period: {Tfull} days, valid: {good_idx.size}, dropped: {Tfull - good_idx.size}")

    pc1 = np.full(Tfull, np.nan, dtype=np.float32)
    pc2 = np.full(Tfull, np.nan, dtype=np.float32)

    X_full = np.concatenate([
        olr_full_anom[good_idx], 
        u850_full_anom[good_idx], 
        u200_full_anom[good_idx]
    ], axis=1)
    X_full = X_full - mu_X

    PCs_full = X_full @ EOFs
    pc1[good_idx] = (PCs_full[:, 0] / pc1_scale).astype(np.float32)
    pc2[good_idx] = (PCs_full[:, 1] / pc2_scale).astype(np.float32)

    amp = np.sqrt(pc1**2 + pc2**2).astype(np.float32)
    phase = _phase8(pc1, pc2).astype(np.int16)
    phase[~np.isfinite(pc1)] = 0

    # ---- 11. OLR Reconstruction using regression (paper-style) ----
    print("[OLR Reconstruction] Using regression (Raw Anomaly)...")
    time_index = pd.to_datetime(olr_bp_2d["time"].values)
    winter_np = np.isin(time_index.month, list(WINTER_MONTHS))

    pc1_winter = pc1[winter_np]
    pc2_winter = pc2[winter_np]
    olr_target_winter = olr_raw_anom.values[winter_np, :]

    beta1 = np.full(P, np.nan, dtype=np.float64)
    beta2 = np.full(P, np.nan, dtype=np.float64)

    for j in range(P):
        valid = np.isfinite(pc1_winter) & np.isfinite(pc2_winter) & np.isfinite(olr_target_winter[:, j])
        if valid.sum() < 30:
            continue
        X_reg = np.column_stack([pc1_winter[valid], pc2_winter[valid]])
        y_reg = olr_target_winter[valid, j]
        try:
            b, _, _, _ = np.linalg.lstsq(X_reg, y_reg, rcond=None)
            beta1[j], beta2[j] = b
        except:
            pass

    recon_flat = pc1[:, None] * beta1[None, :] + pc2[:, None] * beta2[None, :]
    
    valid_beta = np.isfinite(beta1) & np.isfinite(beta2)
    print(f"  Regression: {valid_beta.sum()}/{P} grid points")
    print(f"  Reconstructed OLR range: [{np.nanmin(recon_flat):.2f}, {np.nanmax(recon_flat):.2f}] W/m²")

    recon_da = xr.DataArray(
        recon_flat.astype(np.float32),
        coords=olr_bp_2d.coords,
        dims=("time", "lon"),
        name="olr_recon"
    )

    # ---- 12. Tracking & Event Identification ----
    center_track_da = _track_center_with_candidates(recon_da, TRACK_LON_MIN, TRACK_LON_MAX, MAX_WEST_JUMP_DEG)
    center_track_np = center_track_da.values.astype(float)
    olr_center = _sample_olr_at_track_center(recon_da, center_track_np)

    west_lon, east_lon, contour_lon = _contour_track_from_threshold(recon_da, OLR_MIN_THRESH)

    contour_active = np.isfinite(contour_lon)
    contour_lon_w = contour_lon.copy()
    contour_lon_w[~winter_np] = np.nan
    contour_active_w = contour_active.copy()
    contour_active_w[~winter_np] = False
    amp_np = amp.copy()
    amp_np[~winter_np] = np.nan

    print(f"Winter days: {winter_np.sum()}, active contour days: {contour_active_w.sum()}")

    df_events, df_failed = _build_events_by_contour_track(
        time_index, contour_lon_w, contour_active_w, amp_np, winter_np
    )

    # ---- 13. Save Outputs ----
    out_nc = OUT_DIR / f"mjo_mvEOF_step3_{START_DATE[:4]}-{END_DATE[:4]}.nc"
    out_csv = OUT_DIR / f"mjo_events_step3_{START_DATE[:4]}-{END_DATE[:4]}.csv"
    out_csv_failed = OUT_DIR / f"mjo_failed_events_step3_{START_DATE[:4]}-{END_DATE[:4]}.csv"

    ds_out = xr.Dataset({
        "pc1": xr.DataArray(pc1, coords={"time": olr_bp_2d["time"]}, dims=("time",)),
        "pc2": xr.DataArray(pc2, coords={"time": olr_bp_2d["time"]}, dims=("time",)),
        "amp": xr.DataArray(amp, coords={"time": olr_bp_2d["time"]}, dims=("time",)),
        "phase": xr.DataArray(phase, coords={"time": olr_bp_2d["time"]}, dims=("time",)),
        "olr_recon": recon_da,
        "center_lon_track": center_track_da.astype(np.float32),
        "olr_center_track": xr.DataArray(olr_center.astype(np.float32), coords={"time": olr_bp_2d["time"]}, dims=("time",)),
        "olr_thr_centroid_lon": xr.DataArray(contour_lon.astype(np.float32), coords={"time": olr_bp_2d["time"]}, dims=("time",)),
    }, attrs={
        "step": "MV-EOF with latitudinal mean (WH04-style) + OLR regression reconstruction",
        "lat_band": f"{LAT_BAND[0]} to {LAT_BAND[1]}",
        "edge_trim_days": str(EDGE_TRIM),
        "olr_min_thresh": str(OLR_MIN_THRESH),
    })

    ds_out.to_netcdf(out_nc, engine="netcdf4")
    
    if df_events.empty:
        df_events = pd.DataFrame(columns=["event_id", "start_date", "end_date", "duration_days",
                                          "lon_start", "lon_end", "east_displacement_deg", "active_days", "amp_mean"])
    if df_failed.empty:
        df_failed = pd.DataFrame(columns=["failed_event_id", "start_date", "end_date", "duration_days",
                                          "lon_start", "lon_end", "east_displacement_deg", "active_days", "amp_mean", "failure_reason"])

    df_events.to_csv(out_csv, index=False, encoding="utf-8-sig")
    df_failed.to_csv(out_csv_failed, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {out_nc}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_csv_failed}")
    print(f"\nEvents found: {len(df_events)}")
    print(f"Failed events: {len(df_failed)}")
    if len(df_events) > 0:
        print(df_events.head(10).to_string(index=False))


if __name__ == "__main__":
    main()