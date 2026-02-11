# -*- coding: utf-8 -*-
"""
02_mvEOF.py: Step3 - 多变量 EOF 分析、MJO 事件识别与 ERA5 场重建

================================================================================
功能描述：
    本脚本实现 Wheeler & Hendon (2004) 的多变量 EOF 方法，提取 MJO 主成分，
    并通过 OLR 重建追踪对流中心，识别 MJO 活跃事件。
    同时使用 PC1/PC2 对 ERA5 气压层变量进行线性回归重建。

主要步骤：
    Part A - MJO EOF 分析：
        1. 纬向平均（15°S-15°N）：OLR、U850、U200
        2. 标准化后拼接为联合矩阵
        3. SVD 提取前两个模态（PC1、PC2）
        4. OLR 回归重建：olr_recon = β₁·PC1 + β₂·PC2
        5. 追踪对流中心（OLR 最小值经度）
        6. 事件识别：活跃天数、东传距离、经过 IO/MC 门槛
    
    Part B - ERA5 场重建（原 02b 功能）：
        对 ERA5 u/v/w/q/t 场进行 PC 回归重建，提取 MJO 相关信号

输出文件：
    - mjo_mvEOF_step3_1979-2022.nc：PC1/PC2、振幅、相位、OLR 重建
    - mjo_events_step3_1979-2022.csv：识别出的 MJO 事件列表
    - era5_mjo_recon_{var}_1979-2022.nc：各变量的 MJO 重建场
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
import warnings
import os
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================
# INPUTS (from Step2)
# ======================
OLR_BP_PATH = r"E:\Datas\ClimateIndex\processed\olr_bp_1979-2022.nc"
U_BP_PATH   = r"E:\Datas\ERA5\processed\pressure_level\era5_u850_u200_bp_1979-2022.nc"
OLR_RAW_PATH = r"E:\Datas\ClimateIndex\raw\olr\olr.day.mean.nc"  # 原始OLR用于重构
ERA5_DIR = Path(r"E:\Datas\ERA5\raw\pressure_level\era5_pl_mean_quvwT")  # ERA5 uvwqt 场重建用

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
# RUN OPTIONS
# ======================
# 设置为 True 重新生成 ERA5 重构文件，False 跳过 Part B（使用已有数据）
RUN_ERA5_RECONSTRUCTION = True

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
# WH04 FIXED SCALING FACTORS (Wheeler & Hendon 2004 standard)
# Based on 1979-2001 climatology for consistent RMM definition
# ======================
WH04_STD_OLR = 15.1   # W/m²
WH04_STD_U850 = 1.81  # m/s
WH04_STD_U200 = 4.81  # m/s
USE_WH04_FIXED_STD = True  # Set to False to use dynamic std

# ======================
# OLR REGRESSION TARGET SWITCH
# ======================
# 回归目标模式：
#   "filtered"  = 使用滤波后的 OLR (olr_bp) 进行回归
#   "raw"       = 使用原始 OLR 进行回归（包含气候态）
#   "raw_anom"  = 使用原始 OLR 异常场进行回归（去除气候态，推荐）
OLR_REGRESSION_MODE = "raw"  # <-- SWITCH: "filtered", "raw", or "raw_anom"

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

def _standardize_by_space_time_std(x: np.ndarray, fixed_std: float = None) -> tuple[np.ndarray, float]:
    """Standardize by space-time std. If fixed_std is provided, use it instead."""
    if fixed_std is not None and fixed_std > 0:
        s = fixed_std
    else:
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
    
    # ============================================================
    # 诊断 1: 经度对齐验证（插值前）
    # ============================================================
    print("\n[DIAGNOSTIC 1] Longitude Alignment Check (BEFORE interp):")
    print(f"  OLR lon range: [{float(ds_olr['lon'].min()):.2f}, {float(ds_olr['lon'].max()):.2f}], n={ds_olr.sizes['lon']}")
    print(f"  ERA5 lon range: [{float(ds_u['lon'].min()):.2f}, {float(ds_u['lon'].max()):.2f}], n={ds_u.sizes['lon']}")
    print(f"  OLR lon first 5: {ds_olr['lon'].values[:5]}")
    print(f"  OLR lon last 5: {ds_olr['lon'].values[-5:]}")
    print(f"  ERA5 lon first 5: {ds_u['lon'].values[:5]}")
    print(f"  ERA5 lon last 5: {ds_u['lon'].values[-5:]}")
    
    ds_u = ds_u.interp(lat=ds_olr["lat"], lon=ds_olr["lon"], method="linear")
    print(f"  Interpolated ERA5 shape: {ds_u['u850_bp'].shape}")
    
    # 诊断 1 续: 插值后验证
    print("[DIAGNOSTIC 1] Longitude Alignment Check (AFTER interp):")
    lon_match = np.allclose(ds_olr['lon'].values, ds_u['lon'].values)
    print(f"  Longitudes exactly match: {lon_match}")
    if not lon_match:
        print(f"  WARNING: Longitude mismatch detected!")
        print(f"  Max lon diff: {np.max(np.abs(ds_olr['lon'].values - ds_u['lon'].values)):.6f}")

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
    
    # ============================================================
    # Load RAW OLR for regression reconstruction (instead of anomaly)
    # ============================================================
    print("[OLR RAW] Loading original OLR data for reconstruction...")
    ds_olr_raw = _to_lon_180(_rename_latlon_if_needed(_open_ds(OLR_RAW_PATH)))
    ds_olr_raw = ds_olr_raw.sel(time=slice(START_DATE, END_DATE))
    ds_olr_raw = _sel_lat_band(ds_olr_raw, LAT_BAND)
    
    # Align raw OLR to filtered OLR grid (in case of slight differences)
    ds_olr_raw = ds_olr_raw.interp(lon=ds_olr["lon"], method="nearest")
    ds_olr_raw, _ = xr.align(ds_olr_raw, ds_olr, join="inner")
    
    # Apply meridional mean
    olr_raw_2d = ds_olr_raw["olr"].mean("lat", skipna=True).transpose("time", "lon")
    print(f"  Raw OLR loaded: shape = {olr_raw_2d.shape}")
    
    print(f"  After lat-mean: OLR shape = {olr_bp_2d.shape}, U850 shape = {u850_bp_2d.shape}")

    # ============================================================
    # CRITICAL FIX 1: Remove filter edge times (first/last 60 days)
    # ============================================================
    print(f"[Clean] Removing filter edge times (first/last {EDGE_TRIM} days)...")
    olr_bp_2d = olr_bp_2d.isel(time=slice(EDGE_TRIM, -EDGE_TRIM))
    u850_bp_2d = u850_bp_2d.isel(time=slice(EDGE_TRIM, -EDGE_TRIM))
    u200_bp_2d = u200_bp_2d.isel(time=slice(EDGE_TRIM, -EDGE_TRIM))
    olr_raw_2d = olr_raw_2d.isel(time=slice(EDGE_TRIM, -EDGE_TRIM))
    print(f"  Time range after trimming: {olr_bp_2d.sizes['time']} days")

    # ============================================================
    # CRITICAL FIX 2: Remove bad longitudes BEFORE dropping days
    # ============================================================
    vars_to_clean = [olr_bp_2d, u850_bp_2d, u200_bp_2d, olr_raw_2d]
    cleaned_vars = _drop_bad_longitudes(vars_to_clean)
    olr_bp_2d = cleaned_vars[0]
    u850_bp_2d = cleaned_vars[1]
    u200_bp_2d = cleaned_vars[2]
    olr_raw_2d = cleaned_vars[3]
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

    # Use WH04 fixed std or dynamic std based on setting
    if USE_WH04_FIXED_STD:
        print(f"[WH04] Using fixed scaling: OLR={WH04_STD_OLR}, U850={WH04_STD_U850}, U200={WH04_STD_U200}")
        olr_tr_std,  s_olr  = _standardize_by_space_time_std(olr_tr_anom, WH04_STD_OLR)
        u850_tr_std, s_u850 = _standardize_by_space_time_std(u850_tr_anom, WH04_STD_U850)
        u200_tr_std, s_u200 = _standardize_by_space_time_std(u200_tr_anom, WH04_STD_U200)
    else:
        olr_tr_std,  s_olr  = _standardize_by_space_time_std(olr_tr_anom)
        u850_tr_std, s_u850 = _standardize_by_space_time_std(u850_tr_anom)
        u200_tr_std, s_u200 = _standardize_by_space_time_std(u200_tr_anom)
        print(f"[Dynamic] Computed scaling: OLR={s_olr:.2f}, U850={s_u850:.2f}, U200={s_u200:.2f}")

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
    print(f"  EOF computed. PC1 scale (training std): {pc1_scale:.3f}, PC2 scale: {pc2_scale:.3f}")

    # ---- Variance fractions (for NCL-style reconstruction) ----
    total_var = np.sum(X_tr ** 2)  # = sum(S_all²)
    var_frac1 = _svals[0] ** 2 / total_var
    var_frac2 = _svals[1] ** 2 / total_var
    print(f"  Variance fractions: EOF1={var_frac1:.4f} ({var_frac1*100:.2f}%), EOF2={var_frac2:.4f} ({var_frac2*100:.2f}%)")
    print(f"  Combined: {(var_frac1+var_frac2)*100:.2f}%")
    
    # ============================================================
    # 诊断 2: PC 标准化一致性检查（训练期 vs 全周期）
    # ============================================================
    print("\n[DIAGNOSTIC 2] PC Standardization Consistency Check:")
    print(f"  Training period PC1 std: {pc1_scale:.4f}")
    print(f"  Training period PC2 std: {pc2_scale:.4f}")

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

    # ---- 11. OLR Reconstruction using regression ----
    # FIX: Normalize PC to σ=1 and add intercept term (WH04 compliance)
    # SWITCH: Use different OLR targets based on OLR_REGRESSION_MODE
    if OLR_REGRESSION_MODE == "filtered":
        print("[OLR Reconstruction] Mode: FILTERED OLR (olr_bp)...")
        olr_target_2d = olr_bp_2d.values  # (time, lon) - bandpass filtered
    elif OLR_REGRESSION_MODE == "raw":
        print("[OLR Reconstruction] Mode: RAW OLR (with climatology)...")
        olr_target_2d = olr_raw_2d.values  # (time, lon) - original
    elif OLR_REGRESSION_MODE == "raw_anom":
        print("[OLR Reconstruction] Mode: RAW OLR ANOMALY (climatology removed)...")
        # 计算原始 OLR 的日气候态
        olr_raw_vals = olr_raw_2d.values  # (time, lon)
        time_index_raw = pd.to_datetime(olr_raw_2d["time"].values)
        doy = time_index_raw.dayofyear
        
        # 按 day-of-year 计算气候态
        clim = np.zeros((366, olr_raw_vals.shape[1]), dtype=np.float64)
        count = np.zeros(366, dtype=np.int32)
        for t in range(len(doy)):
            d = doy[t] - 1  # 0-indexed
            clim[d, :] += olr_raw_vals[t, :]
            count[d] += 1
        for d in range(366):
            if count[d] > 0:
                clim[d, :] /= count[d]
        
        # 计算异常场
        olr_anom = np.zeros_like(olr_raw_vals)
        for t in range(len(doy)):
            d = doy[t] - 1
            olr_anom[t, :] = olr_raw_vals[t, :] - clim[d, :]
        
        olr_target_2d = olr_anom
        print(f"  Raw OLR anomaly range: [{np.nanmin(olr_anom):.2f}, {np.nanmax(olr_anom):.2f}] W/m²")
    else:
        raise ValueError(f"Unknown OLR_REGRESSION_MODE: {OLR_REGRESSION_MODE}")
    
    time_index = pd.to_datetime(olr_bp_2d["time"].values)
    winter_np = np.isin(time_index.month, list(WINTER_MONTHS))

    pc1_winter = pc1[winter_np]
    pc2_winter = pc2[winter_np]
    olr_target_winter = olr_target_2d[winter_np, :]

    # ============================================================
    # 修复：移除二次标准化
    # PC 已在 EOF 投影时被 pc_scale 标准化（L673-674），此处不再重复标准化
    # 直接使用 PC 进行回归，回归系数 β 的物理意义更清晰：
    #   β = dOLR / dPC (单位: W/m² per unit PC)
    # ============================================================
    pc1_valid_mask = np.isfinite(pc1_winter)
    pc2_valid_mask = np.isfinite(pc2_winter)
    
    # 仅去除均值（中心化），不再除以标准差
    pc1_winter_mean = np.nanmean(pc1_winter[pc1_valid_mask])
    pc2_winter_mean = np.nanmean(pc2_winter[pc2_valid_mask])
    pc1_winter_std = np.nanstd(pc1_winter[pc1_valid_mask])  # 仅用于诊断
    pc2_winter_std = np.nanstd(pc2_winter[pc2_valid_mask])  # 仅用于诊断
    
    # 仅中心化，不缩放（PC 已经是约 σ≈1）
    pc1_centered = pc1_winter - pc1_winter_mean
    pc2_centered = pc2_winter - pc2_winter_mean
    print(f"  PC centered (no re-scaling): PC1 std={pc1_winter_std:.3f}, PC2 std={pc2_winter_std:.3f}")
    
    # ============================================================
    # 诊断: PC 标准化一致性检查
    # ============================================================
    print("\n[DIAGNOSTIC] PC Standardization Check:")
    print(f"  EOF training PC1 scale: {pc1_scale:.4f}")
    print(f"  EOF training PC2 scale: {pc2_scale:.4f}")
    print(f"  Winter PC1 std: {pc1_winter_std:.4f}, PC2 std: {pc2_winter_std:.4f}")
    print(f"  Ratio (winter_std / training_scale): PC1={pc1_winter_std/pc1_scale:.4f}, PC2={pc2_winter_std/pc2_scale:.4f}")
    print("  ✓ Using single standardization (no double-scaling)")

    beta0 = np.full(P, np.nan, dtype=np.float64)  # Intercept
    beta1 = np.full(P, np.nan, dtype=np.float64)
    beta2 = np.full(P, np.nan, dtype=np.float64)

    for j in range(P):
        valid = np.isfinite(pc1_centered) & np.isfinite(pc2_centered) & np.isfinite(olr_target_winter[:, j])
        if valid.sum() < 30:
            continue
        # 回归模型：OLR = beta0 + beta1*PC1 + beta2*PC2
        X_reg = np.column_stack([np.ones(valid.sum()), pc1_centered[valid], pc2_centered[valid]])
        y_reg = olr_target_winter[valid, j]
        try:
            b, _, _, _ = np.linalg.lstsq(X_reg, y_reg, rcond=None)
            beta0[j], beta1[j], beta2[j] = b
        except:
            pass

    # 使用中心化的全周期 PC 进行重建（仅去均值，与训练保持一致）
    pc1_full_centered = pc1 - pc1_winter_mean
    pc2_full_centered = pc2 - pc2_winter_mean
    
    # MJO 信号：NCL 方差归一化方法 (Hu & Li style)
    # olr_reg = (beta1*PC1/var_frac1 + beta2*PC2/var_frac2) / 2
    recon_anomaly = (
        pc1_full_centered[:, None] * beta1[None, :] / var_frac1
        + pc2_full_centered[:, None] * beta2[None, :] / var_frac2
    ) / 2.0
    recon_full = beta0[None, :] + recon_anomaly  # 完整重建（含气候态）
    
    valid_beta = np.isfinite(beta0) & np.isfinite(beta1) & np.isfinite(beta2)
    print(f"  Regression: {valid_beta.sum()}/{P} grid points (with intercept)")
    print(f"  Intercept (climatology) range: [{np.nanmin(beta0):.2f}, {np.nanmax(beta0):.2f}] W/m²")
    print(f"  OLR anomaly (MJO signal) range: [{np.nanmin(recon_anomaly):.2f}, {np.nanmax(recon_anomaly):.2f}] W/m²")
    print(f"  Reconstruction scaling: /var_frac1={1/var_frac1:.2f}x, /var_frac2={1/var_frac2:.2f}x, /2")
    
    # ============================================================
    # 诊断 3: PC1 与 OLR 异常场的相关系数
    # ============================================================
    print("\n[DIAGNOSTIC 3] PC1-OLR Correlation Check:")
    
    # 选取几个关键经度点计算相关系数
    diag_lons = [60.0, 90.0, 120.0, 150.0]  # 印度洋到西太平洋
    lon_values = olr_bp_2d["lon"].values
    
    for diag_lon in diag_lons:
        # 找到最近的经度索引
        lon_idx = int(np.argmin(np.abs(lon_values - diag_lon)))
        actual_lon = lon_values[lon_idx]
        
        # 获取该经度的 OLR 时间序列
        olr_at_lon = olr_bp_2d.values[:, lon_idx]  # 滤波后的 OLR 异常
        
        # 计算与 PC1 的相关系数
        valid_mask = np.isfinite(pc1) & np.isfinite(olr_at_lon)
        if valid_mask.sum() > 30:
            corr_pc1 = np.corrcoef(pc1[valid_mask], olr_at_lon[valid_mask])[0, 1]
            corr_pc2 = np.corrcoef(pc2[valid_mask], olr_at_lon[valid_mask])[0, 1]
            print(f"  @ {actual_lon:.1f}°E: PC1-OLR corr = {corr_pc1:.3f}, PC2-OLR corr = {corr_pc2:.3f}")
        else:
            print(f"  @ {actual_lon:.1f}°E: Insufficient valid data points")
    
    # 综合诊断结论
    # 选取印度洋 (90°E) 作为主要诊断点
    io_lon_idx = int(np.argmin(np.abs(lon_values - 90.0)))
    olr_at_io = olr_bp_2d.values[:, io_lon_idx]
    valid_io = np.isfinite(pc1) & np.isfinite(olr_at_io)
    corr_io = np.corrcoef(pc1[valid_io], olr_at_io[valid_io])[0, 1] if valid_io.sum() > 30 else np.nan
    
    print(f"\n[DIAGNOSTIC CONCLUSION]")
    if np.isfinite(corr_io):
        if abs(corr_io) < 0.3:
            print(f"  ❌ PC1-OLR (90°E) corr = {corr_io:.3f} << 0.5")
            print(f"     EOF可能计算错误（数据未对齐或权重极度错误）")
            print(f"     PC 没有捕捉到 OLR 的变化，重构自然很小。")
        elif abs(corr_io) < 0.5:
            print(f"  ⚠️ PC1-OLR (90°E) corr = {corr_io:.3f} < 0.5")
            print(f"     EOF 可能有问题，建议检查数据对齐和权重。")
        elif abs(corr_io) < 0.8:
            print(f"  ⚠️ PC1-OLR (90°E) corr = {corr_io:.3f} (0.5~0.8)")
            print(f"     EOF 基本合理，但回归公式可能需要检查。")
        else:
            print(f"  ✓ PC1-OLR (90°E) corr = {corr_io:.3f} > 0.8")
            print(f"     EOF 正确，如果重构仍有问题，检查回归公式（如 PC 是否正确标准化）。")
    else:
        print(f"  无法计算相关系数（数据不足）")
    
    # Use anomaly for tracking (negative = enhanced convection)
    recon_flat = recon_anomaly

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

    # ---- 12b. Post-process center_lon_track: clamp jump outliers per event ----
    # 对每个事件，检测 center_lon_track 的异常跳跃并钳位：
    #   - 开头从东侧跳到西侧 → 开头钳位到 TRACK_LON_MIN (60°)
    #   - 结尾突然跳离主序列 → 结尾钳位到 TRACK_LON_MAX (180°)
    _MAX_JUMP_DEG = 20.0
    center_track_np_fixed = center_track_np.copy()
    all_ev_for_fix = pd.concat([df_events, df_failed], ignore_index=True) if not df_failed.empty else df_events.copy()
    for _, ev_row in all_ev_for_fix.iterrows():
        t0 = pd.Timestamp(ev_row['start_date'])
        t1 = pd.Timestamp(ev_row['end_date'])
        idx_pos = np.where((time_index >= t0) & (time_index <= t1))[0]
        if len(idx_pos) < 2:
            continue
        seg = (center_track_np_fixed[idx_pos] + 360) % 360
        # 开头钳位: 从前往后找第一个西向大跳跃（从东侧跳到西侧），将跳跃前的点钳位到 TRACK_LON_MIN
        for i in range(len(seg) - 1):
            if seg[i] - seg[i + 1] > _MAX_JUMP_DEG:  # 西向跳跃（经度突然减小）
                seg[:i + 1] = TRACK_LON_MIN
                break
        # 结尾钳位: 从后往前找第一个西向大跳跃（从东侧跳回西侧），将跳跃后的点钳位到 TRACK_LON_MAX
        for i in range(len(seg) - 1, 0, -1):
            if seg[i - 1] - seg[i] > _MAX_JUMP_DEG:  # 西向跳跃（经度突然减小）
                seg[i:] = TRACK_LON_MAX
                break
        center_track_np_fixed[idx_pos] = seg

    center_track_da = xr.DataArray(
        center_track_np_fixed, coords={"time": center_track_da["time"]},
        dims=("time",), name="center_lon_track"
    )
    center_track_np = center_track_np_fixed
    olr_center = _sample_olr_at_track_center(recon_da, center_track_np)
    print(f"  center_lon_track jump-clamp applied to {len(all_ev_for_fix)} events")

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
        "step": "MV-EOF with latitudinal mean (WH04-style) + Raw OLR regression reconstruction",
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





# ======================
# PART B: ERA5 uvwqt 场重建 (原 02b_reconstruct_era5.py)
# ======================
ERA5_RECON_VARIABLES = ["u", "v", "w", "q", "t"]


def _load_era5_for_recon(var: str, time_index: pd.DatetimeIndex) -> xr.DataArray:
    """
    Load ERA5 monthly files for one variable and concatenate.
    Returns DataArray with dims (time, pressure_level, lon) after lat-averaging.
    """
    print(f"  Loading ERA5 {var}...")
    
    # Determine year-months to load
    ym_set = set()
    for t in time_index:
        ym_set.add((t.year, t.month))
    ym_list = sorted(ym_set)
    
    arrays = []
    for year, month in ym_list:
        fpath = ERA5_DIR / f"era5_pl_dailymean_quvwT_{year}{month:02d}.nc"
        if not fpath.exists():
            print(f"    WARNING: {fpath.name} not found, skipping")
            continue
        ds = xr.open_dataset(fpath)
        ds = _rename_latlon_if_needed(ds)
        ds = _to_lon_180(ds)
        
        # Select latitude band and average
        lat_band = LAT_BAND
        da = ds[var].sel(lat=slice(lat_band[1], lat_band[0]))  # ERA5 has decreasing lat
        da = da.mean(dim="lat", keepdims=False)  # (time, level, lon)
        arrays.append(da)
    
    # Concatenate along time
    combined = xr.concat(arrays, dim="time")
    
    # Align to target time index
    combined = combined.sel(time=time_index, method="nearest")
    combined = combined.assign_coords(time=time_index)
    
    print(f"    Shape: {combined.shape}")
    return combined


def _reconstruct_era5_field(
    field: xr.DataArray,
    pc1: np.ndarray,
    pc2: np.ndarray,
    winter_mask: np.ndarray
) -> xr.DataArray:
    """
    Reconstruct field using PC1/PC2 regression.
    
    Input field dims: (time, level, lon)
    Output: same dims, MJO-reconstructed
    """
    # Get dimensions
    time_coord = field["time"]
    level_coord = field["pressure_level"]
    lon_coord = field["lon"]
    
    T, L, X = field.shape  # time, level, lon
    out = np.full((T, L, X), np.nan, dtype=np.float32)
    
    # Precompute valid PC mask
    pc_valid = np.isfinite(pc1) & np.isfinite(pc2)
    train_mask = winter_mask & pc_valid
    
    # Loop over levels and longitudes
    for lev_idx in range(L):
        for lon_idx in range(X):
            y = field.values[:, lev_idx, lon_idx].astype(float)
            m = train_mask & np.isfinite(y)
            
            if m.sum() < 10:
                continue
            
            # Build design matrix [pc1, pc2] (no intercept for anomaly-like recon)
            X_train = np.column_stack([pc1[m], pc2[m]])
            y_train = y[m]
            
            # Least squares
            beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
            
            # Reconstruct for all valid PC times
            out[pc_valid, lev_idx, lon_idx] = (
                beta[0] * pc1[pc_valid] + beta[1] * pc2[pc_valid]
            )
    
    return xr.DataArray(
        out,
        coords={"time": time_coord, "pressure_level": level_coord, "lon": lon_coord},
        dims=("time", "pressure_level", "lon"),
        name=f"{field.name}_mjo_recon"
    )


def reconstruct_era5_fields():
    """
    Part B: Reconstruct ERA5 uvwqt fields using MJO PC1/PC2.
    This function is merged from 02b_reconstruct_era5.py.
    
    Outputs two versions for each variable:
    1. era5_mjo_recon_{var}_*.nc - Raw MJO reconstruction
    2. era5_mjo_recon_{var}_norm_*.nc - Normalized by MJO amplitude (Hu & Li 2021)
    """
    print("\n" + "="*70)
    print("Part B: ERA5 uvwqT Reconstruction using MJO PC1/PC2")
    print("="*70)
    start_time = datetime.now()
    
    # Load PC data from just-created Step3 output
    step3_nc = OUT_DIR / f"mjo_mvEOF_step3_{START_DATE[:4]}-{END_DATE[:4]}.nc"
    if not step3_nc.exists():
        print(f"[ERROR] Step3 output not found: {step3_nc}")
        print("        Run Part A (main) first to generate PC1/PC2.")
        return
    
    print("Loading PC1/PC2 and amp from Step3...")
    ds_pc = xr.open_dataset(step3_nc)
    pc1 = ds_pc["pc1"]
    pc2 = ds_pc["pc2"]
    amp = ds_pc["amp"]  # MJO amplitude for normalization
    time_index = pd.to_datetime(pc1["time"].values)
    print(f"  PC time range: {time_index[0].strftime('%Y-%m-%d')} to {time_index[-1].strftime('%Y-%m-%d')}")
    print(f"  Total days: {len(time_index)}")
    
    pc1_np = pc1.values.astype(float)
    pc2_np = pc2.values.astype(float)
    amp_np = amp.values.astype(float)
    winter = np.array([t.month in WINTER_MONTHS for t in time_index])
    
    # Normalization settings (consistent with src/03_compute_tilt_daily.py)
    AMP_FLOOR = 1.0  # Prevent small-amplitude blow-up
    amp_safe = np.maximum(amp_np, AMP_FLOOR)
    
    print(f"\nWinter days (training): {winter.sum()}")
    print(f"Amp stats: min={np.nanmin(amp_np):.2f}, mean={np.nanmean(amp_np):.2f}, max={np.nanmax(amp_np):.2f}")
    print(f"Using AMP_FLOOR={AMP_FLOOR} for normalization")
    
    # Process each variable
    for var in ERA5_RECON_VARIABLES:
        print(f"\n{'='*50}")
        print(f"Processing: {var}")
        print("="*50)
        
        # Load ERA5 data
        field = _load_era5_for_recon(var, time_index)
        
        # Reconstruct
        print(f"  Reconstructing...")
        recon = _reconstruct_era5_field(field, pc1_np, pc2_np, winter)
        
        # Check NaN ratio
        nan_ratio = float(np.isnan(recon.values).mean())
        print(f"  NaN ratio: {nan_ratio:.4f}")
        
        # ============================================================
        # Save 1: Raw MJO reconstruction (unchanged)
        # ============================================================
        output_path = OUT_DIR / f"era5_mjo_recon_{var}_{START_DATE[:4]}-{END_DATE[:4]}.nc"
        
        ds_out = xr.Dataset({
            f"{var}_mjo_recon": recon
        })
        ds_out.attrs["description"] = f"MJO-reconstructed {var} field via PC1/PC2 regression"
        ds_out.attrs["method"] = "Linear regression: field_hat = b1*pc1 + b2*pc2, trained on winter (Nov-Apr)"
        ds_out.attrs["source"] = "ERA5 daily mean pressure level data"
        ds_out.attrs["created"] = datetime.now().isoformat()
        
        ds_out.to_netcdf(output_path)
        print(f"  Saved (raw): {output_path}")
        
        # ============================================================
        # Save 2: Normalized by MJO amplitude (Hu & Li 2021 method)
        # ============================================================
        recon_np = recon.values.astype(float)  # (time, level, lon)
        
        # Normalize: divide by amp along time axis
        # amp_safe shape: (time,) -> expand to (time, 1, 1) for broadcasting
        recon_norm_np = recon_np / amp_safe[:, None, None]
        
        recon_norm = xr.DataArray(
            recon_norm_np.astype(np.float32),
            coords=recon.coords,
            dims=recon.dims,
            name=f"{var}_mjo_recon_norm"
        )
        
        output_path_norm = OUT_DIR / f"era5_mjo_recon_{var}_norm_{START_DATE[:4]}-{END_DATE[:4]}.nc"
        
        ds_out_norm = xr.Dataset({
            f"{var}_mjo_recon_norm": recon_norm
        })
        ds_out_norm.attrs["description"] = f"MJO-reconstructed {var} field, normalized by MJO amplitude"
        ds_out_norm.attrs["method"] = "field_norm = (b1*pc1 + b2*pc2) / max(amp, 1.0)"
        ds_out_norm.attrs["reference"] = "Hu & Li (2021) normalization method"
        ds_out_norm.attrs["amp_floor"] = str(AMP_FLOOR)
        ds_out_norm.attrs["source"] = "ERA5 daily mean pressure level data"
        ds_out_norm.attrs["created"] = datetime.now().isoformat()
        
        ds_out_norm.to_netcdf(output_path_norm)
        print(f"  Saved (norm): {output_path_norm}")
    
    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"All ERA5 reconstructions completed in {elapsed}")
    print(f"Output directory: {OUT_DIR}")
    print("="*70)


# ======================
# PART C: ERA5 3D 场重建（保留纬度维度）
# ======================
ERA5_3D_RECON_VARIABLES = ["u", "q", "w", "t"]  # 只重构 3D 可视化需要的变量


def _load_era5_for_recon_3d(var: str, time_index: pd.DatetimeIndex) -> xr.DataArray:
    """
    Load ERA5 monthly files for one variable WITHOUT latitude averaging.
    Returns DataArray with dims (time, pressure_level, lat, lon).
    """
    print(f"  Loading ERA5 {var} (keeping lat dimension)...")
    
    ym_set = set()
    for t in time_index:
        ym_set.add((t.year, t.month))
    ym_list = sorted(ym_set)
    
    arrays = []
    for year, month in ym_list:
        fpath = ERA5_DIR / f"era5_pl_dailymean_quvwT_{year}{month:02d}.nc"
        if not fpath.exists():
            print(f"    WARNING: {fpath.name} not found, skipping")
            continue
        ds = xr.open_dataset(fpath)
        ds = _rename_latlon_if_needed(ds)
        ds = _to_lon_180(ds)
        
        # Select latitude band but DO NOT average
        lat_band = LAT_BAND
        da = ds[var].sel(lat=slice(lat_band[1], lat_band[0]))  # ERA5 has decreasing lat
        # NO latitude averaging here - keep (time, level, lat, lon)
        arrays.append(da)
    
    combined = xr.concat(arrays, dim="time")
    combined = combined.sel(time=time_index, method="nearest")
    combined = combined.assign_coords(time=time_index)
    
    print(f"    Shape: {combined.shape}")
    return combined


def _reconstruct_era5_field_3d(
    field: xr.DataArray,
    pc1: np.ndarray,
    pc2: np.ndarray,
    winter_mask: np.ndarray
) -> xr.DataArray:
    """
    Reconstruct field using PC1/PC2 regression, keeping latitude dimension.
    
    Input field dims: (time, level, lat, lon)
    Output: same dims, MJO-reconstructed
    """
    time_coord = field["time"]
    level_coord = field["pressure_level"]
    lat_coord = field["lat"]
    lon_coord = field["lon"]
    
    T, L, Y, X = field.shape  # time, level, lat, lon
    out = np.full((T, L, Y, X), np.nan, dtype=np.float32)
    
    pc_valid = np.isfinite(pc1) & np.isfinite(pc2)
    train_mask = winter_mask & pc_valid
    
    # Progress tracking
    total_points = L * Y * X
    processed = 0
    
    # Loop over levels, latitudes, and longitudes
    for lev_idx in range(L):
        for lat_idx in range(Y):
            for lon_idx in range(X):
                y = field.values[:, lev_idx, lat_idx, lon_idx].astype(float)
                m = train_mask & np.isfinite(y)
                
                if m.sum() < 10:
                    continue
                
                X_train = np.column_stack([pc1[m], pc2[m]])
                y_train = y[m]
                
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
                    out[pc_valid, lev_idx, lat_idx, lon_idx] = (
                        beta[0] * pc1[pc_valid] + beta[1] * pc2[pc_valid]
                    )
                except:
                    pass
                
                processed += 1
        
        # Progress update per level
        pct = (lev_idx + 1) / L * 100
        print(f"    Level {lev_idx+1}/{L} done ({pct:.0f}%)")
    
    return xr.DataArray(
        out,
        coords={"time": time_coord, "pressure_level": level_coord, "lat": lat_coord, "lon": lon_coord},
        dims=("time", "pressure_level", "lat", "lon"),
        name=f"{field.name}_mjo_recon_3d"
    )


def reconstruct_era5_fields_3d():
    """
    Part C: Reconstruct ERA5 fields WITH latitude dimension for 3D visualization.
    
    Output: era5_mjo_recon_{var}_norm_3d_*.nc with dims (time, level, lat, lon)
    
    Skips if output file already exists.
    """
    print("\n" + "="*70)
    print("Part C: ERA5 3D Reconstruction (keeping latitude)")
    print("="*70)
    start_time = datetime.now()
    
    # Load PC data
    step3_nc = OUT_DIR / f"mjo_mvEOF_step3_{START_DATE[:4]}-{END_DATE[:4]}.nc"
    if not step3_nc.exists():
        print(f"[ERROR] Step3 output not found: {step3_nc}")
        return
    
    print("Loading PC1/PC2 and amp from Step3...")
    ds_pc = xr.open_dataset(step3_nc)
    pc1 = ds_pc["pc1"]
    pc2 = ds_pc["pc2"]
    amp = ds_pc["amp"]
    time_index = pd.to_datetime(pc1["time"].values)
    
    pc1_np = pc1.values.astype(float)
    pc2_np = pc2.values.astype(float)
    amp_np = amp.values.astype(float)
    winter = np.array([t.month in WINTER_MONTHS for t in time_index])
    
    AMP_FLOOR = 1.0
    amp_safe = np.maximum(amp_np, AMP_FLOOR)
    
    print(f"Time range: {time_index[0].strftime('%Y-%m-%d')} to {time_index[-1].strftime('%Y-%m-%d')}")
    print(f"Total days: {len(time_index)}, Winter days: {winter.sum()}")
    
    for var in ERA5_3D_RECON_VARIABLES:
        output_path_norm = OUT_DIR / f"era5_mjo_recon_{var}_norm_3d_{START_DATE[:4]}-{END_DATE[:4]}.nc"
        
        # === Check if file already exists ===
        if output_path_norm.exists():
            print(f"\n[SKIP] {var}: 3D normalized file already exists: {output_path_norm.name}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing 3D: {var}")
        print("="*50)
        
        # Load ERA5 data (without lat averaging)
        field = _load_era5_for_recon_3d(var, time_index)
        
        # Reconstruct
        print(f"  Reconstructing (this may take a while)...")
        recon = _reconstruct_era5_field_3d(field, pc1_np, pc2_np, winter)
        
        nan_ratio = float(np.isnan(recon.values).mean())
        print(f"  NaN ratio: {nan_ratio:.4f}")
        
        # Normalize by MJO amplitude
        recon_np = recon.values.astype(float)
        # Broadcast amp_safe from (time,) to (time, level, lat, lon)
        recon_norm_np = recon_np / amp_safe[:, None, None, None]
        
        recon_norm = xr.DataArray(
            recon_norm_np.astype(np.float32),
            coords=recon.coords,
            dims=recon.dims,
            name=f"{var}_mjo_recon_norm_3d"
        )
        
        ds_out_norm = xr.Dataset({
            f"{var}_mjo_recon_norm_3d": recon_norm
        })
        ds_out_norm.attrs["description"] = f"MJO-reconstructed {var} field (3D with lat), normalized by MJO amplitude"
        ds_out_norm.attrs["method"] = "field_norm = (b1*pc1 + b2*pc2) / max(amp, 1.0)"
        ds_out_norm.attrs["reference"] = "For 3D visualization (lon-lat-height)"
        ds_out_norm.attrs["source"] = "ERA5 daily mean pressure level data"
        ds_out_norm.attrs["created"] = datetime.now().isoformat()
        
        ds_out_norm.to_netcdf(output_path_norm)
        print(f"  Saved: {output_path_norm}")
    
    elapsed = datetime.now() - start_time
    print(f"\n{'='*70}")
    print(f"3D ERA5 reconstructions completed in {elapsed}")
    print("="*70)


def run_all():
    """Run both Part A and Part B (if enabled)."""
    # Part A: MJO EOF analysis and event identification
    main()
    # Part B: ERA5 field reconstruction (merged from 02b)
    if RUN_ERA5_RECONSTRUCTION:
        reconstruct_era5_fields()
    else:
        print("\n[SKIP] Part B: ERA5 reconstruction skipped (RUN_ERA5_RECONSTRUCTION=False)")
        print("       Using existing normalized files in E:\\Datas\\Derived\\")


if __name__ == "__main__":
    run_all()

