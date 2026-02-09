# -*- coding: utf-8 -*-
"""
tilt_2d_relative_coord.py: 完全相对坐标系的 2D Tilt 图

横轴：经度方向阈值 (0%→100%→0%) = 西边界→核心→东边界
纵轴：纬度方向阈值 (0%→100%→0%) = 南边界→核心→北边界

两个轴都基于 omega 最小值的百分比阈值，形成完全相对的坐标系统。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.interpolate import interp1d

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
W_NORM_3D_NC = Path(r"E:\Datas\Derived\era5_mjo_recon_w_norm_3d_1979-2022.nc")
STEP3_NC = Path(r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc")
EVENTS_CSV = Path(r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv")
ONI_FILE = Path(r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt")
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\mjo_3d_structure")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
THRESHOLDS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
LOW_LAYER = (1000.0, 600.0)
UP_LAYER = (400.0, 200.0)
AMP_THRESHOLD = 0.5
LON_HALF_WIDTH = 60
ONI_THRESHOLD = 0.5
P_VALUE_THRESHOLD = 0.10
LAT_INTERP_FACTOR = 4  # 插值倍数：13 → 49 个点

ENSO_ORDER = ["El Nino", "La Nina", "Neutral"]
ENSO_COLORS = {"El Nino": "#E74C3C", "La Nina": "#3498DB", "Neutral": "#95A5A6"}


def load_oni():
    """Load ONI index data."""
    oni = pd.read_csv(ONI_FILE, sep=r'\s+', header=0, engine='python')
    month_map = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
                'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}
    records = []
    for _, row in oni.iterrows():
        seas = row['SEAS']
        year = int(row['YR'])
        anom = row['ANOM']
        if seas in month_map:
            records.append({'year': year, 'month': month_map[seas], 'oni': anom})
    oni_df = pd.DataFrame(records)
    oni_df['date'] = pd.to_datetime(oni_df[['year', 'month']].assign(day=1))
    return oni_df.set_index('date')['oni']


def classify_enso(date, oni_series):
    """Classify ENSO phase for a given date."""
    target = pd.Timestamp(year=date.year, month=date.month, day=1)
    if target in oni_series.index:
        oni_val = oni_series.loc[target]
    else:
        idx = oni_series.index.get_indexer([target], method='nearest')[0]
        if idx >= 0 and idx < len(oni_series):
            oni_val = oni_series.iloc[idx]
        else:
            return None
    if oni_val >= ONI_THRESHOLD:
        return 'El Nino'
    elif oni_val <= -ONI_THRESHOLD:
        return 'La Nina'
    return 'Neutral'

def extrapolate_boundary(coord, w, threshold_val, pivot_idx, direction):
    """
    线性外推边界位置。
    
    当搜索到数据边界仍未找到满足阈值的点时，使用边界附近的斜率进行线性外推。
    
    Args:
        coord: 坐标数组（经度或纬度）
        w: omega 剖面
        threshold_val: 阈值（已经乘过 wmin）
        pivot_idx: 核心位置索引
        direction: 'forward' (索引增大) 或 'backward' (索引减小)
    
    Returns:
        外推的边界坐标
    """
    n = len(w)
    
    if direction == 'backward':
        # 向索引减小方向搜索，使用 coord[0] 和 coord[1] 附近进行外推
        if n < 3:
            return float(coord[0])
        
        # 使用边界附近 2 个点计算斜率
        edge_idx = 0
        w0, w1 = float(w[edge_idx]), float(w[edge_idx + 1])
        c0, c1 = float(coord[edge_idx]), float(coord[edge_idx + 1])
        
        if not (np.isfinite(w0) and np.isfinite(w1)):
            return float(coord[0])
        
        # 斜率 dw/dc
        dc = c1 - c0
        dw = w1 - w0
        
        if abs(dw) < 1e-10:
            return float(coord[0])  # 斜率太小，无法外推
        
        # 线性外推: c_boundary = c0 + (threshold_val - w0) / (dw/dc)
        c_boundary = c0 + (threshold_val - w0) * dc / dw
        
        # 限制外推范围（最多外推 30°）
        max_extrap = 30.0
        if abs(c_boundary - c0) > max_extrap:
            c_boundary = c0 - np.sign(dc) * max_extrap
        
        return float(c_boundary)
    
    else:  # forward
        # 向索引增大方向搜索，使用 coord[-1] 和 coord[-2] 附近进行外推
        if n < 3:
            return float(coord[-1])
        
        edge_idx = n - 1
        w0, w1 = float(w[edge_idx]), float(w[edge_idx - 1])
        c0, c1 = float(coord[edge_idx]), float(coord[edge_idx - 1])
        
        if not (np.isfinite(w0) and np.isfinite(w1)):
            return float(coord[-1])
        
        dc = c0 - c1  # c0 是边界点
        dw = w0 - w1
        
        if abs(dw) < 1e-10:
            return float(coord[-1])
        
        c_boundary = c0 + (threshold_val - w0) * dc / dw
        
        max_extrap = 30.0
        if abs(c_boundary - c0) > max_extrap:
            c_boundary = c0 + np.sign(dc) * max_extrap
        
        return float(c_boundary)


def load_mjo_data():
    """Load MJO center track and event data."""
    print("Loading MJO tracking data...")
    ds = xr.open_dataset(STEP3_NC)
    center_lon = ds['center_lon_track'].values
    mjo_amp = ds['amp'].values
    time_mjo = pd.to_datetime(ds.time.values)
    ds.close()
    
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])
    print(f"  Loaded {len(events)} MJO events.")
    return center_lon, mjo_amp, time_mjo, events


def find_boundary_1d(coord, w, threshold_frac, direction='forward'):
    """
    一维边界查找：沿某方向找到 omega 达到阈值的位置
    
    Args:
        coord: 坐标数组 (经度或纬度)
        w: omega 剖面 (与 coord 等长)
        threshold_frac: 阈值百分比 (0-1)
        direction: 'forward' (从 pivot 向正方向) 或 'backward' (从 pivot 向负方向)
    
    Returns:
        boundary: 边界坐标值
    """
    m = np.isfinite(w) & np.isfinite(coord)
    if m.sum() < 5:
        return np.nan
    
    cc = coord[m].astype(float)
    ww = w[m].astype(float)
    
    # Find pivot (minimum omega)
    pivot_idx = int(np.nanargmin(ww))
    wmin = float(ww[pivot_idx])
    
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return np.nan
    
    thr = threshold_frac * wmin
    
    if direction == 'backward':
        # Search from pivot towards negative direction
        for i in range(pivot_idx, -1, -1):
            if ww[i] >= thr:
                return float(cc[min(i + 1, pivot_idx)])
        return float(cc[0])
    else:
        # Search from pivot towards positive direction
        for i in range(pivot_idx, len(ww)):
            if ww[i] >= thr:
                return float(cc[max(i - 1, pivot_idx)])
        return float(cc[-1])


def compute_tilt_at_threshold_2d(w_low_2d, w_up_2d, lons, lats, lon_thr, lat_thr, 
                                  pivot_lon_idx, pivot_lat_idx):
    """
    在给定的经度阈值和纬度阈值处计算 tilt
    
    Args:
        w_low_2d, w_up_2d: (lat, lon) 的 omega 场
        lons, lats: 经纬度坐标
        lon_thr: 经度方向阈值 (0-1)
        lat_thr: 纬度方向阈值 (0-1)
        pivot_lon_idx, pivot_lat_idx: omega 最小值的位置索引
    
    Returns:
        tilt: 在该点的 tilt 值
    """
    # 在 pivot 纬度处，沿经度方向找边界
    w_low_lon_profile = w_low_2d[pivot_lat_idx, :]
    w_up_lon_profile = w_up_2d[pivot_lat_idx, :]
    
    wmin_low = float(w_low_lon_profile[pivot_lon_idx])
    wmin_up = float(w_up_lon_profile[pivot_lon_idx])
    
    if not (np.isfinite(wmin_low) and wmin_low < 0 and 
            np.isfinite(wmin_up) and wmin_up < 0):
        return np.nan
    
    # 经度方向的阈值转换为边界位置
    thr_low_lon = lon_thr * wmin_low
    thr_up_lon = lon_thr * wmin_up
    
    # 找西边界 (backward)
    low_west = np.nan
    for i in range(pivot_lon_idx, -1, -1):
        if w_low_lon_profile[i] >= thr_low_lon:
            low_west = float(lons[min(i + 1, pivot_lon_idx)])
            break
    if np.isnan(low_west):
        low_west = float(lons[0])
    
    up_west = np.nan
    for i in range(pivot_lon_idx, -1, -1):
        if w_up_lon_profile[i] >= thr_up_lon:
            up_west = float(lons[min(i + 1, pivot_lon_idx)])
            break
    if np.isnan(up_west):
        up_west = float(lons[0])
    
    # Tilt = low_west - up_west
    if np.isfinite(low_west) and np.isfinite(up_west):
        return low_west - up_west
    return np.nan


def main():
    print("=" * 70)
    print("2D Relative Coordinate Tilt Map")
    print("  X: Longitude threshold (0%→100%→0%)")
    print("  Y: Latitude threshold (0%→100%→0%)")
    print("=" * 70)
    
    # Load data
    center_lon, mjo_amp, time_mjo, events = load_mjo_data()
    oni_series = load_oni()
    
    print("\nLoading 3D omega data...")
    ds = xr.open_dataset(W_NORM_3D_NC)
    data = ds['w_mjo_recon_norm_3d']
    
    if "pressure_level" in data.dims:
        data = data.rename({"pressure_level": "level"})
    
    time_data = pd.to_datetime(data.time.values)
    levels = data.level.values
    lats = data.lat.values
    lon = data.lon.values
    lon_360 = np.mod(lon, 360)
    
    low_mask = (levels >= LOW_LAYER[1]) & (levels <= LOW_LAYER[0])
    up_mask = (levels >= UP_LAYER[1]) & (levels <= UP_LAYER[0])
    
    n_thr = len(THRESHOLDS)
    n_thr_full_lon = 2 * n_thr - 1  # West + East
    n_thr_full_lat = 2 * n_thr - 1  # South + North
    
    print(f"  Lon threshold points: {n_thr_full_lon}")
    print(f"  Lat threshold points: {n_thr_full_lat}")
    
    # Build relative longitude grid
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int(2 * LON_HALF_WIDTH / dlon) + 1
    rel_lons = np.linspace(-LON_HALF_WIDTH, LON_HALF_WIDTH, n_rel_bins)
    
    # Collect tilt by ENSO phase
    # Shape: list of (n_thr_full_lon, n_thr_full_lat) arrays
    tilt_by_enso = {phase: [] for phase in ENSO_ORDER}
    
    print("\nProcessing MJO events...")
    n_samples = 0
    enso_counts = {phase: 0 for phase in ENSO_ORDER}
    
    for ev_idx, (_, ev) in enumerate(events.iterrows()):
        start = ev['start_date']
        end = ev['end_date']
        center_date = start + (end - start) / 2
        
        enso_phase = classify_enso(center_date, oni_series)
        if enso_phase is None:
            continue
        
        mask = (time_mjo >= start) & (time_mjo <= end)
        day_indices = np.where(mask)[0]
        
        for idx in day_indices:
            if time_mjo[idx].month not in WINTER_MONTHS:
                continue
            
            clon = center_lon[idx]
            amp_val = mjo_amp[idx]
            
            if not np.isfinite(clon) or not np.isfinite(amp_val) or amp_val < AMP_THRESHOLD:
                continue
            
            t = time_mjo[idx]
            try:
                data_idx = np.where(time_data == t)[0]
                if len(data_idx) == 0:
                    continue
                data_idx = data_idx[0]
            except:
                continue
            
            daily_data = data.isel(time=data_idx).values
            if np.all(np.isnan(daily_data)):
                continue
            
            # Layer means
            w_low_full = np.nanmean(daily_data[low_mask, :, :], axis=0)  # (lat, lon)
            w_up_full = np.nanmean(daily_data[up_mask, :, :], axis=0)
            
            # Sample to relative longitude grid (keeping original lats)
            clon_360 = np.mod(clon, 360)
            w_low_rel = np.zeros((len(lats), len(rel_lons)))
            w_up_rel = np.zeros((len(lats), len(rel_lons)))
            
            for j, rlon in enumerate(rel_lons):
                target_lon = np.mod(clon_360 + rlon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - target_lon))
                w_low_rel[:, j] = w_low_full[:, lon_idx]
                w_up_rel[:, j] = w_up_full[:, lon_idx]
            
            # === 纬度插值：将 13 个点插值到更高分辨率 ===
            # 注意：lats 是降序 (15°N → -15°S)，插值后需要保持相同顺序
            n_lats_interp = (len(lats) - 1) * LAT_INTERP_FACTOR + 1  # 13 → 49
            lats_interp = np.linspace(lats[0], lats[-1], n_lats_interp)  # 保持降序
            
            w_low_interp = np.zeros((n_lats_interp, len(rel_lons)))
            w_up_interp = np.zeros((n_lats_interp, len(rel_lons)))
            
            for j in range(len(rel_lons)):
                # 对每条经度剖面进行纬度插值
                f_low = interp1d(lats, w_low_rel[:, j], kind='cubic', 
                                 bounds_error=False, fill_value='extrapolate')
                f_up = interp1d(lats, w_up_rel[:, j], kind='cubic',
                                bounds_error=False, fill_value='extrapolate')
                w_low_interp[:, j] = f_low(lats_interp)
                w_up_interp[:, j] = f_up(lats_interp)
            
            # 使用插值后的数据
            w_low_rel = w_low_interp
            w_up_rel = w_up_interp
            lats_work = lats_interp
            
            # Find pivot (omega minimum) for low layer
            # Find 2D pivot (minimum omega location)
            flat_idx = np.nanargmin(w_low_rel)
            pivot_lat_idx, pivot_lon_idx = np.unravel_index(flat_idx, w_low_rel.shape)
            
            wmin_low = w_low_rel[pivot_lat_idx, pivot_lon_idx]
            if not (np.isfinite(wmin_low) and wmin_low < 0):
                continue
            
            # Compute tilt at each (lon_thr, lat_thr) combination
            daily_tilt = np.full((n_thr_full_lon, n_thr_full_lat), np.nan)
            
            # 纬度阈值：0%→100%→0%
            # 索引 0..10 对应南侧 (0%→100%)
            # 索引 10 是核心 (100%)
            # 索引 10..20 对应北侧 (100%→0%)
            
            for lat_thr_idx in range(n_thr_full_lat):
                # 确定纬度阈值和方向
                if lat_thr_idx <= n_thr - 1:
                    # 南侧: 索引 0→10 对应阈值 0%→100%
                    actual_lat_thr = THRESHOLDS[lat_thr_idx]
                    search_dir = 'south'
                else:
                    # 北侧: 索引 11→20 对应阈值 90%→0%
                    # 索引 11 → THRESHOLDS[9] = 90%
                    # 索引 12 → THRESHOLDS[8] = 80%
                    # ...
                    # 索引 20 → THRESHOLDS[0] = 0%
                    actual_lat_thr = THRESHOLDS[n_thr_full_lat - 1 - lat_thr_idx]
                    search_dir = 'north'
                
                # 找纬度边界位置
                w_low_lat_profile = w_low_rel[:, pivot_lon_idx]
                wmin_lat = float(w_low_lat_profile[pivot_lat_idx])
                
                if not (np.isfinite(wmin_lat) and wmin_lat < 0):
                    continue
                
                thr_val = actual_lat_thr * wmin_lat
                
                # 注意：lats 是降序 (15°N → -15°S)，索引 0 = 北，索引末位 = 南
                if search_dir == 'south':
                    # 向南搜索 = 索引增大（向 -15°S）
                    target_lat_idx = pivot_lat_idx
                    for i in range(pivot_lat_idx, len(lats_work)):
                        if w_low_lat_profile[i] >= thr_val:
                            target_lat_idx = max(i - 1, pivot_lat_idx)
                            break
                    else:
                        target_lat_idx = len(lats_work) - 1
                else:
                    # 向北搜索 = 索引减小（向 15°N）
                    target_lat_idx = pivot_lat_idx
                    for i in range(pivot_lat_idx, -1, -1):
                        if w_low_lat_profile[i] >= thr_val:
                            target_lat_idx = min(i + 1, pivot_lat_idx)
                            break
                    else:
                        target_lat_idx = 0
                
                # 在这个纬度位置，计算经度方向的 tilt
                w_low_lon = w_low_rel[target_lat_idx, :]
                w_up_lon = w_up_rel[target_lat_idx, :]
                
                wmin_low_lon = np.nanmin(w_low_lon)
                wmin_up_lon = np.nanmin(w_up_lon)
                
                if not (np.isfinite(wmin_low_lon) and wmin_low_lon < 0 and
                        np.isfinite(wmin_up_lon) and wmin_up_lon < 0):
                    continue
                
                pivot_lon_at_lat = int(np.nanargmin(w_low_lon))
                pivot_up_lon = int(np.nanargmin(w_up_lon))
                
                for lon_thr_idx in range(n_thr_full_lon):
                    # 确定经度阈值和方向
                    if lon_thr_idx <= n_thr - 1:
                        # 西侧: 索引 0→10 对应阈值 0%→100%
                        actual_lon_thr = THRESHOLDS[lon_thr_idx]
                        search_lon_dir = 'west'
                    else:
                        # 东侧: 索引 11→20 对应阈值 90%→0%
                        actual_lon_thr = THRESHOLDS[n_thr_full_lon - 1 - lon_thr_idx]
                        search_lon_dir = 'east'
                    
                    thr_low = actual_lon_thr * wmin_low_lon
                    thr_up = actual_lon_thr * wmin_up_lon
                    
                    if search_lon_dir == 'west':
                        # 向西搜索低层边界
                        low_boundary = None
                        for i in range(pivot_lon_at_lat, -1, -1):
                            if w_low_lon[i] >= thr_low:
                                low_boundary = float(rel_lons[min(i + 1, pivot_lon_at_lat)])
                                break
                        else:
                            # 外推
                            low_boundary = extrapolate_boundary(
                                rel_lons, w_low_lon, thr_low, pivot_lon_at_lat, 'backward')
                        
                        # 向西搜索高层边界
                        up_boundary = None
                        for i in range(pivot_up_lon, -1, -1):
                            if w_up_lon[i] >= thr_up:
                                up_boundary = float(rel_lons[min(i + 1, pivot_up_lon)])
                                break
                        else:
                            up_boundary = extrapolate_boundary(
                                rel_lons, w_up_lon, thr_up, pivot_up_lon, 'backward')
                    else:
                        # 向东搜索低层边界
                        low_boundary = None
                        for i in range(pivot_lon_at_lat, len(rel_lons)):
                            if w_low_lon[i] >= thr_low:
                                low_boundary = float(rel_lons[max(i - 1, pivot_lon_at_lat)])
                                break
                        else:
                            low_boundary = extrapolate_boundary(
                                rel_lons, w_low_lon, thr_low, pivot_lon_at_lat, 'forward')
                        
                        # 向东搜索高层边界
                        up_boundary = None
                        for i in range(pivot_up_lon, len(rel_lons)):
                            if w_up_lon[i] >= thr_up:
                                up_boundary = float(rel_lons[max(i - 1, pivot_up_lon)])
                                break
                        else:
                            up_boundary = extrapolate_boundary(
                                rel_lons, w_up_lon, thr_up, pivot_up_lon, 'forward')
                    
                    # Tilt
                    if np.isfinite(low_boundary) and np.isfinite(up_boundary):
                        daily_tilt[lon_thr_idx, lat_thr_idx] = low_boundary - up_boundary
            
            tilt_by_enso[enso_phase].append(daily_tilt)
            n_samples += 1
            enso_counts[enso_phase] += 1
        
        if (ev_idx + 1) % 20 == 0:
            print(f"  Processed {ev_idx + 1}/{len(events)} events, {n_samples} samples")
    
    print(f"\n  Total samples: {n_samples}")
    for phase in ENSO_ORDER:
        print(f"    {phase}: {enso_counts[phase]} samples")
    
    # Compute mean tilt for each ENSO phase
    mean_tilt_by_enso = {}
    all_tilt_by_enso = {}
    
    for phase in ENSO_ORDER:
        arr = np.array(tilt_by_enso[phase])  # (n_samples, n_thr_full_lon, n_thr_full_lat)
        mean_tilt_by_enso[phase] = np.nanmean(arr, axis=0)
        all_tilt_by_enso[phase] = arr
    
    # Determine max phase at each point
    print("\nDetermining dominant ENSO phase...")
    max_phase_map = np.empty((n_thr_full_lon, n_thr_full_lat), dtype=object)
    
    for i in range(n_thr_full_lon):
        for j in range(n_thr_full_lat):
            vals = {phase: mean_tilt_by_enso[phase][i, j] for phase in ENSO_ORDER}
            max_phase_map[i, j] = max(vals, key=lambda k: vals[k] if np.isfinite(vals[k]) else -np.inf)
    
    # Significance test (El Nino vs La Nina)
    print("\nComputing significance (El Nino vs La Nina)...")
    p_values = np.full((n_thr_full_lon, n_thr_full_lat), np.nan)
    
    en_arr = all_tilt_by_enso['El Nino']
    ln_arr = all_tilt_by_enso['La Nina']
    
    for i in range(n_thr_full_lon):
        for j in range(n_thr_full_lat):
            en_vals = en_arr[:, i, j]
            ln_vals = ln_arr[:, i, j]
            
            en_valid = en_vals[np.isfinite(en_vals)]
            ln_valid = ln_vals[np.isfinite(ln_vals)]
            
            if len(en_valid) > 5 and len(ln_valid) > 5:
                _, p = stats.ttest_ind(en_valid, ln_valid, equal_var=False)
                p_values[i, j] = p
    
    significant = p_values < P_VALUE_THRESHOLD
    print(f"  Significant points (p < {P_VALUE_THRESHOLD}): {np.sum(significant)}")
    
    # === Plot ===
    print("\nGenerating 2D ENSO Dominance Map (Relative Coordinates)...")
    
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    
    # Axis labels
    x_labels = [f"{int(t*100)}" for t in THRESHOLDS] + [f"{int(t*100)}" for t in THRESHOLDS[-2::-1]]
    y_labels = [f"{int(t*100)}" for t in THRESHOLDS] + [f"{int(t*100)}" for t in THRESHOLDS[-2::-1]]
    
    x = np.arange(n_thr_full_lon)
    y = np.arange(n_thr_full_lat)
    
    # Color map
    from matplotlib.colors import ListedColormap
    phase_to_num = {'El Nino': 0, 'La Nina': 1, 'Neutral': 2}
    num_to_color = [ENSO_COLORS['El Nino'], ENSO_COLORS['La Nina'], ENSO_COLORS['Neutral']]
    cmap = ListedColormap(num_to_color)
    
    color_map = np.zeros((n_thr_full_lon, n_thr_full_lat))
    for i in range(n_thr_full_lon):
        for j in range(n_thr_full_lat):
            color_map[i, j] = phase_to_num[max_phase_map[i, j]]
    
    X, Y = np.meshgrid(x, y)
    pcm = ax.pcolormesh(X, Y, color_map.T, cmap=cmap, vmin=-0.5, vmax=2.5, shading='auto')
    
    # Significance stippling
    sig_x, sig_y = [], []
    for i in range(n_thr_full_lon):
        for j in range(n_thr_full_lat):
            if significant[i, j]:
                sig_x.append(x[i])
                sig_y.append(y[j])
    
    ax.scatter(sig_x, sig_y, c='black', s=30, marker='o', alpha=0.9, linewidths=0.5, edgecolors='white')
    
    # Mark cores
    core_lon_idx = n_thr - 1
    core_lat_idx = n_thr - 1
    ax.axvline(core_lon_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axhline(core_lat_idx, color='black', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Labels
    ax.set_xticks(x[::2])
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=9)
    ax.set_yticks(y[::2])
    ax.set_yticklabels([y_labels[i] for i in range(0, len(y_labels), 2)], fontsize=9)
    
    ax.set_xlabel('经度阈值 (%) ← 西侧 | 东侧 →', fontsize=12)
    ax.set_ylabel('纬度阈值 (%) ← 南侧 | 北侧 →', fontsize=12)
    ax.set_title(f'2D ENSO Tilt 主导区域图（相对坐标系）(N={n_samples})\n'
                 f'颜色=Tilt最大的ENSO相 | 黑点=El Nino vs La Nina显著差异',
                 fontsize=13, fontweight='bold')
    
    # Legend
    legend_elements = [Patch(facecolor=ENSO_COLORS[p], label=f'{p} ({enso_counts[p]})') 
                       for p in ENSO_ORDER]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_aspect('equal')
    
    plt.tight_layout()
    out_path = OUT_DIR / "mjo_tilt_2d_relative_coord_enso.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()
    
    # === Plot El Nino - La Nina difference ===
    print("\nGenerating El Nino - La Nina difference map...")
    
    fig2, ax2 = plt.subplots(figsize=(12, 10), dpi=150)
    
    diff = mean_tilt_by_enso['El Nino'] - mean_tilt_by_enso['La Nina']
    vmax = np.nanpercentile(np.abs(diff), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    cf = ax2.contourf(X, Y, diff.T, levels=21, cmap='RdBu_r', norm=norm)
    ax2.contour(X, Y, diff.T, levels=[0], colors='k', linewidths=1.5)
    ax2.scatter(sig_x, sig_y, c='black', s=30, marker='o', alpha=0.9, linewidths=0.5, edgecolors='white')
    
    ax2.axvline(core_lon_idx, color='purple', linestyle='--', linewidth=1.5, alpha=0.6)
    ax2.axhline(core_lat_idx, color='purple', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=9)
    ax2.set_yticks(y[::2])
    ax2.set_yticklabels([y_labels[i] for i in range(0, len(y_labels), 2)], fontsize=9)
    
    ax2.set_xlabel('经度阈值 (%) ← 西侧 | 东侧 →', fontsize=12)
    ax2.set_ylabel('纬度阈值 (%) ← 南侧 | 北侧 →', fontsize=12)
    ax2.set_title(f'El Nino - La Nina Tilt 差异（相对坐标系）\n'
                  f'红色=El Nino更大 | 蓝色=La Nina更大 | 黑点=显著差异',
                  fontsize=13, fontweight='bold')
    
    cbar = fig2.colorbar(cf, ax=ax2, shrink=0.8, pad=0.02)
    cbar.set_label('Tilt 差异 (°)', fontsize=11)
    
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    out_path2 = OUT_DIR / "mjo_tilt_2d_relative_coord_diff.png"
    plt.savefig(out_path2, bbox_inches='tight')
    print(f"  Saved: {out_path2}")
    plt.close()
    
    # === Plot El Nino - Neutral difference ===
    print("\nGenerating El Nino - Neutral difference map...")
    
    # Significance test (El Nino vs Neutral)
    p_values_en_neu = np.full((n_thr_full_lon, n_thr_full_lat), np.nan)
    for i in range(n_thr_full_lon):
        for j in range(n_thr_full_lat):
            en_vals = en_arr[:, i, j]
            neu_vals = all_tilt_by_enso['Neutral'][:, i, j]
            en_valid = en_vals[np.isfinite(en_vals)]
            neu_valid = neu_vals[np.isfinite(neu_vals)]
            if len(en_valid) > 5 and len(neu_valid) > 5:
                _, p = stats.ttest_ind(en_valid, neu_valid, equal_var=False)
                p_values_en_neu[i, j] = p
    sig_en_neu = p_values_en_neu < P_VALUE_THRESHOLD
    
    sig_x_en_neu, sig_y_en_neu = [], []
    for i in range(n_thr_full_lon):
        for j in range(n_thr_full_lat):
            if sig_en_neu[i, j]:
                sig_x_en_neu.append(x[i])
                sig_y_en_neu.append(y[j])
    
    fig3, ax3 = plt.subplots(figsize=(12, 10), dpi=150)
    diff_en_neu = mean_tilt_by_enso['El Nino'] - mean_tilt_by_enso['Neutral']
    vmax3 = np.nanpercentile(np.abs(diff_en_neu), 95)
    norm3 = TwoSlopeNorm(vmin=-vmax3, vcenter=0, vmax=vmax3)
    
    cf3 = ax3.contourf(X, Y, diff_en_neu.T, levels=21, cmap='RdBu_r', norm=norm3)
    ax3.contour(X, Y, diff_en_neu.T, levels=[0], colors='k', linewidths=1.5)
    ax3.scatter(sig_x_en_neu, sig_y_en_neu, c='black', s=30, marker='o', alpha=0.9, linewidths=0.5, edgecolors='white')
    
    ax3.axvline(core_lon_idx, color='purple', linestyle='--', linewidth=1.5, alpha=0.6)
    ax3.axhline(core_lat_idx, color='purple', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax3.set_xticks(x[::2])
    ax3.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=9)
    ax3.set_yticks(y[::2])
    ax3.set_yticklabels([y_labels[i] for i in range(0, len(y_labels), 2)], fontsize=9)
    
    ax3.set_xlabel('经度阈值 (%) ← 西侧 | 东侧 →', fontsize=12)
    ax3.set_ylabel('纬度阈值 (%) ← 南侧 | 北侧 →', fontsize=12)
    ax3.set_title(f'El Nino - Neutral Tilt 差异（相对坐标系）\n'
                  f'红色=El Nino更大 | 蓝色=Neutral更大 | 黑点=显著差异 (N_sig={np.sum(sig_en_neu)})',
                  fontsize=13, fontweight='bold')
    
    cbar3 = fig3.colorbar(cf3, ax=ax3, shrink=0.8, pad=0.02)
    cbar3.set_label('Tilt 差异 (°)', fontsize=11)
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    out_path3 = OUT_DIR / "mjo_tilt_2d_relative_coord_diff_en_neu.png"
    plt.savefig(out_path3, bbox_inches='tight')
    print(f"  Saved: {out_path3}")
    plt.close()
    
    # === Plot La Nina - Neutral difference ===
    print("\nGenerating La Nina - Neutral difference map...")
    
    # Significance test (La Nina vs Neutral)
    p_values_ln_neu = np.full((n_thr_full_lon, n_thr_full_lat), np.nan)
    for i in range(n_thr_full_lon):
        for j in range(n_thr_full_lat):
            ln_vals = ln_arr[:, i, j]
            neu_vals = all_tilt_by_enso['Neutral'][:, i, j]
            ln_valid = ln_vals[np.isfinite(ln_vals)]
            neu_valid = neu_vals[np.isfinite(neu_vals)]
            if len(ln_valid) > 5 and len(neu_valid) > 5:
                _, p = stats.ttest_ind(ln_valid, neu_valid, equal_var=False)
                p_values_ln_neu[i, j] = p
    sig_ln_neu = p_values_ln_neu < P_VALUE_THRESHOLD
    
    sig_x_ln_neu, sig_y_ln_neu = [], []
    for i in range(n_thr_full_lon):
        for j in range(n_thr_full_lat):
            if sig_ln_neu[i, j]:
                sig_x_ln_neu.append(x[i])
                sig_y_ln_neu.append(y[j])
    
    fig4, ax4 = plt.subplots(figsize=(12, 10), dpi=150)
    diff_ln_neu = mean_tilt_by_enso['La Nina'] - mean_tilt_by_enso['Neutral']
    vmax4 = np.nanpercentile(np.abs(diff_ln_neu), 95)
    norm4 = TwoSlopeNorm(vmin=-vmax4, vcenter=0, vmax=vmax4)
    
    cf4 = ax4.contourf(X, Y, diff_ln_neu.T, levels=21, cmap='RdBu_r', norm=norm4)
    ax4.contour(X, Y, diff_ln_neu.T, levels=[0], colors='k', linewidths=1.5)
    ax4.scatter(sig_x_ln_neu, sig_y_ln_neu, c='black', s=30, marker='o', alpha=0.9, linewidths=0.5, edgecolors='white')
    
    ax4.axvline(core_lon_idx, color='purple', linestyle='--', linewidth=1.5, alpha=0.6)
    ax4.axhline(core_lat_idx, color='purple', linestyle='--', linewidth=1.5, alpha=0.6)
    
    ax4.set_xticks(x[::2])
    ax4.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=9)
    ax4.set_yticks(y[::2])
    ax4.set_yticklabels([y_labels[i] for i in range(0, len(y_labels), 2)], fontsize=9)
    
    ax4.set_xlabel('经度阈值 (%) ← 西侧 | 东侧 →', fontsize=12)
    ax4.set_ylabel('纬度阈值 (%) ← 南侧 | 北侧 →', fontsize=12)
    ax4.set_title(f'La Nina - Neutral Tilt 差异（相对坐标系）\n'
                  f'红色=La Nina更大 | 蓝色=Neutral更大 | 黑点=显著差异 (N_sig={np.sum(sig_ln_neu)})',
                  fontsize=13, fontweight='bold')
    
    cbar4 = fig4.colorbar(cf4, ax=ax4, shrink=0.8, pad=0.02)
    cbar4.set_label('Tilt 差异 (°)', fontsize=11)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    out_path4 = OUT_DIR / "mjo_tilt_2d_relative_coord_diff_ln_neu.png"
    plt.savefig(out_path4, bbox_inches='tight')
    print(f"  Saved: {out_path4}")
    plt.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  El Nino dominant: {np.sum(max_phase_map == 'El Nino')}")
    print(f"  La Nina dominant: {np.sum(max_phase_map == 'La Nina')}")
    print(f"  Neutral dominant: {np.sum(max_phase_map == 'Neutral')}")
    print(f"  Significant El Nino vs La Nina (p < {P_VALUE_THRESHOLD}): {np.sum(significant)}")
    print(f"  Significant El Nino vs Neutral (p < {P_VALUE_THRESHOLD}): {np.sum(sig_en_neu)}")
    print(f"  Significant La Nina vs Neutral (p < {P_VALUE_THRESHOLD}): {np.sum(sig_ln_neu)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
