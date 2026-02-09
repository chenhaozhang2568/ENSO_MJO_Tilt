# -*- coding: utf-8 -*-
"""
tilt_2d_enso_dominance.py: 2D ENSO Tilt 主导区域图

横轴：阈值 (0%→100%→0%) = 西边界→核心→东边界
纵轴：纬度 (15°S ~ 15°N)
颜色：哪个 ENSO 相的 tilt 最大 (El Nino / La Nina / Neutral)
黑点：El Nino vs La Nina 显著性检验通过的区域
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy import stats

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
THRESHOLDS_HALF = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
LOW_LAYER = (1000.0, 600.0)
UP_LAYER = (400.0, 200.0)
PIVOT_DELTA_DEG = 10.0
MIN_VALID_POINTS = 5
AMP_THRESHOLD = 0.5
LON_HALF_WIDTH = 60
ONI_THRESHOLD = 0.5
P_VALUE_THRESHOLD = 0.10  # 显著性阈值

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


def calc_boundary(rel_lon, w, threshold_frac):
    """计算边界位置"""
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return np.nan, np.nan
    
    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)
    
    win = (rr >= -PIVOT_DELTA_DEG) & (rr <= PIVOT_DELTA_DEG)
    if win.any():
        j0 = int(np.nanargmin(ww[win]))
        pivot_idx = np.where(win)[0][j0]
    else:
        pivot_idx = int(np.nanargmin(ww))
    
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return np.nan, np.nan
    
    thr = threshold_frac * wmin
    
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:
            west_idx = i + 1
            west_idx = min(west_idx, pivot_idx)
            break
    if west_idx is None:
        west_idx = 0
    
    east_idx = None
    for i in range(pivot_idx, len(ww)):
        if ww[i] >= thr:
            east_idx = i - 1
            east_idx = max(east_idx, pivot_idx)
            break
    if east_idx is None:
        east_idx = len(ww) - 1
    
    west = float(rr[west_idx])
    east = float(rr[east_idx])
    
    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return np.nan, np.nan
    
    return west, east


def main():
    print("=" * 70)
    print("2D ENSO Tilt Dominance Map (Threshold x Latitude)")
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
    
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int(2 * LON_HALF_WIDTH / dlon) + 1
    rel_lons = np.linspace(-LON_HALF_WIDTH, LON_HALF_WIDTH, n_rel_bins)
    
    n_half = len(THRESHOLDS_HALF)
    n_thresh_full = 2 * n_half - 1
    n_lats = len(lats)
    
    print(f"  Threshold points: {n_thresh_full}")
    print(f"  Latitude points: {n_lats}")
    
    # Collect tilt samples by ENSO phase
    # Structure: {enso_phase: list of (n_half, n_lats) arrays for west, similarly for east}
    tilt_by_enso_west = {phase: [] for phase in ENSO_ORDER}
    tilt_by_enso_east = {phase: [] for phase in ENSO_ORDER}
    
    print("\nProcessing MJO events...")
    n_samples = 0
    enso_counts = {phase: 0 for phase in ENSO_ORDER}
    
    for ev_idx, (_, ev) in enumerate(events.iterrows()):
        start = ev['start_date']
        end = ev['end_date']
        center_date = start + (end - start) / 2
        
        # Classify event ENSO phase
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
            
            w_low_full = np.nanmean(daily_data[low_mask, :, :], axis=0)
            w_up_full = np.nanmean(daily_data[up_mask, :, :], axis=0)
            
            clon_360 = np.mod(clon, 360)
            low_sample = np.zeros((n_lats, len(rel_lons)))
            up_sample = np.zeros((n_lats, len(rel_lons)))
            
            for j, rlon in enumerate(rel_lons):
                target_lon = np.mod(clon_360 + rlon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - target_lon))
                low_sample[:, j] = w_low_full[:, lon_idx]
                up_sample[:, j] = w_up_full[:, lon_idx]
            
            daily_west_tilt = np.full((n_half, n_lats), np.nan)
            daily_east_tilt = np.full((n_half, n_lats), np.nan)
            
            for lat_idx in range(n_lats):
                for thr_idx, thr in enumerate(THRESHOLDS_HALF):
                    lw, le = calc_boundary(rel_lons, low_sample[lat_idx, :], thr)
                    uw, ue = calc_boundary(rel_lons, up_sample[lat_idx, :], thr)
                    
                    if np.isfinite(lw) and np.isfinite(uw):
                        daily_west_tilt[thr_idx, lat_idx] = lw - uw
                    
                    if np.isfinite(le) and np.isfinite(ue):
                        daily_east_tilt[thr_idx, lat_idx] = le - ue
            
            tilt_by_enso_west[enso_phase].append(daily_west_tilt)
            tilt_by_enso_east[enso_phase].append(daily_east_tilt)
            n_samples += 1
            enso_counts[enso_phase] += 1
        
        if (ev_idx + 1) % 20 == 0:
            print(f"  Processed {ev_idx + 1}/{len(events)} events, {n_samples} samples")
    
    print(f"\n  Total samples: {n_samples}")
    for phase in ENSO_ORDER:
        print(f"    {phase}: {enso_counts[phase]} samples")
    
    # Compute mean tilt for each ENSO phase
    # Shape: (n_thresh_full, n_lats)
    mean_tilt_by_enso = {}
    all_tilt_by_enso = {}  # For significance testing
    
    for phase in ENSO_ORDER:
        west_arr = np.array(tilt_by_enso_west[phase])  # (n_samples, n_half, n_lats)
        east_arr = np.array(tilt_by_enso_east[phase])
        
        mean_west = np.nanmean(west_arr, axis=0)  # (n_half, n_lats)
        mean_east = np.nanmean(east_arr, axis=0)
        
        # Combine: West (0→100) + East (100→0)
        mean_full = np.vstack([mean_west, mean_east[-2::-1, :]])
        mean_tilt_by_enso[phase] = mean_full
        
        # Keep all samples for significance testing
        west_full = np.zeros((west_arr.shape[0], n_thresh_full, n_lats))
        west_full[:, :n_half, :] = west_arr
        west_full[:, n_half:, :] = east_arr[:, -2::-1, :]
        all_tilt_by_enso[phase] = west_full
    
    # Determine max phase at each point
    print("\nDetermining dominant ENSO phase at each point...")
    max_phase_map = np.empty((n_thresh_full, n_lats), dtype=object)
    
    for i in range(n_thresh_full):
        for j in range(n_lats):
            vals = {phase: mean_tilt_by_enso[phase][i, j] for phase in ENSO_ORDER}
            max_phase_map[i, j] = max(vals, key=lambda k: vals[k] if np.isfinite(vals[k]) else -np.inf)
    
    # Compute significance (El Nino vs La Nina t-test)
    print("\nComputing significance (El Nino vs La Nina)...")
    p_values = np.full((n_thresh_full, n_lats), np.nan)
    
    en_samples = all_tilt_by_enso['El Nino']  # (n_en, n_thresh_full, n_lats)
    ln_samples = all_tilt_by_enso['La Nina']
    
    for i in range(n_thresh_full):
        for j in range(n_lats):
            en_vals = en_samples[:, i, j]
            ln_vals = ln_samples[:, i, j]
            
            en_valid = en_vals[np.isfinite(en_vals)]
            ln_valid = ln_vals[np.isfinite(ln_vals)]
            
            if len(en_valid) > 5 and len(ln_valid) > 5:
                _, p = stats.ttest_ind(en_valid, ln_valid, equal_var=False)
                p_values[i, j] = p
    
    significant = p_values < P_VALUE_THRESHOLD
    print(f"  Significant points (p < {P_VALUE_THRESHOLD}): {np.sum(significant)} / {n_thresh_full * n_lats}")
    
    # === Plot 2D ENSO Dominance Map ===
    print("\nGenerating 2D ENSO Dominance Map...")
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    
    x = np.arange(n_thresh_full)
    x_labels = [f"{int(t*100)}" for t in THRESHOLDS_HALF] + [f"{int(t*100)}" for t in THRESHOLDS_HALF[-2::-1]]
    
    # Create color map based on max phase
    phase_to_num = {'El Nino': 0, 'La Nina': 1, 'Neutral': 2}
    num_to_color = [ENSO_COLORS['El Nino'], ENSO_COLORS['La Nina'], ENSO_COLORS['Neutral']]
    
    color_map = np.zeros((n_thresh_full, n_lats))
    for i in range(n_thresh_full):
        for j in range(n_lats):
            color_map[i, j] = phase_to_num[max_phase_map[i, j]]
    
    # Plot using pcolormesh
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(num_to_color)
    
    X, Y = np.meshgrid(x, lats)
    pcm = ax.pcolormesh(X, Y, color_map.T, cmap=cmap, vmin=-0.5, vmax=2.5, shading='auto')
    
    # Add significance stippling (black dots)
    sig_x, sig_y = [], []
    for i in range(n_thresh_full):
        for j in range(n_lats):
            if significant[i, j]:
                sig_x.append(x[i])
                sig_y.append(lats[j])
    
    ax.scatter(sig_x, sig_y, c='black', s=5, marker='.', alpha=0.8, label=f'p < {P_VALUE_THRESHOLD}')
    
    # Mark the core (100%)
    core_idx = len(THRESHOLDS_HALF) - 1
    ax.axvline(core_idx, color='black', linestyle='--', linewidth=2, alpha=0.6)
    ax.text(core_idx - 0.5, lats.max() + 0.5, '核心\n(100%)', fontsize=9, ha='center', fontweight='bold')
    
    # Labels
    ax.set_xticks(x[::2])
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=9)
    ax.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=12)
    ax.set_ylabel('纬度 (°)', fontsize=12)
    ax.set_title(f'2D ENSO Tilt 主导区域图 (N={n_samples})\n'
                 f'颜色=Tilt最大的ENSO相 | 黑点=El Niño vs La Niña显著差异 (p<{P_VALUE_THRESHOLD})',
                 fontsize=13, fontweight='bold')
    
    # Legend
    legend_elements = [Patch(facecolor=ENSO_COLORS[p], label=f'{p} ({enso_counts[p]})') 
                       for p in ENSO_ORDER]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_ylim(lats.min() - 1, lats.max() + 1)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    out_path = OUT_DIR / "mjo_tilt_2d_enso_dominance.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()
    
    # === Additional plot: Tilt difference (El Nino - La Nina) with significance ===
    print("\nGenerating El Nino - La Nina difference map...")
    
    fig2, ax2 = plt.subplots(figsize=(14, 6), dpi=150)
    
    diff = mean_tilt_by_enso['El Nino'] - mean_tilt_by_enso['La Nina']
    
    from matplotlib.colors import TwoSlopeNorm
    vmax = np.nanpercentile(np.abs(diff), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    cf = ax2.contourf(X, Y, diff.T, levels=21, cmap='RdBu_r', norm=norm)
    ax2.contour(X, Y, diff.T, levels=[0], colors='k', linewidths=1.5)
    
    # Significance stippling
    ax2.scatter(sig_x, sig_y, c='black', s=5, marker='.', alpha=0.8)
    
    ax2.axvline(core_idx, color='purple', linestyle='--', linewidth=2, alpha=0.6)
    
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=9)
    ax2.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=12)
    ax2.set_ylabel('纬度 (°)', fontsize=12)
    ax2.set_title(f'El Niño - La Niña Tilt 差异 (N={n_samples})\n'
                  f'红色=El Niño更大 | 蓝色=La Niña更大 | 黑点=显著差异',
                  fontsize=13, fontweight='bold')
    
    cbar = fig2.colorbar(cf, ax=ax2, shrink=0.9, pad=0.02)
    cbar.set_label('Tilt 差异 (°)', fontsize=11)
    
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    out_path2 = OUT_DIR / "mjo_tilt_2d_enso_diff.png"
    plt.savefig(out_path2, bbox_inches='tight')
    print(f"  Saved: {out_path2}")
    plt.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  El Nino dominant points: {np.sum(max_phase_map == 'El Nino')}")
    print(f"  La Nina dominant points: {np.sum(max_phase_map == 'La Nina')}")
    print(f"  Neutral dominant points: {np.sum(max_phase_map == 'Neutral')}")
    print(f"  Significant points (p < {P_VALUE_THRESHOLD}): {np.sum(significant)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
