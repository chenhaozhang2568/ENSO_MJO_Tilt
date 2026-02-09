# -*- coding: utf-8 -*-
"""
tilt_2d_sensitivity_map.py: 二维 Tilt 敏感性平面图

将一维的"阈值→tilt"扩展为二维"(阈值, 纬度) → tilt"

横轴：阈值 (0%→100%→0%) = 西边界→核心→东边界
纵轴：纬度 (15°S ~ 15°N)
颜色：Tilt 值 (低层西边界 - 高层西边界)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
W_NORM_3D_NC = Path(r"E:\Datas\Derived\era5_mjo_recon_w_norm_3d_1979-2022.nc")
STEP3_NC = Path(r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc")
EVENTS_CSV = Path(r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv")
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\mjo_3d_structure")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings (match sensitivity_full_range.py)
THRESHOLDS_HALF = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]  # 简化为 11 个点
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
LOW_LAYER = (1000.0, 600.0)
UP_LAYER = (400.0, 200.0)
PIVOT_DELTA_DEG = 10.0
MIN_VALID_POINTS = 5
AMP_THRESHOLD = 0.5
LON_HALF_WIDTH = 60


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
    """计算边界位置
    
    Returns: (west_boundary, east_boundary)
    """
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return np.nan, np.nan
    
    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)
    
    # Find pivot (strongest ascent near center)
    win = (rr >= -PIVOT_DELTA_DEG) & (rr <= PIVOT_DELTA_DEG)
    if win.any():
        j0 = int(np.nanargmin(ww[win]))
        pivot_idx = np.where(win)[0][j0]
    else:
        pivot_idx = int(np.nanargmin(ww))
    
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return np.nan, np.nan
    
    # Threshold
    thr = threshold_frac * wmin
    
    # West boundary
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:
            west_idx = i + 1
            west_idx = min(west_idx, pivot_idx)
            break
    if west_idx is None:
        west_idx = 0
    
    # East boundary
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
    print("2D Tilt Sensitivity Map (Threshold x Latitude)")
    print("=" * 70)
    
    # Load MJO data
    center_lon, mjo_amp, time_mjo, events = load_mjo_data()
    
    # Load 3D omega data
    print("\nLoading 3D omega data...")
    ds = xr.open_dataset(W_NORM_3D_NC)
    data = ds['w_mjo_recon_norm_3d']  # (time, level, lat, lon)
    
    if "pressure_level" in data.dims:
        data = data.rename({"pressure_level": "level"})
    
    time_data = pd.to_datetime(data.time.values)
    levels = data.level.values
    lats = data.lat.values
    lon = data.lon.values
    lon_360 = np.mod(lon, 360)
    
    # Layer masks
    low_mask = (levels >= LOW_LAYER[1]) & (levels <= LOW_LAYER[0])
    up_mask = (levels >= UP_LAYER[1]) & (levels <= UP_LAYER[0])
    print(f"  Low layer: {levels[low_mask]} hPa")
    print(f"  Upper layer: {levels[up_mask]} hPa")
    
    # Build relative longitude grid
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int(2 * LON_HALF_WIDTH / dlon) + 1
    rel_lons = np.linspace(-LON_HALF_WIDTH, LON_HALF_WIDTH, n_rel_bins)
    
    # Full threshold axis: West (0→100%) + East (100→0%)
    n_half = len(THRESHOLDS_HALF)
    thresholds_full = THRESHOLDS_HALF + THRESHOLDS_HALF[-2::-1]  # 0→100→0
    n_thresh_full = len(thresholds_full)
    
    print(f"  Threshold points: {n_thresh_full}")
    print(f"  Latitude points: {len(lats)}")
    
    # Collect daily tilt samples for each (threshold, lat)
    # Shape: (n_samples, n_thresh_full, n_lats)
    all_west_tilt = []  # For west side
    all_east_tilt = []  # For east side
    
    print("\nProcessing MJO events...")
    n_samples = 0
    
    for ev_idx, (_, ev) in enumerate(events.iterrows()):
        start = ev['start_date']
        end = ev['end_date']
        
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
            
            daily_data = data.isel(time=data_idx).values  # (level, lat, lon)
            if np.all(np.isnan(daily_data)):
                continue
            
            # Layer means
            w_low_full = np.nanmean(daily_data[low_mask, :, :], axis=0)  # (lat, lon)
            w_up_full = np.nanmean(daily_data[up_mask, :, :], axis=0)
            
            # Sample to relative longitude grid
            clon_360 = np.mod(clon, 360)
            low_sample = np.zeros((len(lats), len(rel_lons)))
            up_sample = np.zeros((len(lats), len(rel_lons)))
            
            for j, rlon in enumerate(rel_lons):
                target_lon = np.mod(clon_360 + rlon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - target_lon))
                low_sample[:, j] = w_low_full[:, lon_idx]
                up_sample[:, j] = w_up_full[:, lon_idx]
            
            # Compute tilt at each (threshold, latitude)
            daily_west_tilt = np.full((n_half, len(lats)), np.nan)  # West side only
            daily_east_tilt = np.full((n_half, len(lats)), np.nan)  # East side only
            
            for lat_idx in range(len(lats)):
                for thr_idx, thr in enumerate(THRESHOLDS_HALF):
                    lw, le = calc_boundary(rel_lons, low_sample[lat_idx, :], thr)
                    uw, ue = calc_boundary(rel_lons, up_sample[lat_idx, :], thr)
                    
                    # West tilt
                    if np.isfinite(lw) and np.isfinite(uw):
                        daily_west_tilt[thr_idx, lat_idx] = lw - uw
                    
                    # East tilt
                    if np.isfinite(le) and np.isfinite(ue):
                        daily_east_tilt[thr_idx, lat_idx] = le - ue
            
            all_west_tilt.append(daily_west_tilt)
            all_east_tilt.append(daily_east_tilt)
            n_samples += 1
        
        if (ev_idx + 1) % 20 == 0:
            print(f"  Processed {ev_idx + 1}/{len(events)} events, {n_samples} samples")
    
    print(f"\n  Total samples: {n_samples}")
    
    # Average
    all_west_tilt = np.array(all_west_tilt)  # (n_samples, n_half, n_lats)
    all_east_tilt = np.array(all_east_tilt)
    
    mean_west_tilt = np.nanmean(all_west_tilt, axis=0)  # (n_half, n_lats)
    mean_east_tilt = np.nanmean(all_east_tilt, axis=0)
    
    # Combine: West (0→100) + East (100→0)
    # East side is reversed, so index [n_half-2::-1]
    mean_tilt_2d = np.vstack([mean_west_tilt, mean_east_tilt[-2::-1, :]])  # (n_thresh_full, n_lats)
    
    print(f"\n  2D tilt shape: {mean_tilt_2d.shape}")
    
    # === Plot 2D Tilt Sensitivity Map ===
    print("\nGenerating 2D Tilt Sensitivity Map...")
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    
    # X axis: threshold labels
    x_labels = [f"{int(t*100)}" for t in THRESHOLDS_HALF] + [f"{int(t*100)}" for t in THRESHOLDS_HALF[-2::-1]]
    x = np.arange(n_thresh_full)
    
    # Create meshgrid
    X, Y = np.meshgrid(x, lats)
    
    # Normalize colormap
    vmax = np.nanpercentile(np.abs(mean_tilt_2d), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Plot
    cf = ax.contourf(X, Y, mean_tilt_2d.T, levels=21, cmap='RdBu_r', norm=norm)
    ax.contour(X, Y, mean_tilt_2d.T, levels=[0], colors='k', linewidths=1.5)
    
    # Mark the core (100%)
    core_idx = len(THRESHOLDS_HALF) - 1
    ax.axvline(core_idx, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    ax.annotate('核心\n(100%)', xy=(core_idx, lats.max()), xytext=(core_idx + 1, lats.max() - 2),
                fontsize=10, color='purple', fontweight='bold')
    
    # Labels
    ax.set_xticks(x[::2])
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=9)
    ax.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=12)
    ax.set_ylabel('纬度 (°)', fontsize=12)
    ax.set_title(f'2D MJO Tilt 敏感性平面图 (N={n_samples})\n'
                 f'Tilt = 低层西边界 - 高层西边界 | 正值=向西倾斜',
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label('Tilt (°)', fontsize=11)
    
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    out_path = OUT_DIR / "mjo_tilt_2d_sensitivity_map.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()
    
    # === Plot with shading showing significant regions ===
    # Second plot: Mean tilt by latitude (1D profile)
    fig2, ax2 = plt.subplots(figsize=(10, 5), dpi=150)
    
    # Mean across all thresholds
    mean_tilt_by_lat = np.nanmean(mean_tilt_2d, axis=0)
    
    ax2.plot(lats, mean_tilt_by_lat, 'b-o', linewidth=2, markersize=8)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.axhline(np.nanmean(mean_tilt_by_lat), color='r', linestyle='--', alpha=0.7,
                label=f'Mean = {np.nanmean(mean_tilt_by_lat):.2f}°')
    ax2.set_xlabel('Latitude (°)', fontsize=12)
    ax2.set_ylabel('Mean Tilt (°)', fontsize=12)
    ax2.set_title(f'MJO Tilt vs Latitude (Averaged across all thresholds, N={n_samples})',
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    out_path2 = OUT_DIR / "mjo_tilt_mean_by_latitude.png"
    plt.savefig(out_path2, bbox_inches='tight')
    print(f"  Saved: {out_path2}")
    plt.close()
    
    # Print stats
    print("\n" + "=" * 70)
    print("Tilt Statistics (2D Map):")
    print(f"  Overall Mean: {np.nanmean(mean_tilt_2d):.2f}°")
    print(f"  Max: {np.nanmax(mean_tilt_2d):.2f}°")
    print(f"  Min: {np.nanmin(mean_tilt_2d):.2f}°")
    print("\nBy Latitude:")
    for i, lat in enumerate(lats):
        print(f"  {lat:6.1f}°: Mean Tilt = {mean_tilt_by_lat[i]:6.2f}°")
    print("=" * 70)


if __name__ == "__main__":
    main()
