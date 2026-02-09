# -*- coding: utf-8 -*-
"""
tilt_2d_map.py: 计算各纬度的 MJO Tilt 并绘制平面图

功能：
    使用带纬度的 3D omega 重构数据，在每个纬度点计算 tilt，
    然后绘制经度-纬度平面图（颜色表示 tilt 大小）。

Tilt 定义（与 03_compute_tilt_daily.py 一致）：
    Tilt = 低层上升区西边界 - 高层上升区西边界（相对经度，单位：°）
    正值表示"低层偏东、高层偏西"的典型 MJO 后倾结构。

输出：
    - 平面 tilt 图：横轴相对经度，纵轴纬度，颜色表示 tilt 大小
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
from pathlib import Path

# ========================
# PATHS
# ========================
DERIVED_DIR = Path(r"E:\Datas\Derived")
W_NORM_3D_NC = DERIVED_DIR / "era5_mjo_recon_w_norm_3d_1979-2022.nc"
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\mjo_3d_structure")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# SETTINGS (match 03_compute_tilt_daily.py)
# ========================
LOW_LAYER = (1000.0, 600.0)   # 低层：1000..600 hPa
UP_LAYER  = (400.0, 200.0)    # 高层：400..200 hPa

HALF_MAX_FRACTION = 0.5  # 50% 半高宽
PIVOT_DELTA_DEG = 10.0   # ω_min 搜索范围
MIN_VALID_POINTS = 5
AMP_THRESHOLD = 0.5      # 最小 MJO 振幅
LON_HALF_WIDTH = 60      # 相对经度范围

SIGMA_SMOOTH = 1.0       # 平滑参数


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


def _ascent_boundary_by_half_max(
    rel_lon: np.ndarray,
    w: np.ndarray,
    half_max_fraction: float = 0.5,
    pivot_delta: float = 10.0,
):
    """
    使用半高宽法 (FWHM-like) 定义边界。
    
    Returns: (west_rel, east_rel, center_rel, wmin)
    """
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return (np.nan, np.nan, np.nan, np.nan)
    
    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)
    
    # 找 pivot：对流中心附近最强上升点
    win = (rr >= -pivot_delta) & (rr <= pivot_delta)
    if win.any():
        j0 = int(np.nanargmin(ww[win]))
        pivot_idx = np.where(win)[0][j0]
    else:
        pivot_idx = int(np.nanargmin(ww))
    
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return (np.nan, np.nan, np.nan, wmin)
    
    # 半高宽阈值
    thr = float(half_max_fraction) * wmin
    
    # West edge
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:
            west_idx = i
            break
    if west_idx is None:
        west_idx = 0
    
    # East edge
    east_idx = None
    for i in range(pivot_idx, len(ww)):
        if ww[i] >= thr:
            east_idx = i
            break
    if east_idx is None:
        east_idx = len(ww) - 1
    
    west = float(rr[west_idx])
    east = float(rr[east_idx])
    
    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return (np.nan, np.nan, np.nan, wmin)
    
    center = 0.5 * (west + east)
    return (west, east, center, wmin)


def create_tilt_composite_2d(center_lon, mjo_amp, time_mjo, events):
    """
    Create 2D tilt composite (lat, rel_lon).
    
    At each (lat, rel_lon), compute tilt using omega profiles.
    """
    print("Loading 3D omega data...")
    ds = xr.open_dataset(W_NORM_3D_NC)
    data = ds['w_mjo_recon_norm_3d']  # (time, pressure_level, lat, lon)
    
    if "pressure_level" in data.dims:
        data = data.rename({"pressure_level": "level"})
    
    time_data = pd.to_datetime(data.time.values)
    levels = data.level.values
    lats = data.lat.values
    lon = data.lon.values
    lon_360 = np.mod(lon, 360)
    
    # Layer indices
    low_mask = (levels >= LOW_LAYER[1]) & (levels <= LOW_LAYER[0])
    up_mask = (levels >= UP_LAYER[1]) & (levels <= UP_LAYER[0])
    print(f"  Low layer: {levels[low_mask]} hPa")
    print(f"  Upper layer: {levels[up_mask]} hPa")
    
    # Build relative longitude grid
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int(2 * LON_HALF_WIDTH / dlon) + 1
    rel_lons = np.linspace(-LON_HALF_WIDTH, LON_HALF_WIDTH, n_rel_bins)
    
    # Collect daily tilt samples: (n_samples, lat, rel_lon)
    # At each point, compute: tilt = low_west - up_west
    all_low_west = []
    all_up_west = []
    n_samples = 0
    
    for _, ev in events.iterrows():
        start = ev['start_date']
        end = ev['end_date']
        
        mask = (time_mjo >= start) & (time_mjo <= end)
        indices = np.where(mask)[0]
        
        for idx in indices:
            clon = center_lon[idx]
            if not np.isfinite(clon):
                continue
            
            amp_today = mjo_amp[idx]
            if not np.isfinite(amp_today) or amp_today <= AMP_THRESHOLD:
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
            w_up_full = np.nanmean(daily_data[up_mask, :, :], axis=0)   # (lat, lon)
            
            # Sample to relative longitude grid
            clon_360 = np.mod(clon, 360)
            low_sample = np.zeros((len(lats), len(rel_lons)))
            up_sample = np.zeros((len(lats), len(rel_lons)))
            
            for j, rlon in enumerate(rel_lons):
                target_lon = np.mod(clon_360 + rlon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - target_lon))
                low_sample[:, j] = w_low_full[:, lon_idx]
                up_sample[:, j] = w_up_full[:, lon_idx]
            
            # Compute tilt at each latitude
            daily_low_west = np.full(len(lats), np.nan)
            daily_up_west = np.full(len(lats), np.nan)
            
            for lat_idx in range(len(lats)):
                lw, le, lc, lmin = _ascent_boundary_by_half_max(
                    rel_lons, low_sample[lat_idx, :], HALF_MAX_FRACTION, PIVOT_DELTA_DEG
                )
                uw, ue, uc, umin = _ascent_boundary_by_half_max(
                    rel_lons, up_sample[lat_idx, :], HALF_MAX_FRACTION, PIVOT_DELTA_DEG
                )
                daily_low_west[lat_idx] = lw
                daily_up_west[lat_idx] = uw
            
            all_low_west.append(daily_low_west)
            all_up_west.append(daily_up_west)
            n_samples += 1
    
    print(f"  Total daily samples: {n_samples}")
    
    # Average across all days
    all_low_west = np.array(all_low_west)  # (n_samples, lat)
    all_up_west = np.array(all_up_west)
    
    mean_low_west = np.nanmean(all_low_west, axis=0)  # (lat,)
    mean_up_west = np.nanmean(all_up_west, axis=0)
    
    # Tilt at each latitude
    mean_tilt = mean_low_west - mean_up_west  # (lat,)
    
    return mean_tilt, mean_low_west, mean_up_west, lats, n_samples


def plot_tilt_1d(mean_tilt, mean_low_west, mean_up_west, lats, n_samples):
    """Plot tilt as a function of latitude (1D profile)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # === Panel 1: Tilt vs Latitude ===
    ax1.plot(lats, mean_tilt, 'b-o', linewidth=2, markersize=6, label='Tilt')
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax1.axhline(np.nanmean(mean_tilt), color='r', linestyle='--', alpha=0.7, 
                label=f'Mean = {np.nanmean(mean_tilt):.2f}°')
    ax1.set_xlabel('Latitude (°)', fontsize=12)
    ax1.set_ylabel('Tilt (°)', fontsize=12)
    ax1.set_title(f'MJO Vertical Tilt vs Latitude (N={n_samples})\n'
                  f'Tilt = Low Layer West - Upper Layer West',
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # === Panel 2: West boundaries ===
    ax2.plot(lats, mean_low_west, 'b-o', linewidth=2, markersize=6, 
             label=f'Low Layer ({LOW_LAYER[0]}-{LOW_LAYER[1]} hPa)')
    ax2.plot(lats, mean_up_west, 'r-s', linewidth=2, markersize=6,
             label=f'Upper Layer ({UP_LAYER[0]}-{UP_LAYER[1]} hPa)')
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Latitude (°)', fontsize=12)
    ax2.set_ylabel('West Boundary (° relative to center)', fontsize=12)
    ax2.set_title('50% Half-Max West Boundaries by Latitude',
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    out_path = FIG_DIR / "mjo_tilt_by_latitude.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def create_tilt_field_2d(center_lon, mjo_amp, time_mjo, events):
    """
    Create full 2D tilt field (lat, rel_lon) by computing tilt 
    at each longitude slice using different center positions.
    
    This gives a map showing how tilt varies spatially.
    """
    print("\nCreating 2D tilt field...")
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
    
    # Collect composites
    all_low = []  # (n_samples, lat, rel_lon)
    all_up = []
    
    for _, ev in events.iterrows():
        start = ev['start_date']
        end = ev['end_date']
        
        mask = (time_mjo >= start) & (time_mjo <= end)
        indices = np.where(mask)[0]
        
        for idx in indices:
            clon = center_lon[idx]
            if not np.isfinite(clon):
                continue
            
            amp_today = mjo_amp[idx]
            if not np.isfinite(amp_today) or amp_today <= AMP_THRESHOLD:
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
            w_low_full = np.nanmean(daily_data[low_mask, :, :], axis=0)
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
            
            all_low.append(low_sample)
            all_up.append(up_sample)
    
    n_samples = len(all_low)
    print(f"  Total samples: {n_samples}")
    
    # Mean composites
    mean_low = np.nanmean(all_low, axis=0)  # (lat, rel_lon)
    mean_up = np.nanmean(all_up, axis=0)
    
    # Smooth
    mean_low = gaussian_filter(mean_low, sigma=SIGMA_SMOOTH)
    mean_up = gaussian_filter(mean_up, sigma=SIGMA_SMOOTH)
    
    return mean_low, mean_up, lats, rel_lons, n_samples


def plot_tilt_field_2d(mean_low, mean_up, lats, rel_lons, n_samples):
    """
    Plot 2D fields showing low-layer and upper-layer omega,
    plus the vertical structure comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    # Normalize
    vmax = np.nanpercentile(np.abs([mean_low, mean_up]), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Panel 1: Low layer omega
    ax1 = axes[0]
    X, Y = np.meshgrid(rel_lons, lats)
    cf1 = ax1.contourf(X, Y, mean_low, levels=21, cmap='RdBu_r', norm=norm)
    ax1.contour(X, Y, mean_low, levels=[0], colors='k', linewidths=1.5)
    ax1.set_xlabel('Relative Longitude (°)', fontsize=11)
    ax1.set_ylabel('Latitude (°)', fontsize=11)
    ax1.set_title(f'Low Layer ({LOW_LAYER[0]}-{LOW_LAYER[1]} hPa)\nω Composite',
                  fontsize=12, fontweight='bold')
    ax1.axvline(0, color='k', linestyle='--', alpha=0.5)
    fig.colorbar(cf1, ax=ax1, shrink=0.9, label='ω (Pa/s per unit amp)')
    
    # Panel 2: Upper layer omega
    ax2 = axes[1]
    cf2 = ax2.contourf(X, Y, mean_up, levels=21, cmap='RdBu_r', norm=norm)
    ax2.contour(X, Y, mean_up, levels=[0], colors='k', linewidths=1.5)
    ax2.set_xlabel('Relative Longitude (°)', fontsize=11)
    ax2.set_ylabel('Latitude (°)', fontsize=11)
    ax2.set_title(f'Upper Layer ({UP_LAYER[0]}-{UP_LAYER[1]} hPa)\nω Composite',
                  fontsize=12, fontweight='bold')
    ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
    fig.colorbar(cf2, ax=ax2, shrink=0.9, label='ω (Pa/s per unit amp)')
    
    # Panel 3: Tilt visualization - overlay west boundaries
    ax3 = axes[2]
    
    # Compute west boundaries at each latitude
    low_west_boundary = np.full(len(lats), np.nan)
    up_west_boundary = np.full(len(lats), np.nan)
    
    for lat_idx in range(len(lats)):
        lw, le, lc, lmin = _ascent_boundary_by_half_max(
            rel_lons, mean_low[lat_idx, :], HALF_MAX_FRACTION, PIVOT_DELTA_DEG
        )
        uw, ue, uc, umin = _ascent_boundary_by_half_max(
            rel_lons, mean_up[lat_idx, :], HALF_MAX_FRACTION, PIVOT_DELTA_DEG
        )
        low_west_boundary[lat_idx] = lw
        up_west_boundary[lat_idx] = uw
    
    tilt = low_west_boundary - up_west_boundary
    
    # Plot tilt as bar chart
    colors = ['red' if t > 0 else 'blue' for t in tilt]
    ax3.barh(lats, tilt, height=np.abs(lats[1]-lats[0])*0.8, color=colors, alpha=0.7)
    ax3.axvline(0, color='k', linewidth=1.5)
    ax3.set_xlabel('Tilt (°)', fontsize=11)
    ax3.set_ylabel('Latitude (°)', fontsize=11)
    ax3.set_title(f'MJO Tilt by Latitude (N={n_samples})\n'
                  f'Tilt = Low West - Upper West\nRed: positive (westward tilt)',
                  fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='x')
    
    # Add mean tilt annotation
    mean_tilt = np.nanmean(tilt)
    ax3.axvline(mean_tilt, color='green', linestyle='--', linewidth=2,
                label=f'Mean Tilt = {mean_tilt:.2f}°')
    ax3.legend(loc='upper right')
    
    plt.tight_layout()
    out_path = FIG_DIR / "mjo_tilt_2d_omega_composite.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()
    
    # Return tilt array for additional plotting
    return tilt, low_west_boundary, up_west_boundary


def plot_west_boundary_overlay(mean_low, mean_up, lats, rel_lons, n_samples):
    """
    Plot omega composite with west boundary lines overlaid.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Plot low layer as contourf
    vmax = np.nanpercentile(np.abs(mean_low), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    X, Y = np.meshgrid(rel_lons, lats)
    cf = ax.contourf(X, Y, mean_low, levels=21, cmap='RdBu_r', norm=norm, alpha=0.8)
    ax.contour(X, Y, mean_low, levels=[0], colors='k', linewidths=1)
    
    # Compute and plot boundaries
    low_west = np.full(len(lats), np.nan)
    up_west = np.full(len(lats), np.nan)
    
    for lat_idx in range(len(lats)):
        lw, _, _, _ = _ascent_boundary_by_half_max(
            rel_lons, mean_low[lat_idx, :], HALF_MAX_FRACTION, PIVOT_DELTA_DEG
        )
        uw, _, _, _ = _ascent_boundary_by_half_max(
            rel_lons, mean_up[lat_idx, :], HALF_MAX_FRACTION, PIVOT_DELTA_DEG
        )
        low_west[lat_idx] = lw
        up_west[lat_idx] = uw
    
    # Plot boundary lines
    ax.plot(low_west, lats, 'b-o', linewidth=3, markersize=8, 
            label=f'Low Layer West (1000-600 hPa)')
    ax.plot(up_west, lats, 'r-s', linewidth=3, markersize=8,
            label=f'Upper Layer West (400-200 hPa)')
    
    ax.axvline(0, color='k', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Relative Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title(f'MJO ω Composite with 50% Half-Max West Boundaries (N={n_samples})\n'
                 f'Shading: Low Layer ω | Lines: West Boundaries',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    
    cbar = fig.colorbar(cf, ax=ax, shrink=0.8)
    cbar.set_label('ω (Pa/s per unit amplitude)', fontsize=10)
    
    plt.tight_layout()
    out_path = FIG_DIR / "mjo_tilt_west_boundary_overlay.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()
    
    return low_west, up_west


def main():
    print("=" * 70)
    print("MJO 2D Tilt Map Visualization")
    print("=" * 70)
    
    # Load MJO data
    center_lon, mjo_amp, time_mjo, events = load_mjo_data()
    
    # Create 2D omega composites
    mean_low, mean_up, lats, rel_lons, n_samples = create_tilt_field_2d(
        center_lon, mjo_amp, time_mjo, events
    )
    
    # Plot omega composites + tilt bar chart
    print("\nPlotting 2D tilt field...")
    tilt, low_west, up_west = plot_tilt_field_2d(
        mean_low, mean_up, lats, rel_lons, n_samples
    )
    
    # Plot west boundary overlay
    print("\nPlotting west boundary overlay...")
    plot_west_boundary_overlay(mean_low, mean_up, lats, rel_lons, n_samples)
    
    # Plot 1D tilt profile
    print("\nPlotting 1D tilt profile...")
    mean_tilt = low_west - up_west
    plot_tilt_1d(mean_tilt, low_west, up_west, lats, n_samples)
    
    # Print stats
    print("\n" + "=" * 70)
    print("Tilt Statistics:")
    print(f"  Mean: {np.nanmean(mean_tilt):.2f}°")
    print(f"  Std:  {np.nanstd(mean_tilt):.2f}°")
    print(f"  Min:  {np.nanmin(mean_tilt):.2f}° (at {lats[np.nanargmin(mean_tilt)]:.1f}°)")
    print(f"  Max:  {np.nanmax(mean_tilt):.2f}° (at {lats[np.nanargmax(mean_tilt)]:.1f}°)")
    print("=" * 70)


if __name__ == "__main__":
    main()
