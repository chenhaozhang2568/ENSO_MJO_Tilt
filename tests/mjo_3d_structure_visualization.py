# -*- coding: utf-8 -*-
"""
mjo_3d_structure_visualization.py: MJO 三维结构可视化

================================================================================
功能描述：
    本脚本生成 MJO 三维结构的多视角可视化图：
    1. 经度-高度剖面图（4 张）：u, q, omega, T
    2. 三维立体图（4 张）：u, q, omega, T

方法：
    - 以 MJO 对流中心为原点，提取相对经度范围 [-60°, +60°] 的数据
    - 按 MJO 振幅归一化（Hu & Li 2021 方法）
    - 对所有 MJO 事件的所有天数取平均，得到合成场

数据源：
    - 归一化重构场: era5_mjo_recon_{var}_norm_1979-2022.nc
    - MJO 事件列表: mjo_events_step3_1979-2022.csv
    - MJO 中心追踪: mjo_mvEOF_step3_1979-2022.nc

Run:
    python E:\\Projects\\ENSO_MJO_Tilt\\tests\\mjo_3d_structure_visualization.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from scipy import stats
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")

# ========================
# PATHS
# ========================
DERIVED_DIR = Path(r"E:\Datas\Derived")
U_RECON_NC = DERIVED_DIR / "era5_mjo_recon_u_norm_1979-2022.nc"
Q_RECON_NC = DERIVED_DIR / "era5_mjo_recon_q_norm_1979-2022.nc"
W_RECON_NC = DERIVED_DIR / "era5_mjo_recon_w_norm_1979-2022.nc"
T_RECON_NC = DERIVED_DIR / "era5_mjo_recon_t_norm_1979-2022.nc"
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\mjo_3d_structure")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# SETTINGS
# ========================
LON_HALF_WIDTH = 60  # degrees from convective center
AMP_THRESHOLD = 0.5  # minimum MJO amplitude to include
SIGMA_SMOOTH = 1.0   # Gaussian smoothing sigma

# Variable configurations
VAR_CONFIG = {
    'u': {
        'file': U_RECON_NC,
        'var_name': 'u_mjo_recon_norm',
        'label': 'Zonal Wind (u)',
        'unit': 'm/s per unit amplitude',
        'cmap': 'RdBu_r',
    },
    'q': {
        'file': Q_RECON_NC,
        'var_name': 'q_mjo_recon_norm',
        'label': 'Specific Humidity (q)',
        'unit': 'g/kg per unit amplitude',
        'cmap': 'BrBG',
    },
    'omega': {
        'file': W_RECON_NC,
        'var_name': 'w_mjo_recon_norm',
        'label': 'Vertical Velocity (ω)',
        'unit': 'Pa/s per unit amplitude',
        'cmap': 'RdBu_r',
    },
    'T': {
        'file': T_RECON_NC,
        'var_name': 't_mjo_recon_norm',
        'label': 'Temperature (T)',
        'unit': 'K per unit amplitude',
        'cmap': 'RdYlBu_r',
    },
}


def load_mjo_data():
    """Load MJO center track and amplitude data."""
    print("Loading MJO tracking data...")
    ds = xr.open_dataset(STEP3_NC)
    center_lon = ds['center_lon_track'].values
    mjo_amp = ds['amp'].values
    time_mjo = pd.to_datetime(ds.time.values)
    ds.close()
    
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])
    print(f"  Loaded {len(events)} MJO events.")
    
    return center_lon, mjo_amp, time_mjo, events


def load_variable_data(var_key):
    """Load reconstructed and normalized variable data."""
    config = VAR_CONFIG[var_key]
    print(f"Loading {var_key} data from {config['file'].name}...")
    ds = xr.open_dataset(config['file'])
    data = ds[config['var_name']]
    
    # Rename pressure_level to level for consistency
    if "pressure_level" in data.dims:
        data = data.rename({"pressure_level": "level"})
    
    time_data = pd.to_datetime(data.time.values)
    levels = data.level.values if 'level' in data.dims else data.pressure_level.values
    lon = data.lon.values
    
    return data, time_data, levels, lon


def create_composite(var_key, center_lon, mjo_amp, time_mjo, events):
    """Create composite field for all MJO events.
    
    Returns:
        mean_comp: Mean composite field (level, rel_lon)
        mean_comp_std: Standardized composite
        significant: Significance mask (p < 0.05)
        rel_lons: Relative longitude array
        levels: Pressure levels
    """
    data, time_data, levels, lon = load_variable_data(var_key)
    
    lon_360 = np.mod(lon, 360)
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int(2 * LON_HALF_WIDTH / dlon) + 1
    rel_lons = np.linspace(-LON_HALF_WIDTH, LON_HALF_WIDTH, n_rel_bins)
    
    all_samples = []
    
    for _, ev in events.iterrows():
        start = ev['start_date']
        end = ev['end_date']
        
        mask = (time_mjo >= start) & (time_mjo <= end)
        indices = np.where(mask)[0]
        
        for idx in indices:
            clon = center_lon[idx]
            if not np.isfinite(clon):
                continue
            
            # Check amplitude threshold
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
            
            daily_data = data.isel(time=data_idx).values  # (level, lon)
            if np.all(np.isnan(daily_data)):
                continue
            
            # Normalize by MJO amplitude (Hu & Li 2021 method)
            daily_data = daily_data / amp_today
            
            # Sample at relative longitudes
            sample = np.zeros((len(levels), len(rel_lons)))
            clon_360 = np.mod(clon, 360)
            
            for j, rlon in enumerate(rel_lons):
                target_lon = np.mod(clon_360 + rlon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - target_lon))
                sample[:, j] = daily_data[:, lon_idx]
            
            all_samples.append(sample)
    
    all_samples = np.array(all_samples)  # (n_samples, level, rel_lon)
    n_samples = len(all_samples)
    print(f"  {var_key}: {n_samples} daily samples from {len(events)} events")
    
    # Compute mean composite
    mean_comp = np.nanmean(all_samples, axis=0)
    
    # Apply Gaussian smoothing
    mean_comp = gaussian_filter(mean_comp, sigma=SIGMA_SMOOTH)
    
    # Standardize
    std_comp = np.nanstd(all_samples, axis=0)
    std_comp[std_comp == 0] = np.nan
    
    # T-test for significance
    n = np.sum(~np.isnan(all_samples), axis=0)
    se = std_comp / np.sqrt(n)
    t_stat = mean_comp / se
    
    from scipy.stats import t as t_dist
    p_val = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=np.maximum(n-1, 1)))
    significant = p_val < 0.05
    
    # Standardized anomaly for plotting
    overall_std = np.nanstd(all_samples)
    mean_comp_std = mean_comp / overall_std if overall_std > 0 else mean_comp
    
    return mean_comp, mean_comp_std, significant, rel_lons, levels, n_samples


def pressure_to_height(p_hpa):
    """Convert pressure (hPa) to approximate height (km) using barometric formula.
    
    Uses scale height H = 7.5 km (typical for troposphere).
    z = H * ln(p0 / p), where p0 = 1013.25 hPa
    """
    p0 = 1013.25  # reference pressure at sea level (hPa)
    H = 7.5  # scale height (km)
    return H * np.log(p0 / p_hpa)


def plot_lon_height_profile(var_key, mean_comp, mean_comp_std, significant, rel_lons, levels, n_samples):
    """Plot longitude-height cross-section with uniform height axis.
    
    Key features:
    - Y-axis is uniform in HEIGHT (km), not pressure
    - Left axis shows pressure labels
    - Right axis shows height labels
    """
    config = VAR_CONFIG[var_key]
    
    # Convert pressure levels to height
    heights = pressure_to_height(levels)  # km
    
    # Create uniform height grid for interpolation
    height_uniform = np.linspace(heights.min(), heights.max(), 50)
    
    # Interpolate data to uniform height grid
    from scipy.interpolate import interp1d
    mean_comp_interp = np.zeros((len(height_uniform), len(rel_lons)))
    significant_interp = np.zeros((len(height_uniform), len(rel_lons)), dtype=bool)
    
    for j in range(len(rel_lons)):
        # Interpolate mean composite
        f = interp1d(heights, mean_comp_std[:, j], kind='linear', bounds_error=False, fill_value=np.nan)
        mean_comp_interp[:, j] = f(height_uniform)
        
        # Interpolate significance (use nearest neighbor)
        f_sig = interp1d(heights, significant[:, j].astype(float), kind='nearest', bounds_error=False, fill_value=0)
        significant_interp[:, j] = f_sig(height_uniform) > 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Color range
    vmax = np.nanpercentile(np.abs(mean_comp_interp), 95)
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Contour fill (on uniform height grid)
    cf = ax.contourf(rel_lons, height_uniform, mean_comp_interp, levels=21, 
                     cmap=config['cmap'], norm=norm, extend='both')
    
    # Contour lines
    ax.contour(rel_lons, height_uniform, mean_comp_interp, levels=11, colors='k', 
               linewidths=0.5, alpha=0.5)
    
    # Significance stippling
    xx, yy = np.meshgrid(rel_lons, height_uniform)
    ax.scatter(xx[significant_interp], yy[significant_interp], s=1, c='k', alpha=0.3, marker='.')
    
    # Convective center line
    ax.axvline(0, color='green', linestyle='-', linewidth=2, label='Convective Center')
    
    # === LEFT Y-AXIS: Pressure ===
    ax.set_ylabel("Height (km)", fontsize=12)
    ax.set_xlabel("Relative Longitude (°)", fontsize=12)
    
    # Add pressure labels on LEFT side using secondary ticks
    # Map pressure levels to their corresponding heights
    pressure_ticks = [1000, 850, 700, 500, 300, 200, 100]
    pressure_heights = [pressure_to_height(p) for p in pressure_ticks]
    
    # Filter to only include pressures within our height range
    valid_idx = [i for i, h in enumerate(pressure_heights) 
                 if height_uniform.min() <= h <= height_uniform.max()]
    pressure_ticks = [pressure_ticks[i] for i in valid_idx]
    pressure_heights = [pressure_heights[i] for i in valid_idx]
    
    # Create secondary axis for pressure (left side)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(pressure_heights)
    ax2.set_yticklabels([f"{p}" for p in pressure_ticks])
    ax2.set_ylabel("Pressure (hPa)", fontsize=12)
    
    # Move right axis to left
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.tick_left()
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    
    ax.set_title(f"MJO {config['label']} Composite (N={n_samples} samples)\n"
                 f"Shading: standardized anomaly; dots: p<0.05", 
                 fontsize=13, fontweight='bold')
    ax.set_xlim(-LON_HALF_WIDTH, LON_HALF_WIDTH)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(linestyle='--', alpha=0.3)
    
    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.12)
    cbar.set_label(f'Standardized {config["label"]} Anomaly', fontsize=10)
    
    out_path = FIG_DIR / f"mjo_composite_{var_key}_lon_height.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def plot_3d_structure(var_key, mean_comp, rel_lons, levels):
    """Plot 3D visualization using stacked horizontal slices.
    
    Since data is zonally-averaged (no latitude dimension), we display:
    - X axis: Relative Longitude
    - Y axis: Pseudo "spread" axis (for 3D effect)  
    - Z axis: Height (converted from pressure)
    
    Each pressure level is shown as a semi-transparent horizontal slice,
    with color indicating variable magnitude. This avoids occlusion issues.
    """
    config = VAR_CONFIG[var_key]
    
    # Convert pressure to height
    heights = pressure_to_height(levels)
    
    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize for coloring
    vmax = np.nanpercentile(np.abs(mean_comp), 95)
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Get colormap
    cmap = plt.get_cmap(config['cmap'])
    
    # Create pseudo Y-axis for 3D spread effect
    # Using a narrow range since data is zonally averaged
    y_spread = np.linspace(-5, 5, 11)  # Pseudo "latitude" spread, just for visualization
    
    # Plot each pressure level as a semi-transparent horizontal slice
    for i, (level, height) in enumerate(zip(levels, heights)):
        data_slice = mean_comp[i, :]  # (lon,)
        
        # Create mesh for this level
        X, Y = np.meshgrid(rel_lons, y_spread)
        Z_height = np.full_like(X, height)  # Constant height for this level
        
        # Replicate data across Y dimension (since zonally averaged)
        C = np.tile(data_slice, (len(y_spread), 1))
        
        # Apply colormap
        colors = cmap(norm(C))
        
        # Plot surface with transparency
        # Lower levels are more transparent to reduce occlusion
        alpha = 0.3 + 0.5 * (i / (len(levels) - 1))  # 0.3 to 0.8
        ax.plot_surface(X, Y, Z_height, facecolors=colors, 
                       linewidth=0, antialiased=True, alpha=alpha, shade=False)
        
        # Add contour on each slice for structure visibility
        if i % 2 == 0:  # Every other level
            contour_levels = 5
            ax.contour(X, Y, C, zdir='z', offset=height, 
                      levels=contour_levels, colors='k', linewidths=0.3, alpha=0.5)
    
    # === Add vertical curtain at Y=0 for clearer 2D structure ===
    X_curtain, Z_curtain = np.meshgrid(rel_lons, heights)
    Y_curtain = np.zeros_like(X_curtain)
    C_curtain = mean_comp
    colors_curtain = cmap(norm(C_curtain))
    ax.plot_surface(X_curtain, Y_curtain, Z_curtain, facecolors=colors_curtain,
                   linewidth=0, antialiased=True, alpha=0.85, shade=False)
    
    # Labels
    ax.set_xlabel('\nRelative Longitude (°)', fontsize=12)
    ax.set_ylabel('\n← West    East →', fontsize=10)  # Pseudo axis
    ax.set_zlabel('\nHeight (km)', fontsize=12)
    ax.set_title(f"3D MJO {config['label']} Structure\n(Horizontal slices at each pressure level)", 
                 fontsize=14, fontweight='bold')
    
    # Set axis limits
    ax.set_xlim(-LON_HALF_WIDTH, LON_HALF_WIDTH)
    ax.set_ylim(-5, 5)
    ax.set_zlim(heights.min(), heights.max())
    
    # Hide Y axis ticks (pseudo axis)
    ax.set_yticks([])
    
    # Add pressure labels on Z axis
    z_ticks = [pressure_to_height(p) for p in [1000, 700, 500, 300, 200]]
    z_labels = ['1000', '700', '500', '300', '200']
    ax.set_zticks(z_ticks)
    ax.set_zticklabels([f'{z:.1f} km\n({p} hPa)' for z, p in zip(z_ticks, [1000, 700, 500, 300, 200])])
    
    # Add colorbar using ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(config['unit'], fontsize=10)
    
    # Adjust view angle for better visibility
    ax.view_init(elev=20, azim=-60)
    
    out_path = FIG_DIR / f"mjo_3d_{var_key}.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


# ========================================
# NEW: 3D Visualization with Latitude
# ========================================
VAR_CONFIG_3D = {
    'u': {
        'file': DERIVED_DIR / "era5_mjo_recon_u_norm_3d_1979-2022.nc",
        'var_name': 'u_mjo_recon_norm_3d',
        'label': 'Zonal Wind (u)',
        'unit': 'm/s per unit amp',
        'cmap': 'RdBu_r',
    },
    'q': {
        'file': DERIVED_DIR / "era5_mjo_recon_q_norm_3d_1979-2022.nc",
        'var_name': 'q_mjo_recon_norm_3d',
        'label': 'Specific Humidity (q)',
        'unit': 'g/kg per unit amp',
        'cmap': 'BrBG',
    },
    'omega': {
        'file': DERIVED_DIR / "era5_mjo_recon_w_norm_3d_1979-2022.nc",
        'var_name': 'w_mjo_recon_norm_3d',
        'label': 'Vertical Velocity (ω)',
        'unit': 'Pa/s per unit amp',
        'cmap': 'RdBu_r',
    },
    'T': {
        'file': DERIVED_DIR / "era5_mjo_recon_t_norm_3d_1979-2022.nc",
        'var_name': 't_mjo_recon_norm_3d',
        'label': 'Temperature (T)',
        'unit': 'K per unit amp',
        'cmap': 'RdYlBu_r',
    },
}


def create_composite_3d(var_key, center_lon, mjo_amp, time_mjo, events):
    """Create 3D composite field (with latitude) for all MJO events.
    
    Returns:
        mean_comp: Mean composite field (level, lat, rel_lon)
        rel_lons: Relative longitude array
        lats: Latitude array
        levels: Pressure levels
    """
    config = VAR_CONFIG_3D[var_key]
    
    # Check if 3D file exists
    if not config['file'].exists():
        print(f"  [WARNING] 3D file not found: {config['file']}")
        return None, None, None, None, 0
    
    print(f"Loading 3D {var_key} data...")
    ds = xr.open_dataset(config['file'])
    data = ds[config['var_name']]  # (time, level, lat, lon)
    
    if "pressure_level" in data.dims:
        data = data.rename({"pressure_level": "level"})
    
    time_data = pd.to_datetime(data.time.values)
    levels = data.level.values
    lats = data.lat.values
    lon = data.lon.values
    lon_360 = np.mod(lon, 360)
    
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int(2 * LON_HALF_WIDTH / dlon) + 1
    rel_lons = np.linspace(-LON_HALF_WIDTH, LON_HALF_WIDTH, n_rel_bins)
    
    all_samples = []
    
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
            
            # Sample at relative longitudes
            sample = np.zeros((len(levels), len(lats), len(rel_lons)))
            clon_360 = np.mod(clon, 360)
            
            for j, rlon in enumerate(rel_lons):
                target_lon = np.mod(clon_360 + rlon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - target_lon))
                sample[:, :, j] = daily_data[:, :, lon_idx]
            
            all_samples.append(sample)
    
    all_samples = np.array(all_samples)  # (n_samples, level, lat, rel_lon)
    n_samples = len(all_samples)
    print(f"  {var_key} 3D: {n_samples} daily samples")
    
    mean_comp = np.nanmean(all_samples, axis=0)
    mean_comp = gaussian_filter(mean_comp, sigma=SIGMA_SMOOTH)
    
    return mean_comp, rel_lons, lats, levels, n_samples


def plot_3d_structure_with_lat(var_key, mean_comp, rel_lons, lats, levels, n_samples):
    """Plot TRUE 3D visualization with longitude, latitude, and height.
    
    Uses horizontal slices at key pressure levels with transparency,
    plus vertical cross-sections for clearer structure.
    """
    config = VAR_CONFIG_3D[var_key]
    
    heights = pressure_to_height(levels)
    
    fig = plt.figure(figsize=(14, 10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize for coloring
    vmax = np.nanpercentile(np.abs(mean_comp), 95)
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap(config['cmap'])
    
    # Plot horizontal slices at selected levels
    selected_levels = [0, 2, 4, 6, 8]  # Indices of levels to show
    
    for i_level in selected_levels:
        if i_level >= len(levels):
            continue
        
        height = heights[i_level]
        data_slice = mean_comp[i_level, :, :]  # (lat, rel_lon)
        
        X, Y = np.meshgrid(rel_lons, lats)
        Z = np.full_like(X, height)
        
        colors = cmap(norm(data_slice))
        alpha = 0.4 + 0.4 * (i_level / (len(levels) - 1))
        
        ax.plot_surface(X, Y, Z, facecolors=colors,
                       linewidth=0, antialiased=True, alpha=alpha, shade=False)
    
    # Add vertical cross-section at equator (lat=0)
    eq_idx = np.argmin(np.abs(lats))
    X_cs, Z_cs = np.meshgrid(rel_lons, heights)
    Y_cs = np.full_like(X_cs, lats[eq_idx])
    data_cs = mean_comp[:, eq_idx, :]
    colors_cs = cmap(norm(data_cs))
    ax.plot_surface(X_cs, Y_cs, Z_cs, facecolors=colors_cs,
                   linewidth=0, antialiased=True, alpha=0.7, shade=False)
    
    # Add vertical cross-section at center longitude
    cen_lon_idx = len(rel_lons) // 2
    Y_cs2, Z_cs2 = np.meshgrid(lats, heights)
    X_cs2 = np.full_like(Y_cs2, rel_lons[cen_lon_idx])
    data_cs2 = mean_comp[:, :, cen_lon_idx]
    colors_cs2 = cmap(norm(data_cs2))
    ax.plot_surface(X_cs2, Y_cs2, Z_cs2, facecolors=colors_cs2,
                   linewidth=0, antialiased=True, alpha=0.7, shade=False)
    
    # Labels and formatting
    ax.set_xlabel('\nRelative Longitude (°)', fontsize=12)
    ax.set_ylabel('\nLatitude (°)', fontsize=12)
    ax.set_zlabel('\nHeight (km)', fontsize=12)
    ax.set_title(f"3D MJO {config['label']} Structure (N={n_samples})\n"
                 f"Horizontal slices + equatorial & center cross-sections",
                 fontsize=13, fontweight='bold')
    
    ax.set_xlim(-LON_HALF_WIDTH, LON_HALF_WIDTH)
    ax.set_ylim(lats.min(), lats.max())
    ax.set_zlim(heights.min(), heights.max())
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.set_label(config['unit'], fontsize=10)
    
    ax.view_init(elev=25, azim=-50)
    
    out_path = FIG_DIR / f"mjo_3d_{var_key}_with_lat.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    """Main function to generate all MJO 3D structure visualizations."""
    print("=" * 70)
    print("MJO 3D Structure Visualization")
    print("=" * 70)
    
    # Load MJO tracking data
    center_lon, mjo_amp, time_mjo, events = load_mjo_data()
    
    # ========================================
    # Part 1: Original lon-height profile + pseudo-3D (zonally-averaged)
    # ========================================
    print("\n" + "=" * 70)
    print("Part 1: Longitude-Height Profiles (Zonally-Averaged)")
    print("=" * 70)
    
    for var_key in ['u', 'q', 'omega', 'T']:
        print(f"\n{'='*40}")
        print(f"Processing: {var_key}")
        print('='*40)
        
        # Create composite
        mean_comp, mean_comp_std, significant, rel_lons, levels, n_samples = \
            create_composite(var_key, center_lon, mjo_amp, time_mjo, events)
        
        # Plot longitude-height profile
        plot_lon_height_profile(var_key, mean_comp, mean_comp_std, significant, 
                               rel_lons, levels, n_samples)
        
        # Plot pseudo-3D structure (zonally-averaged)
        plot_3d_structure(var_key, mean_comp, rel_lons, levels)
    
    # ========================================
    # Part 2: TRUE 3D Visualization (with latitude)
    # ========================================
    print("\n" + "=" * 70)
    print("Part 2: TRUE 3D Visualization (Lon-Lat-Height)")
    print("=" * 70)
    
    for var_key in ['u', 'q', 'omega', 'T']:
        print(f"\n{'='*40}")
        print(f"Processing 3D: {var_key}")
        print('='*40)
        
        # Create 3D composite
        mean_comp_3d, rel_lons_3d, lats, levels_3d, n_samples_3d = \
            create_composite_3d(var_key, center_lon, mjo_amp, time_mjo, events)
        
        if mean_comp_3d is None:
            print(f"  Skipping {var_key} - 3D data not available")
            continue
        
        # Plot TRUE 3D structure
        plot_3d_structure_with_lat(var_key, mean_comp_3d, rel_lons_3d, lats, levels_3d, n_samples_3d)
    
    print("\n" + "=" * 70)
    print("All visualizations completed!")
    print(f"Figures saved to: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

