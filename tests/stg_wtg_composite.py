# -*- coding: utf-8 -*-
"""
stg_wtg_composite.py: STG vs WTG 垂直结构合成分析

================================================================================
功能描述：
    本脚本将 MJO 事件按倾斜强度分为两组，对比分析其垂直结构差异。

分组定义（基于标准差阈值）：
    - STG（Strong Tilt Group，强倾斜组）: tilt > mean + 0.7×std
    - WTG（Weak Tilt Group，弱倾斜组）: tilt < mean - 0.7×std

主要分析内容：
    1. omega（垂直速度）的高度-经度合成剖面图
    2. 纬向风（u）的垂直结构对比
    3. STG 和 WTG 的相速度差异统计
    4. 倾斜-相速度散点图

科学意义：
    通过对比强倾斜和弱倾斜事件的环流结构，揭示倾斜对 MJO 传播特性的影响。
- Standardized anomaly vertical velocity (shading)
- Zonal and vertical velocity (vectors)

Compare phase speeds between groups and create scatter plot.

Run:
  python E:\\Projects\\ENSO_MJO_Tilt\\tests\\stg_wtg_composite.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ========================
# PATHS
# ========================
W_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
U_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_u_norm_1979-2022.nc"  # Normalized u for shear analysis
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\stg_wtg")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# SETTINGS
# ========================
TILT_THRESHOLD = 0.7  # standard deviations


def load_data():
    """Load MJO-reconstructed omega and u data"""
    print("Loading data...")
    
    # Load omega (w) - MJO reconstructed and normalized
    ds_w = xr.open_dataset(W_RECON_NC)
    w_recon = ds_w['w_mjo_recon_norm']  # (time, pressure_level, lon) - already normalized
    # Rename pressure_level to level for consistency
    if "pressure_level" in w_recon.dims:
        w_recon = w_recon.rename({"pressure_level": "level"})
    
    # Load u - MJO reconstructed and normalized
    ds_u = xr.open_dataset(U_RECON_NC)
    u_recon = ds_u['u_mjo_recon_norm']  # (time, pressure_level, lon) - already normalized
    if "pressure_level" in u_recon.dims:
        u_recon = u_recon.rename({"pressure_level": "level"})
    
    # Load MJO center track and amplitude
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon = ds3['center_lon_track'].values
    time_mjo = pd.to_datetime(ds3.time.values)
    
    # Load MJO amplitude for normalization (Hu & Li 2021 method)
    mjo_amp = ds3['amp'].values  # sqrt(PC1^2 + PC2^2)
    
    # Load events and tilt stats
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])
    enso_stats = pd.read_csv(ENSO_STATS_CSV)
    
    # Calculate phase speed for each event
    phase_speeds = []
    for _, ev in events.iterrows():
        eid = ev['event_id']
        start = ev['start_date']
        end = ev['end_date']
        
        mask = (time_mjo >= start) & (time_mjo <= end)
        lons = center_lon[mask]
        days = np.arange(len(lons))
        
        valid = np.isfinite(lons)
        if valid.sum() < 5:
            phase_speeds.append({'event_id': eid, 'phase_speed_degday': np.nan, 'phase_speed_ms': np.nan})
            continue
        
        slope, _, _, _, _ = stats.linregress(days[valid], lons[valid])
        speed_ms = slope * 111e3 / 86400
        phase_speeds.append({'event_id': eid, 'phase_speed_degday': slope, 'phase_speed_ms': speed_ms})
    
    speed_df = pd.DataFrame(phase_speeds)
    enso_stats = enso_stats.merge(speed_df, on='event_id')
    
    # Define STG and WTG
    tilt_mean = enso_stats['mean_tilt'].mean()
    tilt_std = enso_stats['mean_tilt'].std()
    
    stg_threshold = tilt_mean + TILT_THRESHOLD * tilt_std
    wtg_threshold = tilt_mean - TILT_THRESHOLD * tilt_std
    
    enso_stats['group'] = 'Normal'
    enso_stats.loc[enso_stats['mean_tilt'] > stg_threshold, 'group'] = 'STG'
    enso_stats.loc[enso_stats['mean_tilt'] < wtg_threshold, 'group'] = 'WTG'
    
    print(f"\n  Tilt mean: {tilt_mean:.2f}, std: {tilt_std:.2f}")
    print(f"  STG threshold (> {stg_threshold:.2f}): {(enso_stats['group'] == 'STG').sum()} events")
    print(f"  WTG threshold (< {wtg_threshold:.2f}): {(enso_stats['group'] == 'WTG').sum()} events")
    print(f"  Normal: {(enso_stats['group'] == 'Normal').sum()} events")
    
    w_time = pd.to_datetime(w_recon.time.values)
    levels = w_recon.level.values if 'level' in w_recon.dims else w_recon.pressure_level.values
    lon = w_recon.lon.values
    
    return {
        'w_recon': w_recon,
        'w_time': w_time,
        'levels': levels,
        'lon': lon,
        'center_lon': center_lon,
        'time_mjo': time_mjo,
        'mjo_amp': mjo_amp,  # MJO amplitude for normalization
        'events': events,
        'enso_stats': enso_stats,
    }


def create_composite(data, group_name, lon_half_width=60, normalize_by_amp=True):
    """Create omega composite for a given group
    
    Args:
        normalize_by_amp: If True, divide omega by MJO amplitude (Hu & Li 2021 method)
    """
    w_recon = data['w_recon']
    w_time = data['w_time']
    levels = data['levels']
    lon = data['lon']
    center_lon = data['center_lon']
    time_mjo = data['time_mjo']
    mjo_amp = data['mjo_amp']  # MJO amplitude
    events = data['events']
    enso_stats = data['enso_stats']
    
    # Get events for this group
    group_events = enso_stats[enso_stats['group'] == group_name]['event_id'].values
    
    lon_360 = np.mod(lon, 360)
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int(2 * lon_half_width / dlon) + 1
    rel_lons = np.linspace(-lon_half_width, lon_half_width, n_rel_bins)
    
    # Collect all daily samples
    all_samples = []
    
    for eid in group_events:
        ev = events[events['event_id'] == eid].iloc[0]
        start = ev['start_date']
        end = ev['end_date']
        
        mask = (time_mjo >= start) & (time_mjo <= end)
        indices = np.where(mask)[0]
        
        for idx in indices:
            clon = center_lon[idx]
            if not np.isfinite(clon):
                continue
            
            t = time_mjo[idx]
            try:
                w_idx = np.where(w_time == t)[0]
                if len(w_idx) == 0:
                    continue
                w_idx = w_idx[0]
            except:
                continue
            
            w_day = w_recon.isel(time=w_idx).values  # (level, lon)
            if np.all(np.isnan(w_day)):
                continue
            
            # Normalize by MJO amplitude (Hu & Li 2021 method)
            # This allows fair comparison across events with different intensities
            amp_today = mjo_amp[idx]
            if normalize_by_amp and np.isfinite(amp_today) and amp_today > 0.5:
                w_day = w_day / amp_today
            elif normalize_by_amp and (not np.isfinite(amp_today) or amp_today <= 0.5):
                continue  # Skip days with too weak amplitude
            
            # Sample at relative longitudes
            sample = np.zeros((len(levels), len(rel_lons)))
            clon_360 = np.mod(clon, 360)
            
            for j, rlon in enumerate(rel_lons):
                target_lon = np.mod(clon_360 + rlon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - target_lon))
                sample[:, j] = w_day[:, lon_idx]
            
            all_samples.append(sample)
    
    all_samples = np.array(all_samples)  # (n_samples, level, rel_lon)
    
    # Compute mean and t-test
    mean_comp = np.nanmean(all_samples, axis=0)
    
    # Standardize by dividing by temporal std
    std_comp = np.nanstd(all_samples, axis=0)
    std_comp[std_comp == 0] = np.nan
    
    # T-test for significance
    n = np.sum(~np.isnan(all_samples), axis=0)
    se = std_comp / np.sqrt(n)
    t_stat = mean_comp / se
    
    # 95% confidence (two-tailed, df ~ n-1)
    from scipy.stats import t as t_dist
    p_val = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=np.maximum(n-1, 1)))
    significant = p_val < 0.05
    
    # Standardized anomaly
    w_std_all = np.nanstd(all_samples)
    mean_comp_std = mean_comp / w_std_all
    
    print(f"  {group_name}: {len(all_samples)} daily samples from {len(group_events)} events")
    
    return mean_comp, mean_comp_std, significant, rel_lons, levels, len(group_events)


def plot_composites(data):
    """Plot STG and WTG composites"""
    print("\nCreating composites...")
    
    stg_mean, stg_std, stg_sig, rel_lons, levels, n_stg = create_composite(data, 'STG')
    wtg_mean, wtg_std, wtg_sig, rel_lons, levels, n_wtg = create_composite(data, 'WTG')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharey=True)
    
    # Common colorbar range
    vmax = max(np.nanpercentile(np.abs(stg_std), 95), np.nanpercentile(np.abs(wtg_std), 95))
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    # Panel a: STG
    ax1 = axes[0]
    cf1 = ax1.contourf(rel_lons, levels, stg_std, levels=20, cmap='RdBu_r', norm=norm, extend='both')
    ax1.contour(rel_lons, levels, stg_std, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    
    # Stippling for significance
    xx, yy = np.meshgrid(rel_lons, levels)
    ax1.scatter(xx[stg_sig], yy[stg_sig], s=1, c='k', alpha=0.3, marker='.')
    
    ax1.axvline(0, color='green', linestyle='-', linewidth=2, label='Convective Center')
    ax1.set_xlabel("Relative Longitude (°)")
    ax1.set_ylabel("Pressure Level (hPa)")
    ax1.set_title(f"(a) STG (N={n_stg} events)", fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.set_xlim(-60, 60)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(linestyle='--', alpha=0.3)
    
    # Panel b: WTG
    ax2 = axes[1]
    cf2 = ax2.contourf(rel_lons, levels, wtg_std, levels=20, cmap='RdBu_r', norm=norm, extend='both')
    ax2.contour(rel_lons, levels, wtg_std, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    
    ax2.scatter(xx[wtg_sig], yy[wtg_sig], s=1, c='k', alpha=0.3, marker='.')
    
    ax2.axvline(0, color='green', linestyle='-', linewidth=2)
    ax2.set_xlabel("Relative Longitude (°)")
    ax2.set_title(f"(b) WTG (N={n_wtg} events)", fontsize=12, fontweight='bold')
    ax2.set_xlim(-60, 60)
    ax2.grid(linestyle='--', alpha=0.3)
    
    # Colorbar
    cbar = fig.colorbar(cf1, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('Standardized Vertical Velocity Anomaly', fontsize=10)
    
    plt.suptitle("Vertical Velocity Composite: STG vs WTG\n(Shading: standardized ω anomaly; dots: p<0.05)",
                 fontsize=13, fontweight='bold', y=1.02)
    
    out_path = FIG_DIR / "stg_wtg_omega_composite.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def analyze_phase_speed(data):
    """Compare phase speed between STG and WTG"""
    print("\n" + "="*60)
    print("Phase Speed Comparison: STG vs WTG")
    print("="*60)
    
    enso_stats = data['enso_stats']
    
    stg = enso_stats[enso_stats['group'] == 'STG']
    wtg = enso_stats[enso_stats['group'] == 'WTG']
    
    stg_speed = stg['phase_speed_ms'].dropna()
    wtg_speed = wtg['phase_speed_ms'].dropna()
    
    print(f"\n[STG] N={len(stg_speed)}, Mean speed: {stg_speed.mean():.2f} m/s, Std: {stg_speed.std():.2f}")
    print(f"[WTG] N={len(wtg_speed)}, Mean speed: {wtg_speed.mean():.2f} m/s, Std: {wtg_speed.std():.2f}")
    
    t, p = stats.ttest_ind(stg_speed, wtg_speed, equal_var=False)
    sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    print(f"\n[T-test] t={t:+.3f}, p={p:.4f} {sig}")
    
    if stg_speed.mean() < wtg_speed.mean():
        print("\n✅ STG (stronger tilt) has SLOWER propagation speed")
    else:
        print("\n❌ STG (stronger tilt) has FASTER propagation speed")
    
    return stg, wtg


def plot_scatter(data):
    """Plot tilt vs phase speed scatter for all events"""
    print("\nPlotting scatter...")
    
    enso_stats = data['enso_stats']
    valid = enso_stats.dropna(subset=['mean_tilt', 'phase_speed_ms'])
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    # Color by group
    colors = {'STG': '#E74C3C', 'WTG': '#3498DB', 'Normal': '#95A5A6'}
    
    for group in ['STG', 'Normal', 'WTG']:
        subset = valid[valid['group'] == group]
        ax.scatter(subset['phase_speed_ms'], subset['mean_tilt'],
                  c=colors[group], label=f"{group} (N={len(subset)})",
                  s=60, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    # Regression line
    x = valid['phase_speed_ms'].values
    y = valid['mean_tilt'].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', linewidth=2, label=f'Fit: r={r:.2f}')
    
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Phase Speed (m/s)", fontsize=11)
    ax.set_ylabel("Tilt Index (°)", fontsize=11)
    ax.set_title(f"MJO Tilt vs Phase Speed (N={len(valid)})\nr = {r:.3f}, p = {p:.4f}",
                fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(linestyle='--', alpha=0.3)
    
    out_path = FIG_DIR / "tilt_vs_phase_speed_scatter.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()
    
    # Print correlation
    print(f"\n[Correlation] Tilt vs Phase Speed:")
    print(f"  r = {r:.3f}, p = {p:.4f}")


def main():
    data = load_data()
    plot_composites(data)
    stg, wtg = analyze_phase_speed(data)
    plot_scatter(data)
    
    # Save group classification
    data['enso_stats'].to_csv(FIG_DIR / "event_stg_wtg_classification.csv", index=False)
    print(f"\nSaved: event_stg_wtg_classification.csv")
    
    print("\n" + "="*60)
    print("STG/WTG analysis completed!")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
