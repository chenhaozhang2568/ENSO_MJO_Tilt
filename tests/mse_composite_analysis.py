# -*- coding: utf-8 -*-
"""
mse_composite_analysis.py — MSE 充放电机制分析

功能：
    分析湿静力能（MSE）在 MJO 传播中的作用，检验 ENSO 不同相位下
    MSE 领先距离的差异，包括经度-高度剖面合成图、差异图和领先距离统计。
输入：
    era5_mjo_recon_{t,q}_norm_1979-2022.nc, mjo_mvEOF_step3_1979-2022.nc,
    mjo_events_step3_1979-2022.csv, tilt_event_stats_with_enso_1979-2022.csv
输出：
    figures/mse_effect/ 下的 MSE 合成、差异、摘要图
用法：
    python tests/mse_composite_analysis.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from scipy import stats

# ========================
# PATHS
# ========================
T_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_t_norm_1979-2022.nc"
Q_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\mse")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# PHYSICAL CONSTANTS
# ========================
CP = 1004.0       # J/(kg·K), specific heat at constant pressure
G = 9.81          # m/s², gravitational acceleration
LV = 2.5e6        # J/kg, latent heat of vaporization

# Approximate geopotential height for each pressure level (m)
# Based on standard atmosphere
Z_APPROX = {
    1000: 100,
    925: 750,
    850: 1500,
    700: 3000,
    600: 4200,
    500: 5500,
    400: 7200,
    300: 9200,
    200: 12000,
}

ENSO_ORDER = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}
AMP_THRESHOLD = 0.5


def load_data():
    """Load reconstructed T, q, and calculate MSE"""
    print("Loading data...")
    
    # Load T and q
    ds_t = xr.open_dataset(T_RECON_NC)
    ds_q = xr.open_dataset(Q_RECON_NC)
    
    t_recon = ds_t['t_mjo_recon_norm']  # (time, level, lon) - already normalized by MJO amp
    q_recon = ds_q['q_mjo_recon_norm']  # (time, level, lon) - already normalized by MJO amp
    
    # Get dimensions
    time = pd.to_datetime(t_recon.time.values)
    levels = t_recon.pressure_level.values if "pressure_level" in t_recon.dims else t_recon.level.values
    lon = t_recon.lon.values
    
    # Calculate MSE anomaly at each level
    # MSE = Cp*T + g*z + Lv*q
    # Since we use reconstructed fields (MJO signal), this is MSE anomaly
    
    mse = np.zeros_like(t_recon.values)
    for i, lev in enumerate(levels):
        z = Z_APPROX.get(int(lev), 5000)
        mse[:, i, :] = CP * t_recon.values[:, i, :] + LV * q_recon.values[:, i, :]
        # Note: g*z is constant per level, doesn't contribute to anomaly
    
    mse_da = xr.DataArray(
        mse,
        coords={"time": t_recon.time, "level": levels, "lon": t_recon.lon},
        dims=("time", "level", "lon"),
        name="mse_mjo_recon"
    )
    
    # Load MJO center track and amplitude
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon = ds3['center_lon_track'].values
    mjo_amp = ds3['amp'].values  # MJO amplitude for normalization
    
    # Load events and ENSO
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])
    enso_stats = pd.read_csv(ENSO_STATS_CSV)
    enso_map = dict(zip(enso_stats['event_id'], enso_stats['enso_phase']))
    
    # Build event days dataframe
    df = pd.DataFrame({'time': time, 'center_lon': center_lon, 'mjo_amp': mjo_amp})
    df['event_id'] = np.nan
    df['enso_phase'] = ''
    
    for _, ev in events.iterrows():
        mask = (df['time'] >= ev['start_date']) & (df['time'] <= ev['end_date'])
        df.loc[mask, 'event_id'] = ev['event_id']
        df.loc[mask, 'enso_phase'] = enso_map.get(ev['event_id'], 'Unknown')
    
    df_events = df[df['event_id'].notna() & df['enso_phase'].isin(ENSO_ORDER)].copy()
    
    print(f"  MSE shape: {mse_da.shape}")
    print(f"  Event days: {len(df_events)}")
    
    return mse_da, df_events, lon, levels


def create_composite(mse_da, df_events, lon, levels, lon_range=(-90, 180)):
    """
    Create MSE composite in relative longitude space.
    Center each day's MSE field on MJO convective center.
    数据来源为 _norm_（已做振幅归一化），无需再归一化。
    """
    print("\nCreating composites...")
    
    lon_360 = np.mod(lon, 360)
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int((lon_range[1] - lon_range[0]) / dlon) + 1
    rel_lons = np.linspace(lon_range[0], lon_range[1], n_rel_bins)
    
    composites = {}
    
    for phase in ENSO_ORDER:
        phase_df = df_events[df_events['enso_phase'] == phase]
        n_days = len(phase_df)
        
        comp = np.zeros((len(levels), len(rel_lons)))
        count = np.zeros((len(levels), len(rel_lons)))
        
        for idx, row in phase_df.iterrows():
            clon = row['center_lon']
            if not np.isfinite(clon):
                continue
            
            # Get this day's MSE profile
            t_idx = np.where(df_events.index == idx)[0]
            if len(t_idx) == 0:
                continue
            
            day_idx = list(mse_da.time.values).index(pd.Timestamp(row['time']))
            mse_day = mse_da.isel(time=day_idx).values  # (level, lon)
            
            # 振幅筛选
            amp_today = row['mjo_amp']
            if not np.isfinite(amp_today) or amp_today < AMP_THRESHOLD:
                continue
            
            # Calculate relative longitude
            clon_360 = np.mod(clon, 360)
            for j, rlon in enumerate(rel_lons):
                target_lon = np.mod(clon_360 + rlon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - target_lon))
                comp[:, j] += mse_day[:, lon_idx]
                count[:, j] += 1
        
        # Average
        with np.errstate(invalid='ignore'):
            comp = comp / np.where(count > 0, count, np.nan)
        
        composites[phase] = comp
        print(f"  {phase}: {n_days} days")
    
    return composites, rel_lons, levels


def find_mse_peak(composite, rel_lons, levels, level_range=(850, 1000)):
    """Find the longitude of MSE peak in lower levels"""
    # Select lower levels
    level_mask = (levels >= min(level_range)) & (levels <= max(level_range))
    if not level_mask.any():
        level_mask = levels >= 700  # fallback
    
    lower_mse = composite[level_mask, :].mean(axis=0)
    peak_idx = np.nanargmax(lower_mse)
    peak_lon = rel_lons[peak_idx]
    
    return peak_lon


def plot_mse_composites(composites, rel_lons, levels):
    """Plot MSE Height-Longitude composites for each ENSO phase"""
    print("\nPlotting MSE composites...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150, sharey=True)
    
    # Find common colorbar range
    all_vals = []
    for comp in composites.values():
        all_vals.extend(comp[np.isfinite(comp)].flatten())
    vmax = np.percentile(np.abs(all_vals), 95)
    vmin = -vmax
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
    for i, phase in enumerate(ENSO_ORDER):
        ax = axes[i]
        comp = composites[phase]
        
        # Contour plot
        cf = ax.contourf(rel_lons, levels, comp, levels=20, cmap='RdBu_r', norm=norm)
        cs = ax.contour(rel_lons, levels, comp, levels=10, colors='k', linewidths=0.5, alpha=0.5)
        
        # Mark convective center
        ax.axvline(0, color='black', linestyle='--', linewidth=2, label='Convective Center')
        
        # Find and mark MSE peak in lower troposphere
        peak_lon = find_mse_peak(comp, rel_lons, levels)
        ax.axvline(peak_lon, color='green', linestyle=':', linewidth=2, label=f'Low-level MSE peak ({peak_lon:+.0f}°)')
        
        ax.set_xlabel("Relative Longitude (°)")
        if i == 0:
            ax.set_ylabel("Pressure Level (hPa)")
        ax.set_title(f"({chr(97+i)}) {phase}", fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(-90, 180)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(linestyle='--', alpha=0.3)
    
    # Colorbar
    cbar = fig.colorbar(cf, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('MSE Anomaly (J/kg)', fontsize=11)
    
    plt.suptitle("MSE Height-Longitude Composite (Centered on MJO Convection)", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    out_path = FIG_DIR / "mse_height_longitude_composite.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def plot_mse_diff(composites, rel_lons, levels):
    """Plot MSE difference between ENSO phases (pairwise comparison)"""
    print("\nPlotting MSE difference composites...")
    
    diff_pairs = [
        ('El Nino', 'La Nina', '(a) El Nino - La Nina'),
        ('El Nino', 'Neutral', '(b) El Nino - Neutral'),
        ('La Nina', 'Neutral', '(c) La Nina - Neutral'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150, sharey=True)
    
    # Calculate differences and find common colorbar range
    diffs = []
    for p1, p2, _ in diff_pairs:
        diff = composites[p1] - composites[p2]
        diffs.append(diff)
    
    all_diff = np.concatenate([d[np.isfinite(d)].flatten() for d in diffs])
    vmax = np.percentile(np.abs(all_diff), 95)
    vmin = -vmax
    
    for i, ((p1, p2, title), diff) in enumerate(zip(diff_pairs, diffs)):
        ax = axes[i]
        
        # Contour plot with symmetric colorbar
        cf = ax.contourf(rel_lons, levels, diff, levels=np.linspace(-vmax, vmax, 21), 
                         cmap='RdBu_r', extend='both')
        cs = ax.contour(rel_lons, levels, diff, levels=[-50, 0, 50], colors='k', 
                        linewidths=0.8, alpha=0.7)
        
        # Mark convective center
        ax.axvline(0, color='limegreen', linewidth=3, alpha=0.9)
        
        ax.set_xlabel("Relative Longitude (°)")
        if i == 0:
            ax.set_ylabel("Pressure Level (hPa)")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(-90, 180)
        ax.grid(linestyle='--', alpha=0.3)
    
    # Colorbar
    cbar = fig.colorbar(cf, ax=axes, orientation='vertical', shrink=0.8, pad=0.02)
    cbar.set_label('MSE Difference (J/kg)', fontsize=11)
    
    plt.suptitle("MSE Difference Between ENSO Phases", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    out_path = FIG_DIR / "mse_height_longitude_diff.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def analyze_mse_leading(composites, rel_lons, levels):
    """Analyze the MSE leading distance for each ENSO phase"""
    print("\n" + "="*70)
    print("Hypothesis 4: MSE Recharge-Discharge Analysis")
    print("="*70)
    
    print("\n[MSE Peak Position] (relative to convective center, low-level 850-1000hPa):")
    print("-"*50)
    
    results = {}
    for phase in ENSO_ORDER:
        peak_lon = find_mse_peak(composites[phase], rel_lons, levels)
        results[phase] = peak_lon
        direction = "EAST (ahead)" if peak_lon > 0 else "WEST (behind)" if peak_lon < 0 else "at center"
        print(f"  {phase:10s}: {peak_lon:+.1f}° ({direction})")
    
    print("\n[Interpretation]:")
    print("-"*50)
    print("  Positive value = MSE leads (east of center) -> recharge ahead")
    print("  Negative value = MSE lags (west of center) -> delayed recharge")
    
    return results


def plot_summary(composites, rel_lons, levels):
    """Plot summary comparison"""
    print("\nPlotting summary...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # =====================
    # Panel 1: Lower-level MSE profile
    # =====================
    ax1 = axes[0]
    
    level_mask = (levels >= 700) & (levels <= 1000)
    
    for phase in ENSO_ORDER:
        comp = composites[phase]
        lower_mse = comp[level_mask, :].mean(axis=0)
        ax1.plot(rel_lons, lower_mse, color=ENSO_COLORS[phase], linewidth=2, label=phase)
    
    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel("Relative Longitude (°)")
    ax1.set_ylabel("Lower-level MSE Anomaly (J/kg)")
    ax1.set_title("(a) Lower Troposphere MSE (700-1000 hPa avg)", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_xlim(-90, 180)
    
    # =====================
    # Panel 2: MSE peak position bar chart
    # =====================
    ax2 = axes[1]
    
    peaks = [find_mse_peak(composites[p], rel_lons, levels) for p in ENSO_ORDER]
    x = np.arange(len(ENSO_ORDER))
    bars = ax2.bar(x, peaks, 0.6, color=[ENSO_COLORS[p] for p in ENSO_ORDER], edgecolor='black')
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ENSO_ORDER)
    ax2.set_ylabel("MSE Peak Position (° from center)")
    ax2.set_title("(b) Low-level MSE Leading Distance", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar, p in zip(bars, peaks):
        va = 'bottom' if p >= 0 else 'top'
        offset = 1 if p >= 0 else -1
        ax2.text(bar.get_x() + bar.get_width()/2, p + offset, f'{p:+.1f}°', 
                ha='center', va=va, fontsize=11, fontweight='bold')
    
    plt.suptitle("MSE Recharge Analysis by ENSO Phase", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = FIG_DIR / "mse_leading_summary.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    mse_da, df_events, lon, levels = load_data()
    composites, rel_lons, levels = create_composite(mse_da, df_events, lon, levels)
    
    results = analyze_mse_leading(composites, rel_lons, levels)
    
    plot_mse_composites(composites, rel_lons, levels)
    plot_mse_diff(composites, rel_lons, levels)
    plot_summary(composites, rel_lons, levels)
    
    print("\n" + "="*70)
    print("MSE composite analysis completed!")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
