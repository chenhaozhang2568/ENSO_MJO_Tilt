# -*- coding: utf-8 -*-
"""
mse_composite_analysis.py: 假设4 - MSE 充放电机制分析

================================================================================
功能描述：
    本脚本分析湿静力能（MSE）在 MJO 传播中的作用，检验 ENSO 不同相位下 MSE 领先距离的差异。
    
科学问题：
    - ENSO 如何影响 MJO 前方的 MSE 充能过程？
    - MSE 异常峰值相对于对流中心的领先距离是否因 ENSO 相位而异？
    
物理机制：
    MJO 传播依赖"充放电"机制：低层 MSE 在对流前方累积（充能），为东传提供热力学条件。
    MSE 定义：MSE = Cp×T + g×z + Lv×q（感热 + 位势能 + 潜热）
    
主要分析内容：
    1. 相对经度坐标下的 MSE 垂直-纬向剖面合成图
    2. 三组 ENSO 相位的 MSE 前沿位置对比
    3. 低层 MSE 峰值领先距离统计
    4. MSE 结构与 MJO 倾斜的关联分析

Calculate composites of MSE anomaly in Height-Longitude space, 
centered on MJO convective center (Lag=0).

Inputs:
- Reconstructed T: E:\Datas\Derived\era5_mjo_recon_t_1979-2022.nc
- Reconstructed q: E:\Datas\Derived\era5_mjo_recon_q_1979-2022.nc
- MJO events and center longitude

Run:
  python E:\\Projects\\ENSO_MJO_Tilt\\tests\\mse_composite_analysis.py
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

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\mse_effect")
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
    levels = t_recon.pressure_level.values
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
        coords={"time": t_recon.time, "pressure_level": t_recon.pressure_level, "lon": t_recon.lon},
        dims=("time", "pressure_level", "lon"),
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


def create_composite(mse_da, df_events, lon, levels, lon_half_width=60, normalize_by_amp=True):
    """
    Create MSE composite in relative longitude space.
    Center each day's MSE field on MJO convective center.
    
    Args:
        normalize_by_amp: If True, divide MSE by MJO amplitude (Hu & Li 2021 method)
    """
    print("\nCreating composites...")
    
    lon_360 = np.mod(lon, 360)
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int(2 * lon_half_width / dlon) + 1
    rel_lons = np.linspace(-lon_half_width, lon_half_width, n_rel_bins)
    
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
            
            # Normalize by MJO amplitude (Hu & Li 2021 method)
            amp_today = row['mjo_amp']
            if normalize_by_amp and np.isfinite(amp_today) and amp_today > 0.5:
                mse_day = mse_day / amp_today
            elif normalize_by_amp and (not np.isfinite(amp_today) or amp_today <= 0.5):
                continue  # Skip days with too weak amplitude
            
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
        ax.set_xlim(-60, 60)
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
    ax1.set_xlim(-60, 60)
    
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
    plot_summary(composites, rel_lons, levels)
    
    print("\n" + "="*70)
    print("MSE composite analysis completed!")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
