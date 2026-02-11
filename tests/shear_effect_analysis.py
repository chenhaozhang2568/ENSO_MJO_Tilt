# -*- coding: utf-8 -*-
"""
shear_effect_analysis.py — 背景垂直风切变效应分析

功能：
    分析背景纬向风垂直切变 |U200−U850| 对 MJO 结构的影响，
    检验不同 ENSO 相位下的风切变差异及其与 Tilt 的相关性。
输入：
    era5_mjo_recon_u_norm_1979-2022.nc, mjo_mvEOF_step3_1979-2022.nc,
    mjo_events_step3_1979-2022.csv, tilt_event_stats_with_enso_1979-2022.csv
输出：
    figures/shear_effect/shear_effect_analysis.png, event_shear_by_enso.csv
用法：
    python tests/shear_effect_analysis.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ========================
# PATHS
# ========================
U_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_u_norm_1979-2022.nc"
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\shear")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# SETTINGS
# ========================
ENSO_ORDER = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}


def load_data():
    """Load U data and calculate shear at MJO center"""
    print("Loading data...")
    
    # Load reconstructed U
    ds_u = xr.open_dataset(U_RECON_NC)
    u_recon = ds_u['u_mjo_recon_norm']
    if "pressure_level" in u_recon.dims:
        u_recon = u_recon.rename({"pressure_level": "level"})
    
    # Load MJO center track
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon = ds3['center_lon_track'].values
    center_lon_360 = np.mod(center_lon, 360)
    
    # Load events and ENSO
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])
    enso_stats = pd.read_csv(ENSO_STATS_CSV)
    enso_map = dict(zip(enso_stats['event_id'], enso_stats['enso_phase']))
    
    # Calculate shear: U200 - U850
    u200 = u_recon.sel(level=200).values
    u850 = u_recon.sel(level=850).values
    shear = u200 - u850
    
    # Build dataframe
    time = pd.to_datetime(u_recon.time.values)
    lon = u_recon.lon.values
    lon_360 = np.mod(lon, 360)
    
    df = pd.DataFrame({'time': time, 'center_lon': center_lon_360})
    df['event_id'] = np.nan
    df['enso_phase'] = ''
    df['shear_at_center'] = np.nan
    
    for _, ev in events.iterrows():
        mask = (df['time'] >= ev['start_date']) & (df['time'] <= ev['end_date'])
        df.loc[mask, 'event_id'] = ev['event_id']
        df.loc[mask, 'enso_phase'] = enso_map.get(ev['event_id'], 'Unknown')
    
    # Sample shear at center longitude
    for i in range(len(df)):
        clon = df.iloc[i]['center_lon']
        if np.isfinite(clon):
            lon_idx = np.argmin(np.abs(lon_360 - clon))
            df.iloc[i, df.columns.get_loc('shear_at_center')] = shear[i, lon_idx]
    
    df_events = df[df['event_id'].notna() & df['enso_phase'].isin(ENSO_ORDER)].copy()
    df_events['abs_shear'] = np.abs(df_events['shear_at_center'])
    
    # Event-level aggregation
    event_shear = df_events.groupby('event_id')['abs_shear'].mean().reset_index()
    event_shear.columns = ['event_id', 'mean_abs_shear']
    event_shear = event_shear.merge(enso_stats[['event_id', 'enso_phase', 'mean_tilt']], on='event_id')
    
    print(f"  Event days: {len(df_events)}, Events: {len(event_shear)}")
    return df_events, event_shear


def analyze_shear(df_daily, event_shear):
    """Analyze shear by ENSO phase"""
    print("\n" + "="*70)
    print("Hypothesis 3: Background Vertical Wind Shear |U200 - U850|")
    print("="*70)
    
    # Daily-level
    print("\n[Daily-level] |U200-U850| at MJO center:")
    print("-"*50)
    for phase in ENSO_ORDER:
        subset = df_daily[df_daily['enso_phase'] == phase]['abs_shear']
        print(f"  {phase:10s}: N={len(subset):4d}, Mean={subset.mean():.2f} m/s, "
              f"Std={subset.std():.2f}")
    
    # Event-level
    print("\n[Event-level] Mean |U200-U850| per event:")
    print("-"*50)
    for phase in ENSO_ORDER:
        subset = event_shear[event_shear['enso_phase'] == phase]['mean_abs_shear']
        print(f"  {phase:10s}: N_events={len(subset):3d}, "
              f"Mean |shear|={subset.mean():.2f} +/- {subset.std():.2f} m/s")
    
    # T-tests
    print("\n[T-tests] Shear differences:")
    print("-"*50)
    for i, p1 in enumerate(ENSO_ORDER):
        for p2 in ENSO_ORDER[i+1:]:
            g1 = event_shear[event_shear['enso_phase'] == p1]['mean_abs_shear']
            g2 = event_shear[event_shear['enso_phase'] == p2]['mean_abs_shear']
            t, p = stats.ttest_ind(g1, g2, equal_var=False)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {p1} vs {p2}: t={t:+.3f}, p={p:.4f} {sig}")
    
    # Correlation
    print("\n[Correlation] |Shear| vs Mean Tilt:")
    print("-"*50)
    valid = event_shear.dropna(subset=['mean_tilt'])
    r, p = stats.pearsonr(valid['mean_abs_shear'], valid['mean_tilt'])
    print(f"  Overall: r = {r:+.3f}, p = {p:.4f}")


def plot_shear_analysis(df_daily, event_shear):
    """Plot shear analysis figures"""
    print("\nPlotting shear analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    
    # =====================
    # Panel 1: Box plot (Daily)
    # =====================
    ax1 = axes[0, 0]
    sns.boxplot(x='enso_phase', y='abs_shear', data=df_daily, order=ENSO_ORDER,
                hue='enso_phase', palette=ENSO_COLORS, ax=ax1, width=0.5, legend=False)
    ax1.set_xlabel("ENSO Phase")
    ax1.set_ylabel("|U200 - U850| (m/s)")
    ax1.set_title("(a) Daily Vertical Shear at MJO Center", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # =====================
    # Panel 2: Bar chart (Event-level mean)
    # =====================
    ax2 = axes[0, 1]
    
    means = []
    sems = []
    ns = []
    for phase in ENSO_ORDER:
        subset = event_shear[event_shear['enso_phase'] == phase]['mean_abs_shear']
        means.append(subset.mean())
        sems.append(subset.sem())
        ns.append(len(subset))
    
    x = np.arange(len(ENSO_ORDER))
    bars = ax2.bar(x, means, 0.6, yerr=sems, capsize=5,
                   color=[ENSO_COLORS[p] for p in ENSO_ORDER], edgecolor='black', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p}\n(N={n})" for p, n in zip(ENSO_ORDER, ns)])
    ax2.set_ylabel("Mean |U200 - U850| +/- SEM (m/s)")
    ax2.set_title("(b) Event-level Mean Shear", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{m:.2f}', ha='center', va='bottom', fontsize=10)
    
    # =====================
    # Panel 3: KDE overlay
    # =====================
    ax3 = axes[1, 0]
    
    for phase in ENSO_ORDER:
        subset = event_shear[event_shear['enso_phase'] == phase]['mean_abs_shear']
        if len(subset) > 3:
            sns.kdeplot(subset, ax=ax3, label=f"{phase} (N={len(subset)})",
                       color=ENSO_COLORS[phase], linewidth=2)
            ax3.axvline(subset.mean(), color=ENSO_COLORS[phase], linestyle='--', alpha=0.7)
    
    ax3.set_xlabel("Mean |U200 - U850| per Event (m/s)")
    ax3.set_ylabel("Density")
    ax3.set_title("(c) Shear Distribution (KDE)", fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(linestyle='--', alpha=0.3)
    
    # =====================
    # Panel 4: Shear vs Tilt scatter
    # =====================
    ax4 = axes[1, 1]
    
    for phase in ENSO_ORDER:
        subset = event_shear[event_shear['enso_phase'] == phase].dropna(subset=['mean_tilt'])
        ax4.scatter(subset['mean_abs_shear'], subset['mean_tilt'],
                   c=ENSO_COLORS[phase], label=phase, s=60, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    # Regression line
    valid = event_shear.dropna(subset=['mean_tilt'])
    if len(valid) > 5:
        slope, intercept, r, p, se = stats.linregress(valid['mean_abs_shear'], valid['mean_tilt'])
        x_line = np.linspace(valid['mean_abs_shear'].min(), valid['mean_abs_shear'].max(), 100)
        y_line = slope * x_line + intercept
        ax4.plot(x_line, y_line, 'k--', linewidth=1.5, label=f'r={r:.2f}, p={p:.3f}')
    
    ax4.set_xlabel("Mean |U200 - U850| (m/s)")
    ax4.set_ylabel("Mean Tilt (deg)")
    ax4.set_title("(d) Shear vs Tilt Correlation", fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(linestyle='--', alpha=0.3)
    ax4.axhline(0, color='gray', linestyle='-', alpha=0.3)
    
    plt.suptitle("Hypothesis 3: Background Vertical Wind Shear Analysis", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    out_path = FIG_DIR / "shear_effect_analysis.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    df_daily, event_shear = load_data()
    analyze_shear(df_daily, event_shear)
    plot_shear_analysis(df_daily, event_shear)
    
    # Save event-level data
    out_csv = FIG_DIR / "event_shear_by_enso.csv"
    event_shear.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")
    
    print("\n" + "="*70)
    print("Shear effect analysis completed!")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
