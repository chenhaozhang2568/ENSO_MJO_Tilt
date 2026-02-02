# -*- coding: utf-8 -*-
"""
intensity_effect_analysis.py: 假设2 - 强度效应验证

================================================================================
功能描述：
    本脚本验证 MJO 强度效应假设，分析 ENSO 不同相位期间 MJO 振幅是否存在显著差异。
    
科学问题：
    - El Niño 和 La Niña 期间 MJO 的振幅（强度）是否有差异？
    - MJO 强度与其结构倾斜之间是否存在关联？
    
物理机制：
    强 MJO 具有更有组织的环流结构，可能表现出更标准化的垂直倾斜特征。
    
主要分析内容：
    1. 逐日 MJO 振幅在三组 ENSO 相位中的分布对比
    2. 事件平均振幅的 ENSO 分组统计
    3. 箱线图、直方图可视化
    4. t-检验统计显著性分析

Inputs:
- Step3 NC: amp (MJO amplitude = sqrt(PC1^2 + PC2^2))
- Events CSV + ENSO classification

Run:
  python E:\\Projects\\ENSO_MJO_Tilt\\tests\\intensity_effect_analysis.py
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
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\intensity_effect")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# SETTINGS
# ========================
ENSO_ORDER = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}


def load_data():
    """Load amplitude and tilt data"""
    print("Loading data...")
    
    ds3 = xr.open_dataset(STEP3_NC)
    amp = ds3['amp'].to_series()
    amp.name = 'amp'
    
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])
    enso_stats = pd.read_csv(ENSO_STATS_CSV, parse_dates=['start_date', 'end_date'])
    enso_map = dict(zip(enso_stats['event_id'], enso_stats['enso_phase']))
    
    # Build daily dataframe
    df = pd.DataFrame({'amp': amp})
    df['event_id'] = np.nan
    df['enso_phase'] = ''
    
    for _, ev in events.iterrows():
        mask = (df.index >= ev['start_date']) & (df.index <= ev['end_date'])
        df.loc[mask, 'event_id'] = ev['event_id']
        df.loc[mask, 'enso_phase'] = enso_map.get(ev['event_id'], 'Unknown')
    
    df_events = df[df['event_id'].notna() & df['enso_phase'].isin(ENSO_ORDER)]
    
    # Event-level aggregation
    event_amps = df_events.groupby('event_id')['amp'].mean().reset_index()
    event_amps.columns = ['event_id', 'mean_amp']
    event_amps = event_amps.merge(enso_stats[['event_id', 'enso_phase', 'mean_tilt']], on='event_id')
    
    print(f"  Daily samples: {len(df_events)}")
    print(f"  Events: {len(event_amps)}")
    
    return df_events, event_amps, enso_stats


def analyze_amplitude(df_daily, event_amps):
    """Analyze amplitude by ENSO phase"""
    print("\n" + "="*70)
    print("Hypothesis 2: Intensity Effect - MJO Amplitude by ENSO Phase")
    print("="*70)
    
    # Daily-level
    print("\n[Daily-level] MJO Amplitude (sqrt(PC1^2 + PC2^2)):")
    print("-"*50)
    for phase in ENSO_ORDER:
        subset = df_daily[df_daily['enso_phase'] == phase]['amp']
        print(f"  {phase:10s}: N={len(subset):4d}, Mean={subset.mean():.3f} +/- {subset.std():.3f}")
    
    # Event-level
    print("\n[Event-level] Mean Amplitude per Event:")
    print("-"*50)
    for phase in ENSO_ORDER:
        subset = event_amps[event_amps['enso_phase'] == phase]['mean_amp']
        print(f"  {phase:10s}: N_events={len(subset):3d}, Mean Amp={subset.mean():.3f} +/- {subset.std():.3f}")
    
    # T-tests
    print("\n[T-tests] Amplitude Differences:")
    print("-"*50)
    results = []
    for i, p1 in enumerate(ENSO_ORDER):
        for p2 in ENSO_ORDER[i+1:]:
            g1 = event_amps[event_amps['enso_phase'] == p1]['mean_amp']
            g2 = event_amps[event_amps['enso_phase'] == p2]['mean_amp']
            t, p = stats.ttest_ind(g1, g2, equal_var=False)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {p1} vs {p2}: t={t:+.3f}, p={p:.4f} {sig}")
            results.append({'comparison': f'{p1} vs {p2}', 't': t, 'p': p, 'sig': sig})
    
    # Correlation with Tilt
    print("\n[Correlation] Amplitude vs Tilt:")
    print("-"*50)
    valid = event_amps.dropna(subset=['mean_tilt'])
    r, p = stats.pearsonr(valid['mean_amp'], valid['mean_tilt'])
    print(f"  Overall: r = {r:+.3f}, p = {p:.4f}")
    
    for phase in ENSO_ORDER:
        subset = valid[valid['enso_phase'] == phase]
        if len(subset) > 5:
            r, p = stats.pearsonr(subset['mean_amp'], subset['mean_tilt'])
            print(f"  {phase:10s}: r = {r:+.3f}, p = {p:.4f}")
    
    return results


def plot_amplitude_analysis(df_daily, event_amps):
    """Plot amplitude analysis figures"""
    print("\nPlotting amplitude analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    
    # =====================
    # Panel 1: Boxplot (Daily)
    # =====================
    ax1 = axes[0, 0]
    sns.boxplot(x='enso_phase', y='amp', data=df_daily, order=ENSO_ORDER,
                hue='enso_phase', palette=ENSO_COLORS, ax=ax1, width=0.5, legend=False)
    ax1.set_xlabel("ENSO Phase")
    ax1.set_ylabel("MJO Amplitude")
    ax1.set_title("(a) Daily MJO Amplitude Distribution", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add sample sizes
    for i, phase in enumerate(ENSO_ORDER):
        n = len(df_daily[df_daily['enso_phase'] == phase])
        ax1.text(i, ax1.get_ylim()[0], f'N={n}', ha='center', va='top', fontsize=9)
    
    # =====================
    # Panel 2: Bar chart (Event-level mean)
    # =====================
    ax2 = axes[0, 1]
    
    means = []
    sems = []
    ns = []
    for phase in ENSO_ORDER:
        subset = event_amps[event_amps['enso_phase'] == phase]['mean_amp']
        means.append(subset.mean())
        sems.append(subset.sem())
        ns.append(len(subset))
    
    x = np.arange(len(ENSO_ORDER))
    bars = ax2.bar(x, means, 0.6, yerr=sems, capsize=5,
                   color=[ENSO_COLORS[p] for p in ENSO_ORDER], edgecolor='black', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{p}\n(N={n})" for p, n in zip(ENSO_ORDER, ns)])
    ax2.set_ylabel("Mean Amplitude +/- SEM")
    ax2.set_title("(b) Event-level Mean Amplitude", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{m:.3f}', ha='center', va='bottom', fontsize=10)
    
    # =====================
    # Panel 3: KDE overlay
    # =====================
    ax3 = axes[1, 0]
    
    for phase in ENSO_ORDER:
        subset = event_amps[event_amps['enso_phase'] == phase]['mean_amp']
        if len(subset) > 3:
            sns.kdeplot(subset, ax=ax3, label=f"{phase} (N={len(subset)})",
                       color=ENSO_COLORS[phase], linewidth=2)
            ax3.axvline(subset.mean(), color=ENSO_COLORS[phase], linestyle='--', alpha=0.7)
    
    ax3.set_xlabel("Mean Amplitude per Event")
    ax3.set_ylabel("Density")
    ax3.set_title("(c) Amplitude Distribution (KDE)", fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(linestyle='--', alpha=0.3)
    
    # =====================
    # Panel 4: Amplitude vs Tilt scatter
    # =====================
    ax4 = axes[1, 1]
    
    for phase in ENSO_ORDER:
        subset = event_amps[event_amps['enso_phase'] == phase].dropna(subset=['mean_tilt'])
        ax4.scatter(subset['mean_amp'], subset['mean_tilt'],
                   c=ENSO_COLORS[phase], label=phase, s=60, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    # Overall regression line
    valid = event_amps.dropna(subset=['mean_tilt'])
    if len(valid) > 5:
        slope, intercept, r, p, se = stats.linregress(valid['mean_amp'], valid['mean_tilt'])
        x_line = np.linspace(valid['mean_amp'].min(), valid['mean_amp'].max(), 100)
        y_line = slope * x_line + intercept
        ax4.plot(x_line, y_line, 'k--', linewidth=1.5, label=f'r={r:.2f}, p={p:.3f}')
    
    ax4.set_xlabel("Mean Amplitude")
    ax4.set_ylabel("Mean Tilt (deg)")
    ax4.set_title("(d) Amplitude vs Tilt Correlation", fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(linestyle='--', alpha=0.3)
    ax4.axhline(0, color='gray', linestyle='-', alpha=0.3)
    
    plt.suptitle("Hypothesis 2: MJO Intensity Effect Analysis", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    out_path = FIG_DIR / "intensity_effect_analysis.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    df_daily, event_amps, enso_stats = load_data()
    analyze_amplitude(df_daily, event_amps)
    plot_amplitude_analysis(df_daily, event_amps)
    
    # Save event-level data
    out_csv = FIG_DIR / "event_amplitude_by_enso.csv"
    event_amps.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")
    
    print("\n" + "="*70)
    print("Intensity effect analysis completed!")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
