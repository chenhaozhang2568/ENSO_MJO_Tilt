# -*- coding: utf-8 -*-
"""
location_residence_analysis.py: 海洋大陆停留时间 + 倾斜-经度关系分析

================================================================================
功能描述：
    本脚本分析 MJO 在海洋大陆（Maritime Continent, MC）区域的停留时间，
    以及逐日 MJO 倾斜指数与对流中心经度的定量关系。

科学问题：
    Q1: 不同 ENSO 相位下，MJO 在 MC 区域的停留比例是否有差异？
    Q2: MJO 倾斜指数与其所在经度位置之间是什么关系？

关键经度定义：
    - MC 西界：100°E
    - MC 东界：160°E

主要分析内容：
    1. MC 停留时间比例的 ENSO 分组统计
    2. 倾斜-经度散点图与分箱回归
    3. MC 内外倾斜差异的统计检验
    4. 综合可视化图表

Inputs:
- Step3 NC: center_lon_track(time)
- Tilt Daily NC: tilt(time)
- Events CSV + ENSO classification

Run:
  python E:\\Projects\\ENSO_MJO_Tilt\\tests\\location_residence_analysis.py
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
TILT_NC = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\location_effect")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# SETTINGS
# ========================
MC_WEST = 100
MC_EAST = 160

ENSO_ORDER = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}


def to_lon_360(lon):
    return np.mod(lon, 360)


def load_daily_data():
    """Load daily data and assign to events and ENSO groups"""
    print("Loading data...")
    
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon = ds3["center_lon_track"].to_series()
    center_lon.name = "center_lon"
    
    ds_tilt = xr.open_dataset(TILT_NC)
    tilt = ds_tilt["tilt"].to_series()
    tilt.name = "tilt"
    
    df_daily = pd.concat([center_lon, tilt], axis=1).dropna()
    df_daily["center_lon"] = to_lon_360(df_daily["center_lon"])
    
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    enso_stats = pd.read_csv(ENSO_STATS_CSV, parse_dates=["start_date", "end_date"])
    enso_map = dict(zip(enso_stats["event_id"], enso_stats["enso_phase"]))
    
    df_daily["event_id"] = np.nan
    df_daily["enso_phase"] = ""
    
    for _, ev in events.iterrows():
        eid = ev["event_id"]
        s, e = ev["start_date"], ev["end_date"]
        mask = (df_daily.index >= s) & (df_daily.index <= e)
        df_daily.loc[mask, "event_id"] = eid
        df_daily.loc[mask, "enso_phase"] = enso_map.get(eid, "Unknown")
    
    df_event_days = df_daily[df_daily["event_id"].notna()].copy()
    df_event_days = df_event_days[df_event_days["enso_phase"].isin(ENSO_ORDER)]
    
    # Mark if in MC region
    df_event_days["in_MC"] = (df_event_days["center_lon"] >= MC_WEST) & \
                              (df_event_days["center_lon"] < MC_EAST)
    
    print(f"  Total event days: {len(df_event_days)}")
    return df_event_days, events, enso_stats


def analyze_residence_fraction(df: pd.DataFrame, enso_stats: pd.DataFrame):
    """
    Analyze MC residence fraction across ENSO phases
    """
    print("\n" + "="*70)
    print("Q1: MC Residence Fraction by ENSO Phase")
    print("="*70)
    
    results = []
    
    # Overall statistics by ENSO phase
    print("\n[Overall] MC Residence Fraction by ENSO Phase:")
    print("-"*50)
    
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        total_days = len(subset)
        mc_days = subset["in_MC"].sum()
        outside_days = total_days - mc_days
        mc_frac = mc_days / total_days * 100 if total_days > 0 else 0
        
        print(f"  {phase:10s}: Total={total_days:4d}, MC={mc_days:4d} ({mc_frac:5.1f}%), "
              f"Outside={outside_days:4d} ({100-mc_frac:5.1f}%)")
        
        results.append({
            "enso_phase": phase,
            "total_days": total_days,
            "mc_days": mc_days,
            "outside_days": outside_days,
            "mc_fraction": mc_frac
        })
    
    # Event-level statistics
    print("\n[Event-level] MC Residence Fraction per Event:")
    print("-"*50)
    
    event_results = []
    for phase in ENSO_ORDER:
        phase_events = enso_stats[enso_stats["enso_phase"] == phase]["event_id"].values
        
        for eid in phase_events:
            ev_data = df[df["event_id"] == eid]
            if len(ev_data) == 0:
                continue
            
            total = len(ev_data)
            mc_count = ev_data["in_MC"].sum()
            mc_frac = mc_count / total * 100 if total > 0 else 0
            
            event_results.append({
                "event_id": eid,
                "enso_phase": phase,
                "total_days": total,
                "mc_days": mc_count,
                "mc_fraction": mc_frac
            })
    
    df_event_res = pd.DataFrame(event_results)
    
    # Summary by ENSO group
    print("\n[Event Summary] Mean MC Fraction by ENSO Phase:")
    print("-"*50)
    
    for phase in ENSO_ORDER:
        phase_df = df_event_res[df_event_res["enso_phase"] == phase]
        if len(phase_df) > 0:
            mean_frac = phase_df["mc_fraction"].mean()
            std_frac = phase_df["mc_fraction"].std()
            n_events = len(phase_df)
            print(f"  {phase:10s}: N_events={n_events:3d}, Mean MC%={mean_frac:5.1f}% +/- {std_frac:5.1f}%")
    
    # Statistical tests
    print("\n[T-test] MC Fraction Differences Between ENSO Groups:")
    print("-"*50)
    
    for i, phase1 in enumerate(ENSO_ORDER):
        for phase2 in ENSO_ORDER[i+1:]:
            group1 = df_event_res[df_event_res["enso_phase"] == phase1]["mc_fraction"]
            group2 = df_event_res[df_event_res["enso_phase"] == phase2]["mc_fraction"]
            
            if len(group1) > 1 and len(group2) > 1:
                t, p = stats.ttest_ind(group1, group2, equal_var=False)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                print(f"  {phase1} vs {phase2}: t={t:+6.2f}, p={p:.4f} {sig}")
    
    # Save results
    out_csv = FIG_DIR / "mc_residence_fraction_by_enso.csv"
    df_event_res.to_csv(out_csv, index=False)
    print(f"\n  Saved event-level data: {out_csv}")
    
    return df_event_res


def plot_residence_fraction(df_event_res: pd.DataFrame):
    """Plot MC residence fraction comparison"""
    print("\nPlotting MC residence fraction comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    
    # =====================
    # Panel 1: Stacked bar (total days)
    # =====================
    ax1 = axes[0]
    
    mc_counts = []
    outside_counts = []
    for phase in ENSO_ORDER:
        phase_df = df_event_res[df_event_res["enso_phase"] == phase]
        mc_counts.append(phase_df["mc_days"].sum())
        outside_counts.append(phase_df["total_days"].sum() - phase_df["mc_days"].sum())
    
    x = np.arange(len(ENSO_ORDER))
    width = 0.6
    
    bars1 = ax1.bar(x, mc_counts, width, label='Inside MC (100-160E)', color='green', alpha=0.7)
    bars2 = ax1.bar(x, outside_counts, width, bottom=mc_counts, label='Outside MC', color='orange', alpha=0.7)
    
    # Add percentage labels
    for i, (mc, outside) in enumerate(zip(mc_counts, outside_counts)):
        total = mc + outside
        ax1.text(i, mc/2, f'{mc/total*100:.0f}%', ha='center', va='center', fontsize=10, fontweight='bold')
        ax1.text(i, mc + outside/2, f'{outside/total*100:.0f}%', ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(ENSO_ORDER)
    ax1.set_ylabel("Number of Days")
    ax1.set_title("(a) Total Days Distribution", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # =====================
    # Panel 2: Boxplot (event-level MC fraction)
    # =====================
    ax2 = axes[1]
    
    sns.boxplot(x="enso_phase", y="mc_fraction", data=df_event_res, 
                order=ENSO_ORDER, hue="enso_phase", palette=ENSO_COLORS, 
                ax=ax2, width=0.5, legend=False)
    sns.swarmplot(x="enso_phase", y="mc_fraction", data=df_event_res,
                  order=ENSO_ORDER, color=".2", alpha=0.5, size=4, ax=ax2)
    
    ax2.set_xlabel("ENSO Phase")
    ax2.set_ylabel("MC Residence Fraction (%)")
    ax2.set_title("(b) Event-level MC Fraction", fontsize=12, fontweight='bold')
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50%')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # =====================
    # Panel 3: Bar chart (mean + SEM)
    # =====================
    ax3 = axes[2]
    
    means = []
    sems = []
    ns = []
    for phase in ENSO_ORDER:
        phase_df = df_event_res[df_event_res["enso_phase"] == phase]["mc_fraction"]
        means.append(phase_df.mean())
        sems.append(phase_df.sem())
        ns.append(len(phase_df))
    
    bars = ax3.bar(x, means, width, yerr=sems, capsize=5,
                   color=[ENSO_COLORS[p] for p in ENSO_ORDER], edgecolor='black', alpha=0.8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{p}\n(N={n})" for p, n in zip(ENSO_ORDER, ns)])
    ax3.set_ylabel("Mean MC Residence Fraction (%)")
    ax3.set_title("(c) Mean MC Fraction +/- SEM", fontsize=12, fontweight='bold')
    ax3.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar, m, s in zip(bars, means, sems):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 1,
                f'{m:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle("MJO Residence Time in Maritime Continent (MC) by ENSO Phase", 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = FIG_DIR / "mc_residence_fraction_comparison.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def analyze_tilt_vs_longitude(df: pd.DataFrame):
    """
    Analyze the relationship between daily Tilt and longitude
    """
    print("\n" + "="*70)
    print("Q2: Daily Tilt vs Longitude Relationship")
    print("="*70)
    
    # Overall correlation
    print("\n[Overall Correlation] (All ENSO groups):")
    print("-"*50)
    
    corr, p_corr = stats.pearsonr(df["center_lon"], df["tilt"])
    print(f"  Pearson r = {corr:+.4f}, p = {p_corr:.2e}")
    
    slope, intercept, r, p, se = stats.linregress(df["center_lon"], df["tilt"])
    print(f"  Linear regression: Tilt = {slope:+.4f} x Lon + {intercept:.2f}")
    print(f"            R^2 = {r**2:.4f}, p = {p:.2e}")
    
    # By ENSO group
    print("\n[Correlation by ENSO Phase]:")
    print("-"*50)
    
    regression_results = []
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        if len(subset) > 10:
            c, pc = stats.pearsonr(subset["center_lon"], subset["tilt"])
            s, i, r, p, se = stats.linregress(subset["center_lon"], subset["tilt"])
            print(f"  {phase:10s}: r={c:+.3f} (p={pc:.2e}), "
                  f"Tilt = {s:+.4f} x Lon + {i:.2f}")
            regression_results.append({
                "phase": phase,
                "r": c,
                "p": pc,
                "slope": s,
                "intercept": i,
                "r_squared": r**2
            })
    
    # By region
    print("\n[Regional Analysis]:")
    print("-"*50)
    
    regions = [
        ("Indian Ocean", 40, 100),
        ("Maritime Continent", 100, 160),
        ("Western Pacific", 160, 220),
    ]
    
    for region_name, lon_min, lon_max in regions:
        region_data = df[(df["center_lon"] >= lon_min) & (df["center_lon"] < lon_max)]
        if len(region_data) > 10:
            mean_tilt = region_data["tilt"].mean()
            std_tilt = region_data["tilt"].std()
            print(f"  {region_name} ({lon_min}-{lon_max}E): "
                  f"N={len(region_data):4d}, Mean Tilt={mean_tilt:+.2f} +/- {std_tilt:.2f} deg")
    
    return regression_results


def plot_tilt_vs_longitude_detailed(df: pd.DataFrame):
    """Plot detailed Tilt vs Longitude relationship"""
    print("\nPlotting Tilt vs Longitude detailed analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    
    # =====================
    # Panel 1: Scatter + regression (overall)
    # =====================
    ax1 = axes[0, 0]
    
    ax1.scatter(df["center_lon"], df["tilt"], alpha=0.3, s=10, c='gray', label='Daily data')
    
    # Regression line
    slope, intercept, r, p, se = stats.linregress(df["center_lon"], df["tilt"])
    x_line = np.linspace(df["center_lon"].min(), df["center_lon"].max(), 100)
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'r-', linewidth=2, 
             label=f'Linear fit: y={slope:.3f}x+{intercept:.1f}\nR2={r**2:.3f}, p<0.001')
    
    ax1.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green', label='MC')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax1.set_xlabel("Longitude (E)", fontsize=11)
    ax1.set_ylabel("Daily Tilt (deg)", fontsize=11)
    ax1.set_title("(a) Overall: Tilt vs Longitude + Linear Regression", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_xlim(40, 220)
    
    # =====================
    # Panel 2: Regression by ENSO group
    # =====================
    ax2 = axes[0, 1]
    
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        ax2.scatter(subset["center_lon"], subset["tilt"], 
                   c=ENSO_COLORS[phase], alpha=0.3, s=8, label=f'{phase} data')
        
        if len(subset) > 10:
            s, i, r, p, se = stats.linregress(subset["center_lon"], subset["tilt"])
            x_line = np.linspace(subset["center_lon"].min(), subset["center_lon"].max(), 100)
            y_line = s * x_line + i
            ax2.plot(x_line, y_line, color=ENSO_COLORS[phase], linewidth=2, linestyle='--',
                    label=f'{phase}: slope={s:.3f}')
    
    ax2.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax2.set_xlabel("Longitude (E)", fontsize=11)
    ax2.set_ylabel("Daily Tilt (deg)", fontsize=11)
    ax2.set_title("(b) By ENSO Phase: Tilt vs Longitude", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(linestyle='--', alpha=0.3)
    ax2.set_xlim(40, 220)
    
    # =====================
    # Panel 3: Binned mean curve
    # =====================
    ax3 = axes[1, 0]
    
    lon_bins = np.arange(40, 221, 10)
    lon_centers = (lon_bins[:-1] + lon_bins[1:]) / 2
    
    # Overall
    means_all = []
    stds_all = []
    for j in range(len(lon_bins) - 1):
        bin_data = df[(df["center_lon"] >= lon_bins[j]) & (df["center_lon"] < lon_bins[j+1])]["tilt"]
        if len(bin_data) >= 5:
            means_all.append(bin_data.mean())
            stds_all.append(bin_data.std())
        else:
            means_all.append(np.nan)
            stds_all.append(np.nan)
    
    means_all = np.array(means_all)
    stds_all = np.array(stds_all)
    valid = ~np.isnan(means_all)
    
    ax3.plot(lon_centers[valid], means_all[valid], 'k-', linewidth=2.5, marker='o', 
             markersize=6, label='All ENSO (Mean)')
    ax3.fill_between(lon_centers[valid], 
                     means_all[valid] - stds_all[valid],
                     means_all[valid] + stds_all[valid],
                     color='gray', alpha=0.2, label='+/- 1 Std')
    
    ax3.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green', label='MC')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax3.set_xlabel("Longitude (E)", fontsize=11)
    ax3.set_ylabel("Mean Tilt +/- Std (deg)", fontsize=11)
    ax3.set_title("(c) Binned Statistics: Mean +/- Std (10 deg bins)", fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(linestyle='--', alpha=0.3)
    ax3.set_xlim(40, 220)
    
    # =====================
    # Panel 4: 2D density (hexbin)
    # =====================
    ax4 = axes[1, 1]
    
    hb = ax4.hexbin(df["center_lon"], df["tilt"], gridsize=30, cmap='YlOrRd', 
                    mincnt=1, extent=[40, 220, df["tilt"].min(), df["tilt"].max()])
    cb = plt.colorbar(hb, ax=ax4, label='Count')
    
    ax4.axvspan(MC_WEST, MC_EAST, alpha=0.15, color='green', label='MC')
    ax4.axhline(0, color='white', linestyle='-', linewidth=1, alpha=0.7)
    ax4.set_xlabel("Longitude (E)", fontsize=11)
    ax4.set_ylabel("Daily Tilt (deg)", fontsize=11)
    ax4.set_title("(d) 2D Density Distribution (Hexbin)", fontsize=12, fontweight='bold')
    ax4.set_xlim(40, 220)
    
    plt.suptitle("Daily Tilt vs Longitude Relationship Analysis", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    out_path = FIG_DIR / "tilt_vs_longitude_detailed.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def plot_tilt_longitude_relationship(df: pd.DataFrame):
    """Plot supplementary Tilt-Longitude analysis"""
    print("\nPlotting Tilt-Longitude supplementary analysis...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    
    # =====================
    # Panel 1: Regional boxplot
    # =====================
    ax1 = axes[0]
    
    df_copy = df.copy()
    df_copy["region"] = pd.cut(df_copy["center_lon"], 
                                bins=[40, 100, 160, 220],
                                labels=["Indian Ocean\n(40-100E)", 
                                       "Maritime Cont.\n(100-160E)", 
                                       "Western Pacific\n(160-220E)"])
    
    region_order = ["Indian Ocean\n(40-100E)", "Maritime Cont.\n(100-160E)", "Western Pacific\n(160-220E)"]
    region_colors = ["#3498DB", "#27AE60", "#E74C3C"]
    
    sns.boxplot(x="region", y="tilt", data=df_copy.dropna(subset=["region"]),
                order=region_order, palette=region_colors, ax=ax1, width=0.5)
    
    ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_xlabel("Region", fontsize=11)
    ax1.set_ylabel("Daily Tilt (deg)", fontsize=11)
    ax1.set_title("(a) Tilt Distribution by Region", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # =====================
    # Panel 2: Region x ENSO mean comparison
    # =====================
    ax2 = axes[1]
    
    region_labels = ["IO (40-100E)", "MC (100-160E)", "WP (160-220E)"]
    region_bounds = [(40, 100), (100, 160), (160, 220)]
    
    x = np.arange(len(region_labels))
    width = 0.25
    
    for i, phase in enumerate(ENSO_ORDER):
        means = []
        sems = []
        for lon_min, lon_max in region_bounds:
            region_data = df[(df["enso_phase"] == phase) & 
                            (df["center_lon"] >= lon_min) & 
                            (df["center_lon"] < lon_max)]["tilt"]
            if len(region_data) > 0:
                means.append(region_data.mean())
                sems.append(region_data.sem() if len(region_data) > 1 else 0)
            else:
                means.append(np.nan)
                sems.append(np.nan)
        
        offset = (i - 1) * width
        ax2.bar(x + offset, means, width, yerr=sems, capsize=3,
               label=phase, color=ENSO_COLORS[phase], edgecolor='black', alpha=0.8)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(region_labels)
    ax2.set_xlabel("Region", fontsize=11)
    ax2.set_ylabel("Mean Tilt +/- SEM (deg)", fontsize=11)
    ax2.set_title("(b) Mean Tilt by Region x ENSO", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # =====================
    # Panel 3: Tilt-Longitude slope comparison
    # =====================
    ax3 = axes[2]
    
    slopes = []
    slope_sems = []
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        if len(subset) > 10:
            s, i, r, p, se = stats.linregress(subset["center_lon"], subset["tilt"])
            slopes.append(s)
            slope_sems.append(se)
        else:
            slopes.append(np.nan)
            slope_sems.append(np.nan)
    
    x_phase = np.arange(len(ENSO_ORDER))
    bars = ax3.bar(x_phase, slopes, 0.6, yerr=slope_sems, capsize=5,
                   color=[ENSO_COLORS[p] for p in ENSO_ORDER], edgecolor='black', alpha=0.8)
    
    ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xticks(x_phase)
    ax3.set_xticklabels(ENSO_ORDER)
    ax3.set_xlabel("ENSO Phase", fontsize=11)
    ax3.set_ylabel("Slope (deg Tilt / deg Lon)", fontsize=11)
    ax3.set_title("(c) Tilt-Longitude Regression Slope", fontsize=12, fontweight='bold')
    ax3.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar, s, se in zip(bars, slopes, slope_sems):
        if not np.isnan(s):
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (se if s > 0 else -se) + 0.01,
                    f'{s:.4f}', ha='center', va='bottom' if s > 0 else 'top', fontsize=10)
    
    plt.suptitle("Tilt-Longitude Relationship: Supplementary Analysis", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = FIG_DIR / "tilt_longitude_relationship_supplement.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    print("="*70)
    print("MC Residence Analysis + Tilt-Longitude Relationship")
    print("="*70)
    
    # Load data
    df_daily, events, enso_stats = load_daily_data()
    
    # Q1: MC residence fraction analysis
    df_event_res = analyze_residence_fraction(df_daily, enso_stats)
    plot_residence_fraction(df_event_res)
    
    # Q2: Tilt vs Longitude analysis
    analyze_tilt_vs_longitude(df_daily)
    plot_tilt_vs_longitude_detailed(df_daily)
    plot_tilt_longitude_relationship(df_daily)
    
    print("\n" + "="*70)
    print("All analyses completed!")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
