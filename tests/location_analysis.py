# -*- coding: utf-8 -*-
"""
location_analysis.py — MJO 位置效应与停留时间综合分析

功能：
    分析 MJO 对流中心经度在不同 ENSO 相位下的分布特征，
    以及 Tilt 与经度的关系和海洋大陆停留时间比例。
输入：
    mjo_mvEOF_step3_1979-2022.nc, tilt_daily_step4_layermean_1979-2022.nc,
    mjo_events_step3_1979-2022.csv, tilt_event_stats_with_enso_1979-2022.csv
输出：
    figures/location_effect/ 下的经度分布、KDE、散点、分箱、MC 停留统计等图表
用法：
    python tests/location_analysis.py             # 全部
    python tests/location_analysis.py effect      # 位置效应
    python tests/location_analysis.py residence   # MC 停留 + Tilt-Longitude
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ======================
# PATHS
# ======================
STEP3_NC       = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
TILT_NC        = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV     = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\location")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# SETTINGS
# ======================
MC_WEST = 100
MC_EAST = 160
WP_EAST = 180

ENSO_ORDER  = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}

LON_BINS = np.arange(40, 221, 10)


# ======================
# SHARED DATA LOADING
# ======================
def to_lon_360(lon):
    return np.mod(lon, 360)


def load_daily_data():
    """加载逐日 center_lon + tilt，归属到事件 + ENSO 分组"""
    print("Loading data...")
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon = ds3["center_lon_track"].to_series()
    center_lon.name = "center_lon"

    ds_tilt = xr.open_dataset(TILT_NC)
    tilt = ds_tilt["tilt"].to_series()
    tilt.name = "tilt"

    df = pd.concat([center_lon, tilt], axis=1).dropna()
    df["center_lon"] = to_lon_360(df["center_lon"])

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    enso_stats = pd.read_csv(ENSO_STATS_CSV, parse_dates=["start_date", "end_date"])
    enso_map = dict(zip(enso_stats["event_id"], enso_stats["enso_phase"]))

    df["event_id"] = np.nan
    df["enso_phase"] = ""
    for _, ev in events.iterrows():
        mask = (df.index >= ev["start_date"]) & (df.index <= ev["end_date"])
        df.loc[mask, "event_id"] = ev["event_id"]
        df.loc[mask, "enso_phase"] = enso_map.get(ev["event_id"], "Unknown")

    df = df[df["event_id"].notna() & df["enso_phase"].isin(ENSO_ORDER)].copy()
    df["in_MC"] = (df["center_lon"] >= MC_WEST) & (df["center_lon"] < MC_EAST)
    print(f"  Event days: {len(df)}")
    return df, events, enso_stats


# ============================================================
# 1. EFFECT (from location_effect_analysis.py)
# ============================================================
def run_effect():
    """位置效应分析：经度分布 + Tilt-Longitude 关系"""
    df, events, enso_stats = load_daily_data()

    # --- 1a. 经度分布直方图 ---
    print("\n[1/5] Longitude histogram...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=150)
    for i, phase in enumerate(ENSO_ORDER):
        ax = axes[i]
        sub = df[df["enso_phase"] == phase]["center_lon"]
        ax.hist(sub, bins=LON_BINS, color=ENSO_COLORS[phase], edgecolor='black', alpha=0.7)
        ax.axvspan(MC_WEST, MC_EAST, alpha=0.15, color='green', label=f'MC')
        ax.axvline(sub.mean(), color='red', lw=2, label=f'Mean: {sub.mean():.1f}°')
        ax.set_xlabel("Longitude (°E)")
        if i == 0: ax.set_ylabel("Number of Days")
        ax.set_title(f"{phase} (N={len(sub)})\nMean={sub.mean():.1f}°, Std={sub.std():.1f}°")
        ax.legend(fontsize=8)
        ax.grid(axis='y', ls='--', alpha=0.3)
        ax.set_xlim(40, 220)
    plt.suptitle("MJO Center Longitude Distribution by ENSO Phase", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "longitude_histogram_by_enso.png", bbox_inches='tight')
    plt.close()

    # --- 1b. KDE ---
    print("[2/5] Longitude KDE...")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]["center_lon"]
        if len(sub) > 10:
            sns.kdeplot(sub, ax=ax, label=f"{phase} (N={len(sub)})",
                        color=ENSO_COLORS[phase], lw=2, fill=True, alpha=0.3)
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green', label='MC')
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Density")
    ax.set_title("MJO Center Longitude KDE by ENSO Phase", fontweight='bold')
    ax.legend()
    ax.set_xlim(40, 220)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "longitude_kde_overlay_by_enso.png", bbox_inches='tight')
    plt.close()

    # --- 1c. Tilt vs Longitude scatter ---
    print("[3/5] Tilt vs Longitude scatter...")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]
        ax.scatter(sub["center_lon"], sub["tilt"], c=ENSO_COLORS[phase],
                   label=f"{phase} (N={len(sub)})", alpha=0.5, s=15, edgecolors='none')
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green', label='MC')
    ax.axhline(0, color='black', alpha=0.5)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Daily Tilt (deg)")
    ax.set_title("MJO Tilt vs. Center Longitude by ENSO Phase", fontweight='bold')
    ax.legend()
    ax.set_xlim(40, 220)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tilt_vs_longitude_scatter_by_enso.png", bbox_inches='tight')
    plt.close()

    # --- 1d. Binned Tilt vs Longitude ---
    print("[4/5] Binned Tilt vs Longitude...")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    edges = np.arange(40, 221, 15)
    centers = (edges[:-1] + edges[1:]) / 2
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]
        means, sems = [], []
        for j in range(len(edges) - 1):
            bm = (sub["center_lon"] >= edges[j]) & (sub["center_lon"] < edges[j + 1])
            bd = sub.loc[bm, "tilt"]
            means.append(bd.mean() if len(bd) >= 5 else np.nan)
            sems.append(bd.sem() if len(bd) >= 5 else np.nan)
        means, sems = np.array(means), np.array(sems)
        v = ~np.isnan(means)
        ax.plot(centers[v], means[v], color=ENSO_COLORS[phase], lw=2, marker='o', ms=6, label=phase)
        ax.fill_between(centers[v], means[v] - sems[v], means[v] + sems[v],
                        color=ENSO_COLORS[phase], alpha=0.2)
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green', label='MC')
    ax.axhline(0, color='black', alpha=0.5)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Mean Tilt ± SEM (deg)")
    ax.set_title("Mean MJO Tilt vs. Longitude (15° bins, N≥5)", fontweight='bold')
    ax.legend()
    ax.set_xlim(40, 220)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tilt_vs_longitude_binned_by_enso.png", bbox_inches='tight')
    plt.close()

    # --- 1e. Combined 2x2 ---
    print("[5/5] Combined 2x2 analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

    # (a) KDE
    ax = axes[0, 0]
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]["center_lon"]
        if len(sub) > 10:
            sns.kdeplot(sub, ax=ax, label=f"{phase}", color=ENSO_COLORS[phase], lw=2)
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax.set_xlim(40, 220)
    ax.set_title("(a) Longitude Distribution", fontweight='bold')
    ax.legend(fontsize=9)

    # (b) Boxplot
    ax = axes[0, 1]
    sns.boxplot(x="enso_phase", y="center_lon", data=df, order=ENSO_ORDER,
                hue="enso_phase", palette=ENSO_COLORS, ax=ax, width=0.5, legend=False)
    ax.axhline(MC_WEST, color='green', ls='--', alpha=0.7)
    ax.axhline(MC_EAST, color='green', ls='--', alpha=0.7)
    ax.set_title("(b) Longitude Boxplot", fontweight='bold')

    # (c) Scatter
    ax = axes[1, 0]
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]
        ax.scatter(sub["center_lon"], sub["tilt"], c=ENSO_COLORS[phase], alpha=0.4, s=10, label=phase)
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax.axhline(0, color='black', alpha=0.5)
    ax.set_xlim(40, 220)
    ax.set_title("(c) Tilt vs. Longitude Scatter", fontweight='bold')
    ax.legend(fontsize=9)

    # (d) Binned
    ax = axes[1, 1]
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]
        means, sems = [], []
        for j in range(len(edges) - 1):
            bm = (sub["center_lon"] >= edges[j]) & (sub["center_lon"] < edges[j + 1])
            bd = sub.loc[bm, "tilt"]
            means.append(bd.mean() if len(bd) >= 5 else np.nan)
            sems.append(bd.sem() if len(bd) >= 5 else np.nan)
        means, sems = np.array(means), np.array(sems)
        v = ~np.isnan(means)
        ax.plot(centers[v], means[v], color=ENSO_COLORS[phase], lw=2, marker='o', ms=5, label=phase)
        ax.fill_between(centers[v], means[v] - sems[v], means[v] + sems[v],
                        color=ENSO_COLORS[phase], alpha=0.2)
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax.axhline(0, color='black', alpha=0.5)
    ax.set_xlim(40, 220)
    ax.set_title("(d) Binned Mean Tilt", fontweight='bold')
    ax.legend(fontsize=9)

    plt.suptitle("Location Effect Analysis", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "location_effect_combined_analysis.png", bbox_inches='tight')
    plt.close()

    # --- Regional statistics ---
    print("\nRegional Tilt Statistics:")
    regions = {
        "Indian Ocean (40-100°E)": (40, 100),
        "Maritime Continent (100-160°E)": (100, 160),
        "Western Pacific (160-180°E)": (160, 180),
        "Central Pacific (180-220°E)": (180, 220),
    }
    results = []
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]
        for rname, (lo, hi) in regions.items():
            rd = sub.loc[(sub["center_lon"] >= lo) & (sub["center_lon"] < hi), "tilt"]
            if len(rd) > 0:
                results.append({"enso_phase": phase, "region": rname,
                                "n": len(rd), "mean_tilt": rd.mean(), "std_tilt": rd.std()})
                print(f"  {phase:10s} {rname}: N={len(rd):4d}, Mean={rd.mean():+6.2f}°")
    pd.DataFrame(results).to_csv(FIG_DIR / "regional_tilt_statistics.csv", index=False)

    # MC vs Outside t-test
    print("\nMC vs Outside t-test:")
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]
        mc = sub.loc[sub["in_MC"], "tilt"]
        out = sub.loc[~sub["in_MC"], "tilt"]
        if len(mc) > 10 and len(out) > 10:
            t, p = stats.ttest_ind(mc, out, equal_var=False)
            print(f"  {phase}: MC={mc.mean():.2f} vs Out={out.mean():.2f}, t={t:.3f}, p={p:.4f}")

    print("  Effect analysis done!")


# ============================================================
# 2. RESIDENCE (from location_residence_analysis.py)
# ============================================================
def run_residence():
    """MC 停留时间 + 详细 Tilt-Longitude 分析"""
    df, events, enso_stats = load_daily_data()

    # --- Q1: MC Residence Fraction ---
    print("\n[Q1] MC Residence Fraction by ENSO Phase:")
    event_results = []
    for phase in ENSO_ORDER:
        phase_eids = enso_stats[enso_stats["enso_phase"] == phase]["event_id"].values
        for eid in phase_eids:
            ed = df[df["event_id"] == eid]
            if len(ed) == 0:
                continue
            event_results.append({"event_id": eid, "enso_phase": phase,
                                  "total_days": len(ed), "mc_days": int(ed["in_MC"].sum()),
                                  "mc_fraction": ed["in_MC"].mean() * 100})
    df_res = pd.DataFrame(event_results)

    for phase in ENSO_ORDER:
        pf = df_res[df_res["enso_phase"] == phase]
        print(f"  {phase}: {len(pf)} events, mean MC%={pf['mc_fraction'].mean():.1f}%")

    # T-tests
    for i, p1 in enumerate(ENSO_ORDER):
        for p2 in ENSO_ORDER[i + 1:]:
            g1 = df_res[df_res["enso_phase"] == p1]["mc_fraction"]
            g2 = df_res[df_res["enso_phase"] == p2]["mc_fraction"]
            if len(g1) > 1 and len(g2) > 1:
                t, p = stats.ttest_ind(g1, g2, equal_var=False)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                print(f"  {p1} vs {p2}: t={t:+.2f}, p={p:.4f} {sig}")
    df_res.to_csv(FIG_DIR / "mc_residence_fraction_by_enso.csv", index=False)

    # Plot residence fraction
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

    # (a) Stacked bar
    ax = axes[0]
    mc_c = [df_res[df_res["enso_phase"] == p]["mc_days"].sum() for p in ENSO_ORDER]
    out_c = [df_res[df_res["enso_phase"] == p]["total_days"].sum() - m for p, m in zip(ENSO_ORDER, mc_c)]
    x = np.arange(3)
    ax.bar(x, mc_c, 0.6, label='Inside MC', color='green', alpha=0.7)
    ax.bar(x, out_c, 0.6, bottom=mc_c, label='Outside MC', color='orange', alpha=0.7)
    for i, (m, o) in enumerate(zip(mc_c, out_c)):
        tot = m + o
        ax.text(i, m / 2, f'{m / tot * 100:.0f}%', ha='center', va='center', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ENSO_ORDER)
    ax.set_ylabel("Days")
    ax.set_title("(a) Total Days Distribution", fontweight='bold')
    ax.legend()

    # (b) Boxplot
    ax = axes[1]
    sns.boxplot(x="enso_phase", y="mc_fraction", data=df_res, order=ENSO_ORDER,
                hue="enso_phase", palette=ENSO_COLORS, ax=ax, width=0.5, legend=False)
    sns.swarmplot(x="enso_phase", y="mc_fraction", data=df_res, order=ENSO_ORDER,
                  color=".2", alpha=0.5, size=4, ax=ax)
    ax.set_ylabel("MC Fraction (%)")
    ax.set_title("(b) Event-level MC Fraction", fontweight='bold')

    # (c) Mean bar
    ax = axes[2]
    means = [df_res[df_res["enso_phase"] == p]["mc_fraction"].mean() for p in ENSO_ORDER]
    sems = [df_res[df_res["enso_phase"] == p]["mc_fraction"].sem() for p in ENSO_ORDER]
    ns = [len(df_res[df_res["enso_phase"] == p]) for p in ENSO_ORDER]
    bars = ax.bar(x, means, 0.6, yerr=sems, capsize=5,
                  color=[ENSO_COLORS[p] for p in ENSO_ORDER], edgecolor='black', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n(N={n})" for p, n in zip(ENSO_ORDER, ns)])
    ax.set_ylabel("Mean MC Fraction (%)")
    ax.set_title("(c) Mean MC% ± SEM", fontweight='bold')
    for bar, m, s in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 1,
                f'{m:.1f}%', ha='center', va='bottom')

    plt.suptitle("MJO MC Residence by ENSO Phase", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "mc_residence_fraction_comparison.png", bbox_inches='tight')
    plt.close()

    # --- Q2: Detailed Tilt vs Longitude ---
    print("\n[Q2] Tilt vs Longitude:")
    corr, pc = stats.pearsonr(df["center_lon"], df["tilt"])
    slope, intercept, r, p, se = stats.linregress(df["center_lon"], df["tilt"])
    print(f"  Overall: r={corr:+.4f}, slope={slope:+.4f}")

    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]
        if len(sub) > 10:
            c, _ = stats.pearsonr(sub["center_lon"], sub["tilt"])
            s, *_ = stats.linregress(sub["center_lon"], sub["tilt"])
            print(f"  {phase}: r={c:+.3f}, slope={s:+.4f}")

    # 2x2 detailed plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

    ax = axes[0, 0]
    ax.scatter(df["center_lon"], df["tilt"], alpha=0.3, s=10, c='gray')
    xl = np.linspace(df["center_lon"].min(), df["center_lon"].max(), 100)
    ax.plot(xl, slope * xl + intercept, 'r-', lw=2, label=f'R²={r**2:.3f}')
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax.set_title("(a) Overall Scatter + Regression", fontweight='bold')
    ax.legend()

    ax = axes[0, 1]
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]
        ax.scatter(sub["center_lon"], sub["tilt"], c=ENSO_COLORS[phase], alpha=0.3, s=8)
        if len(sub) > 10:
            s, i, *_ = stats.linregress(sub["center_lon"], sub["tilt"])
            xl = np.linspace(sub["center_lon"].min(), sub["center_lon"].max(), 100)
            ax.plot(xl, s * xl + i, color=ENSO_COLORS[phase], lw=2, ls='--', label=f'{phase}: slope={s:.3f}')
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax.set_title("(b) By ENSO Phase", fontweight='bold')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    lb = np.arange(40, 221, 10)
    lc = (lb[:-1] + lb[1:]) / 2
    means, stds = [], []
    for j in range(len(lb) - 1):
        bd = df[(df["center_lon"] >= lb[j]) & (df["center_lon"] < lb[j + 1])]["tilt"]
        means.append(bd.mean() if len(bd) >= 5 else np.nan)
        stds.append(bd.std() if len(bd) >= 5 else np.nan)
    means, stds = np.array(means), np.array(stds)
    v = ~np.isnan(means)
    ax.plot(lc[v], means[v], 'k-', lw=2.5, marker='o', ms=6)
    ax.fill_between(lc[v], means[v] - stds[v], means[v] + stds[v], color='gray', alpha=0.2)
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax.set_title("(c) Binned Mean ± Std (10° bins)", fontweight='bold')

    ax = axes[1, 1]
    ax.hexbin(df["center_lon"], df["tilt"], gridsize=30, cmap='YlOrRd', mincnt=1,
              extent=[40, 220, df["tilt"].min(), df["tilt"].max()])
    plt.colorbar(ax.collections[0], ax=ax, label='Count')
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.15, color='green')
    ax.set_title("(d) 2D Density (Hexbin)", fontweight='bold')

    for a in axes.flat:
        a.set_xlim(40, 220)
        a.grid(ls='--', alpha=0.3)
    plt.suptitle("Daily Tilt vs Longitude Analysis", fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tilt_vs_longitude_detailed.png", bbox_inches='tight')
    plt.close()

    # Supplementary 1x3
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    ax = axes[0]
    df_copy = df.copy()
    df_copy["region"] = pd.cut(df_copy["center_lon"], bins=[40, 100, 160, 220],
                               labels=["IO\n(40-100E)", "MC\n(100-160E)", "WP\n(160-220E)"])
    sns.boxplot(x="region", y="tilt", data=df_copy.dropna(subset=["region"]),
                palette=["#3498DB", "#27AE60", "#E74C3C"], ax=ax, width=0.5)
    ax.axhline(0, ls='--', alpha=0.5)
    ax.set_title("(a) Tilt by Region", fontweight='bold')

    ax = axes[1]
    rbounds = [(40, 100), (100, 160), (160, 220)]
    rlabels = ["IO", "MC", "WP"]
    xr_ = np.arange(3)
    w = 0.25
    for i, phase in enumerate(ENSO_ORDER):
        ms = [df[(df["enso_phase"] == phase) & (df["center_lon"] >= lo) &
                 (df["center_lon"] < hi)]["tilt"].mean() for lo, hi in rbounds]
        ses = [df[(df["enso_phase"] == phase) & (df["center_lon"] >= lo) &
                  (df["center_lon"] < hi)]["tilt"].sem() for lo, hi in rbounds]
        ax.bar(xr_ + (i - 1) * w, ms, w, yerr=ses, capsize=3,
               label=phase, color=ENSO_COLORS[phase], edgecolor='black', alpha=0.8)
    ax.set_xticks(xr_)
    ax.set_xticklabels(rlabels)
    ax.axhline(0, ls='--', alpha=0.5)
    ax.set_title("(b) Mean Tilt by Region × ENSO", fontweight='bold')
    ax.legend(fontsize=9)

    ax = axes[2]
    slopes, slope_ses = [], []
    for phase in ENSO_ORDER:
        sub = df[df["enso_phase"] == phase]
        if len(sub) > 10:
            s, _, _, _, se = stats.linregress(sub["center_lon"], sub["tilt"])
            slopes.append(s)
            slope_ses.append(se)
        else:
            slopes.append(np.nan)
            slope_ses.append(np.nan)
    ax.bar(np.arange(3), slopes, 0.6, yerr=slope_ses, capsize=5,
           color=[ENSO_COLORS[p] for p in ENSO_ORDER], edgecolor='black', alpha=0.8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(ENSO_ORDER)
    ax.axhline(0, ls='--', alpha=0.5)
    ax.set_title("(c) Regression Slope", fontweight='bold')

    plt.suptitle("Tilt-Longitude Supplementary", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "tilt_longitude_relationship_supplement.png", bbox_inches='tight')
    plt.close()

    print("  Residence analysis done!")


# ============================================================
# MAIN
# ============================================================
ANALYSES = {"effect": run_effect, "residence": run_residence}


def main():
    print("=" * 70)
    print("Location Analysis (Effect + Residence)")
    print("=" * 70)
    if len(sys.argv) > 1:
        name = sys.argv[1].lower()
        if name in ANALYSES:
            ANALYSES[name]()
        else:
            print(f"Unknown: {name}. Available: {list(ANALYSES.keys())}")
    else:
        for func in ANALYSES.values():
            func()
    print("\nDone!")


if __name__ == "__main__":
    main()
