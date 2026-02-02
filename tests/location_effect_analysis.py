# -*- coding: utf-8 -*-
"""
location_effect_analysis.py: 假设1 - 位置效应验证

================================================================================
功能描述：
    本脚本验证 MJO 位置效应假设，分析不同 ENSO 相位下 MJO 对流中心的经度分布差异，
    以及 MJO 垂直倾斜与地理位置的关系。
    
科学问题：
    - ENSO 如何影响 MJO 的活动区域？
    - MJO 倾斜是否与所在经度位置相关？
    
物理机制：
    - El Niño 时期：暖池东移，MJO 可延伸至中太平洋开阔洋面（摩擦较小）
    - La Niña 时期：MJO 主要活跃于西太平洋近海洋大陆区域（地形复杂，摩擦大）
    
主要分析内容：
    1. 三组 ENSO 相位的 MJO 中心经度分布直方图
    2. 经度-倾斜散点图及分箱统计
    3. 海洋大陆区域 vs 非海洋大陆区域的倾斜对比
    4. KDE 核密度估计可视化
- Neutral 年份 MJO 可能更多被截留在海洋大陆 (Maritime Continent, MC)
- MC 的复杂地形会破坏低层结构的相干性，导致计算出的倾斜度变小（显得直立）

分析内容：
1. 三组事件的经度分布直方图
2. Tilt vs. Longitude 散点图（按 ENSO 状态分组）

Inputs:
- Step3 NC: center_lon_track(time) - MJO 对流中心经度轨迹
- Tilt Daily NC: tilt(time) - 逐日倾斜度
- Events CSV: mjo_events_step3_1979-2022.csv
- ENSO classified stats: tilt_event_stats_with_enso_1979-2022.csv

Run:
  python E:\\Projects\\ENSO_MJO_Tilt\\tests\\location_effect_analysis.py
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
# 关键经度区域（度）
MC_WEST = 100   # 海洋大陆西界
MC_EAST = 160   # 海洋大陆东界
WP_EAST = 180   # 西太平洋东界

# ENSO 分组颜色和顺序
ENSO_ORDER = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}

# 直方图分箱
LON_BINS = np.arange(40, 221, 10)  # 40E to 220E (=140W)


def to_lon_360(lon: np.ndarray) -> np.ndarray:
    """Convert longitude to 0-360 range."""
    return np.mod(lon, 360)


def load_daily_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载逐日数据：MJO 中心经度 + Tilt 值
    返回按事件归类的逐日 DataFrame
    """
    print("Loading Step3 NC (center_lon_track)...")
    ds3 = xr.open_dataset(STEP3_NC)
    if "center_lon_track" not in ds3:
        raise KeyError(f"center_lon_track not found. Vars = {list(ds3.data_vars)}")
    
    center_lon = ds3["center_lon_track"].to_series()
    center_lon.name = "center_lon"
    
    print("Loading Tilt NC...")
    ds_tilt = xr.open_dataset(TILT_NC)
    tilt = ds_tilt["tilt"].to_series()
    tilt.name = "tilt"
    
    # 合并
    df_daily = pd.concat([center_lon, tilt], axis=1).dropna()
    df_daily["center_lon"] = to_lon_360(df_daily["center_lon"])
    
    print(f"  Daily data: {len(df_daily)} valid samples")
    
    # 加载事件列表
    print("Loading Events CSV...")
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    
    # 加载 ENSO 分类
    print("Loading ENSO Classification...")
    enso_stats = pd.read_csv(ENSO_STATS_CSV, parse_dates=["start_date", "end_date"])
    
    # 构建 event_id -> enso_phase 映射
    enso_map = dict(zip(enso_stats["event_id"], enso_stats["enso_phase"]))
    
    # 把每日数据归属到事件
    df_daily["event_id"] = np.nan
    df_daily["enso_phase"] = ""
    
    for _, ev in events.iterrows():
        eid = ev["event_id"]
        s = ev["start_date"]
        e = ev["end_date"]
        mask = (df_daily.index >= s) & (df_daily.index <= e)
        df_daily.loc[mask, "event_id"] = eid
        df_daily.loc[mask, "enso_phase"] = enso_map.get(eid, "Unknown")
    
    # 过滤只保留属于事件的数据
    df_event_days = df_daily[df_daily["event_id"].notna()].copy()
    df_event_days = df_event_days[df_event_days["enso_phase"].isin(ENSO_ORDER)]
    
    print(f"  Event days with ENSO classification: {len(df_event_days)}")
    
    return df_event_days, enso_stats


def plot_longitude_histogram(df: pd.DataFrame) -> None:
    """
    绘制三组事件的经度分布直方图
    """
    print("\nGenerating Longitude Distribution Histogram...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, dpi=150)
    
    for i, phase in enumerate(ENSO_ORDER):
        ax = axes[i]
        subset = df[df["enso_phase"] == phase]["center_lon"]
        
        # 绘制直方图
        n, bins, patches = ax.hist(subset, bins=LON_BINS, 
                                    color=ENSO_COLORS[phase], 
                                    edgecolor='black', alpha=0.7)
        
        # 标记关键经度区域
        ax.axvspan(MC_WEST, MC_EAST, alpha=0.15, color='green', 
                   label=f'MC ({MC_WEST}°-{MC_EAST}°E)')
        ax.axvline(MC_WEST, color='green', linestyle='--', linewidth=1)
        ax.axvline(MC_EAST, color='green', linestyle='--', linewidth=1)
        ax.axvline(WP_EAST, color='gray', linestyle=':', linewidth=1, 
                   label=f'Dateline ({WP_EAST}°)')
        
        # 计算统计量
        mean_lon = subset.mean()
        median_lon = subset.median()
        std_lon = subset.std()
        
        ax.axvline(mean_lon, color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {mean_lon:.1f}°')
        
        ax.set_xlabel("Longitude (°E)")
        if i == 0:
            ax.set_ylabel("Number of Days")
        ax.set_title(f"{phase} (N={len(subset)})\nMean={mean_lon:.1f}°, Std={std_lon:.1f}°")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.set_xlim(40, 220)
    
    plt.suptitle("MJO Convective Center Longitude Distribution by ENSO Phase", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = FIG_DIR / "longitude_histogram_by_enso.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def plot_longitude_kde_overlay(df: pd.DataFrame) -> None:
    """
    绘制三组事件的经度 KDE 叠加图
    """
    print("\nGenerating Longitude KDE Overlay...")
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]["center_lon"]
        if len(subset) > 10:
            sns.kdeplot(subset, ax=ax, label=f"{phase} (N={len(subset)})", 
                       color=ENSO_COLORS[phase], linewidth=2, fill=True, alpha=0.3)
    
    # 标记关键区域
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green', label='Maritime Continent')
    ax.axvline(MC_WEST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(MC_EAST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(WP_EAST, color='gray', linestyle=':', linewidth=1.5, label='Dateline (180°)')
    
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("MJO Convective Center Longitude Distribution (KDE) by ENSO Phase", 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='both', linestyle='--', alpha=0.3)
    ax.set_xlim(40, 220)
    
    plt.tight_layout()
    
    out_path = FIG_DIR / "longitude_kde_overlay_by_enso.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def plot_tilt_vs_longitude_scatter(df: pd.DataFrame) -> None:
    """
    绘制 Tilt vs. Longitude 散点图（按 ENSO 分组）
    """
    print("\nGenerating Tilt vs. Longitude Scatter Plot...")
    
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        ax.scatter(subset["center_lon"], subset["tilt"], 
                  c=ENSO_COLORS[phase], label=f"{phase} (N={len(subset)})", 
                  alpha=0.5, s=15, edgecolors='none')
    
    # 标记关键经度区域
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green', label='Maritime Continent')
    ax.axvline(MC_WEST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(MC_EAST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(WP_EAST, color='gray', linestyle=':', linewidth=1.5, label='Dateline (180°)')
    
    # 水平参考线
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Daily Tilt (deg)", fontsize=12)
    ax.set_title("MJO Tilt vs. Convective Center Longitude by ENSO Phase", 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_xlim(40, 220)
    
    plt.tight_layout()
    
    out_path = FIG_DIR / "tilt_vs_longitude_scatter_by_enso.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def plot_tilt_vs_longitude_binned(df: pd.DataFrame) -> None:
    """
    绘制分箱统计的 Tilt vs. Longitude 曲线（均值 + 误差带）
    """
    print("\nGenerating Binned Tilt vs. Longitude Plot...")
    
    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    
    # 经度分箱
    lon_bin_edges = np.arange(40, 221, 15)  # 15度一箱
    lon_bin_centers = (lon_bin_edges[:-1] + lon_bin_edges[1:]) / 2
    
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        
        means = []
        stds = []
        sems = []
        counts = []
        
        for j in range(len(lon_bin_edges) - 1):
            bin_mask = (subset["center_lon"] >= lon_bin_edges[j]) & \
                       (subset["center_lon"] < lon_bin_edges[j+1])
            bin_data = subset.loc[bin_mask, "tilt"]
            
            if len(bin_data) >= 5:  # 至少5个样本才统计
                means.append(bin_data.mean())
                stds.append(bin_data.std())
                sems.append(bin_data.sem())
                counts.append(len(bin_data))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                sems.append(np.nan)
                counts.append(len(bin_data))
        
        means = np.array(means)
        sems = np.array(sems)
        
        # 绘制均值线和误差带
        valid = ~np.isnan(means)
        ax.plot(lon_bin_centers[valid], means[valid], 
               color=ENSO_COLORS[phase], linewidth=2, marker='o', 
               markersize=6, label=phase)
        ax.fill_between(lon_bin_centers[valid], 
                       means[valid] - sems[valid], 
                       means[valid] + sems[valid],
                       color=ENSO_COLORS[phase], alpha=0.2)
    
    # 标记关键区域
    ax.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green', label='Maritime Continent')
    ax.axvline(MC_WEST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(MC_EAST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(WP_EAST, color='gray', linestyle=':', linewidth=1.5, label='Dateline')
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel("Longitude (°E)", fontsize=12)
    ax.set_ylabel("Mean Tilt (deg) ± SEM", fontsize=12)
    ax.set_title("Mean MJO Tilt vs. Longitude by ENSO Phase (15° bins, N≥5)", 
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(linestyle='--', alpha=0.3)
    ax.set_xlim(40, 220)
    
    plt.tight_layout()
    
    out_path = FIG_DIR / "tilt_vs_longitude_binned_by_enso.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def analyze_regional_tilt(df: pd.DataFrame) -> None:
    """
    分析不同经度区域的 Tilt 差异（MC vs 非MC）
    """
    print("\nAnalyzing Regional Tilt Differences...")
    print("="*60)
    
    # 定义区域
    regions = {
        "Indian Ocean (40-100°E)": (40, 100),
        "Maritime Continent (100-160°E)": (100, 160),
        "Western Pacific (160-180°E)": (160, 180),
        "Central Pacific (180-220°E)": (180, 220),
    }
    
    results = []
    
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        
        print(f"\n{phase}:")
        for region_name, (lon_min, lon_max) in regions.items():
            region_mask = (subset["center_lon"] >= lon_min) & (subset["center_lon"] < lon_max)
            region_data = subset.loc[region_mask, "tilt"]
            
            if len(region_data) > 0:
                mean_tilt = region_data.mean()
                std_tilt = region_data.std()
                n = len(region_data)
                print(f"  {region_name}: N={n:4d}, Mean={mean_tilt:+6.2f}°, Std={std_tilt:5.2f}°")
                
                results.append({
                    "enso_phase": phase,
                    "region": region_name,
                    "n": n,
                    "mean_tilt": mean_tilt,
                    "std_tilt": std_tilt
                })
    
    # 保存结果
    df_results = pd.DataFrame(results)
    out_csv = FIG_DIR / "regional_tilt_statistics.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n  Saved statistics to: {out_csv}")
    
    # t检验：MC vs 非MC
    print("\n" + "="*60)
    print("T-test: Maritime Continent vs Outside")
    print("="*60)
    
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        mc_data = subset.loc[(subset["center_lon"] >= MC_WEST) & 
                             (subset["center_lon"] < MC_EAST), "tilt"]
        outside_data = subset.loc[(subset["center_lon"] < MC_WEST) | 
                                   (subset["center_lon"] >= MC_EAST), "tilt"]
        
        if len(mc_data) > 10 and len(outside_data) > 10:
            t, p = stats.ttest_ind(mc_data, outside_data, equal_var=False)
            sig = "*" if p < 0.05 else ("**" if p < 0.01 else "")
            print(f"  {phase}: MC (N={len(mc_data)}, mean={mc_data.mean():.2f}) vs "
                  f"Outside (N={len(outside_data)}, mean={outside_data.mean():.2f})")
            print(f"          t={t:.3f}, p={p:.4f} {sig}")
        else:
            print(f"  {phase}: Insufficient samples for t-test")


def plot_combined_analysis(df: pd.DataFrame) -> None:
    """
    绘制综合分析图（2x2 布局）
    """
    print("\nGenerating Combined Analysis Figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    
    # =====================
    # Panel 1: 经度分布 KDE
    # =====================
    ax1 = axes[0, 0]
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]["center_lon"]
        if len(subset) > 10:
            sns.kdeplot(subset, ax=ax1, label=f"{phase} (N={len(subset)})", 
                       color=ENSO_COLORS[phase], linewidth=2)
    
    ax1.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax1.axvline(MC_WEST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(MC_EAST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax1.axvline(WP_EAST, color='gray', linestyle=':', linewidth=1)
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Density")
    ax1.set_title("(a) Longitude Distribution", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_xlim(40, 220)
    
    # =====================
    # Panel 2: 经度分布箱线图
    # =====================
    ax2 = axes[0, 1]
    sns.boxplot(x="enso_phase", y="center_lon", data=df, order=ENSO_ORDER,
                hue="enso_phase", palette=ENSO_COLORS, ax=ax2, 
                width=0.5, legend=False)
    ax2.axhline(MC_WEST, color='green', linestyle='--', linewidth=1, alpha=0.7, 
                label=f'MC bounds ({MC_WEST}°-{MC_EAST}°E)')
    ax2.axhline(MC_EAST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax2.axhline(WP_EAST, color='gray', linestyle=':', linewidth=1, label='Dateline')
    ax2.set_xlabel("ENSO Phase")
    ax2.set_ylabel("Longitude (°E)")
    ax2.set_title("(b) Longitude Boxplot", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # =====================
    # Panel 3: Tilt vs Longitude 散点
    # =====================
    ax3 = axes[1, 0]
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        ax3.scatter(subset["center_lon"], subset["tilt"], 
                   c=ENSO_COLORS[phase], label=phase, alpha=0.4, s=10)
    
    ax3.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax3.axvline(MC_WEST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax3.axvline(MC_EAST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel("Longitude (°E)")
    ax3.set_ylabel("Daily Tilt (deg)")
    ax3.set_title("(c) Tilt vs. Longitude Scatter", fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(linestyle='--', alpha=0.3)
    ax3.set_xlim(40, 220)
    
    # =====================
    # Panel 4: 分箱统计
    # =====================
    ax4 = axes[1, 1]
    lon_bin_edges = np.arange(40, 221, 15)
    lon_bin_centers = (lon_bin_edges[:-1] + lon_bin_edges[1:]) / 2
    
    for phase in ENSO_ORDER:
        subset = df[df["enso_phase"] == phase]
        means = []
        sems = []
        
        for j in range(len(lon_bin_edges) - 1):
            bin_mask = (subset["center_lon"] >= lon_bin_edges[j]) & \
                       (subset["center_lon"] < lon_bin_edges[j+1])
            bin_data = subset.loc[bin_mask, "tilt"]
            
            if len(bin_data) >= 5:
                means.append(bin_data.mean())
                sems.append(bin_data.sem())
            else:
                means.append(np.nan)
                sems.append(np.nan)
        
        means = np.array(means)
        sems = np.array(sems)
        valid = ~np.isnan(means)
        
        ax4.plot(lon_bin_centers[valid], means[valid], 
                color=ENSO_COLORS[phase], linewidth=2, marker='o', 
                markersize=5, label=phase)
        ax4.fill_between(lon_bin_centers[valid], 
                        means[valid] - sems[valid], 
                        means[valid] + sems[valid],
                        color=ENSO_COLORS[phase], alpha=0.2)
    
    ax4.axvspan(MC_WEST, MC_EAST, alpha=0.1, color='green')
    ax4.axvline(MC_WEST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax4.axvline(MC_EAST, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax4.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax4.set_xlabel("Longitude (°E)")
    ax4.set_ylabel("Mean Tilt ± SEM (deg)")
    ax4.set_title("(d) Binned Mean Tilt", fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(linestyle='--', alpha=0.3)
    ax4.set_xlim(40, 220)
    
    plt.suptitle("Location Effect Analysis: MJO Tilt vs. Longitude by ENSO Phase", 
                fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    out_path = FIG_DIR / "location_effect_combined_analysis.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    print("="*60)
    print("Location Effect Analysis: MJO Tilt vs. Longitude")
    print("="*60)
    
    # 加载数据
    df_daily, enso_stats = load_daily_data()
    
    # 绘图
    plot_longitude_histogram(df_daily)
    plot_longitude_kde_overlay(df_daily)
    plot_tilt_vs_longitude_scatter(df_daily)
    plot_tilt_vs_longitude_binned(df_daily)
    plot_combined_analysis(df_daily)
    
    # 统计分析
    analyze_regional_tilt(df_daily)
    
    print("\n" + "="*60)
    print("All location effect analyses completed!")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
