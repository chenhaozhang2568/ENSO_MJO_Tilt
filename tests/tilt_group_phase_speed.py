# -*- coding: utf-8 -*-
"""
tilt_group_phase_speed.py: MJO 倾斜分组相速度对比分析

================================================================================
功能描述：
    本脚本按照垂直倾斜强度将 MJO 事件分为两组，对比分析其相速度差异。

分组定义：
    - STG（强倾斜组）: tilt > mean + 0.7×std
    - WTG（弱倾斜组）: tilt < mean - 0.7×std

主要分析内容：
    1. 两组相速度的 t-检验统计显著性
    2. 箱线图可视化分组差异
    3. 相速度分布直方图
    4. 输出分组统计结果

科学问题：
    倾斜更强的 MJO 事件是否传播更快或更慢？
    倾斜结构如何影响 MJO 的动力学行为？
- WTG (弱倾斜组): tilt < mean - 0.7 * std

运行:
cd /d E:\Projects\ENSO_MJO_Tilt
python tests\tilt_group_phase_speed.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ======================
# 路径配置
# ======================
SUMMARY_CSV = r"E:\Datas\Derived\phase_speed_tilt_summary.csv"

OUT_DIR = Path(r"E:\Datas\Derived")
OUT_CSV = OUT_DIR / "tilt_group_comparison.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = FIG_DIR / "tilt_group_phase_speed_comparison.png"

# ======================
# 分组参数
# ======================
TILT_THRESHOLD_SIGMA = 0.7  # 标准差倍数阈值


def main():
    print("=" * 60)
    print("MJO Tilt Groups Phase Speed Comparison")
    print("=" * 60)
    
    # ---- 1. 加载数据 ----
    print("\n[1] Loading data...")
    
    if not Path(SUMMARY_CSV).exists():
        raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}\nPlease run phase_speed_tilt_analysis.py first.")
    
    df = pd.read_csv(SUMMARY_CSV, parse_dates=["start_date", "end_date"])
    
    # 过滤有效数据
    df_valid = df.dropna(subset=["speed_m_s", "mean_tilt"]).copy()
    print(f"  Loaded {len(df_valid)} valid events")
    
    # ---- 2. 计算分组阈值 ----
    print("\n[2] Computing group thresholds...")
    
    tilt = df_valid["mean_tilt"].values
    tilt_mean = np.mean(tilt)
    tilt_std = np.std(tilt)
    
    threshold_high = tilt_mean + TILT_THRESHOLD_SIGMA * tilt_std
    threshold_low = tilt_mean - TILT_THRESHOLD_SIGMA * tilt_std
    
    print(f"  Tilt mean:  {tilt_mean:.2f}°")
    print(f"  Tilt std:   {tilt_std:.2f}°")
    print(f"  Threshold (±{TILT_THRESHOLD_SIGMA}σ):")
    print(f"    STG (Strong Tilt): tilt > {threshold_high:.2f}°")
    print(f"    WTG (Weak Tilt):   tilt < {threshold_low:.2f}°")
    
    # ---- 3. 分组 ----
    print("\n[3] Grouping events...")
    
    df_valid["group"] = "Normal"
    df_valid.loc[df_valid["mean_tilt"] > threshold_high, "group"] = "STG"
    df_valid.loc[df_valid["mean_tilt"] < threshold_low, "group"] = "WTG"
    
    stg = df_valid[df_valid["group"] == "STG"]
    wtg = df_valid[df_valid["group"] == "WTG"]
    normal = df_valid[df_valid["group"] == "Normal"]
    
    print(f"  STG (Strong Tilt Group): {len(stg)} events")
    print(f"  WTG (Weak Tilt Group):   {len(wtg)} events")
    print(f"  Normal (middle):         {len(normal)} events")
    
    # ---- 4. 计算各组相速度统计 ----
    print("\n[4] Phase speed statistics by group...")
    
    speed_stg = stg["speed_m_s"].values
    speed_wtg = wtg["speed_m_s"].values
    
    mean_stg = np.mean(speed_stg)
    std_stg = np.std(speed_stg, ddof=1)
    sem_stg = std_stg / np.sqrt(len(speed_stg))
    
    mean_wtg = np.mean(speed_wtg)
    std_wtg = np.std(speed_wtg, ddof=1)
    sem_wtg = std_wtg / np.sqrt(len(speed_wtg))
    
    print(f"\n  STG Phase Speed: {mean_stg:.3f} ± {std_stg:.3f} m/s (n={len(stg)})")
    print(f"  WTG Phase Speed: {mean_wtg:.3f} ± {std_wtg:.3f} m/s (n={len(wtg)})")
    print(f"  Difference (STG - WTG): {mean_stg - mean_wtg:.3f} m/s")
    
    # ---- 5. 独立样本t检验 ----
    print("\n[5] Independent samples t-test (STG vs WTG)...")
    
    if len(speed_stg) >= 2 and len(speed_wtg) >= 2:
        # Welch's t-test (不假设方差相等)
        t_stat, p_value = stats.ttest_ind(speed_stg, speed_wtg, equal_var=False)
        
        # 计算效应量 (Cohen's d)
        pooled_std = np.sqrt(((len(speed_stg)-1)*std_stg**2 + (len(speed_wtg)-1)*std_wtg**2) 
                            / (len(speed_stg) + len(speed_wtg) - 2))
        cohens_d = (mean_stg - mean_wtg) / pooled_std if pooled_std > 0 else np.nan
        
        print(f"\n  Welch's t-test:")
        print(f"    t-statistic: {t_stat:.4f}")
        print(f"    p-value:     {p_value:.4e}")
        print(f"    Cohen's d:   {cohens_d:.4f}")
        
        alpha = 0.05
        if p_value < alpha:
            print(f"\n  ✓ Significant difference at α = {alpha}")
        else:
            print(f"\n  ✗ No significant difference at α = {alpha}")
    else:
        t_stat, p_value, cohens_d = np.nan, np.nan, np.nan
        print("  Insufficient data for t-test")
    
    # ---- 6. 可视化 ----
    print("\n[6] Generating comparison plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：箱线图比较
    ax1 = axes[0]
    groups_data = [speed_wtg, speed_stg]
    group_labels = [f"WTG\n(n={len(wtg)})", f"STG\n(n={len(stg)})"]
    colors = ['#3498DB', '#E74C3C']
    
    bp = ax1.boxplot(groups_data, labels=group_labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 添加均值点
    ax1.scatter([1, 2], [mean_wtg, mean_stg], c='black', s=100, zorder=5, marker='D', label='Mean')
    
    ax1.set_ylabel("Phase Speed (m/s)", fontsize=12)
    ax1.set_title("Phase Speed by Tilt Group", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加统计标注
    if not np.isnan(p_value):
        sig_text = f"p = {p_value:.3f}" if p_value >= 0.001 else f"p = {p_value:.2e}"
        y_max = max(np.max(speed_wtg), np.max(speed_stg))
        ax1.plot([1, 1, 2, 2], [y_max + 0.1, y_max + 0.2, y_max + 0.2, y_max + 0.1], 'k-', lw=1)
        ax1.text(1.5, y_max + 0.25, sig_text, ha='center', fontsize=10)
    
    # 右图：散点图（带分组颜色）
    ax2 = axes[1]
    
    # 分别绘制各组
    for grp, color, label in [("WTG", '#3498DB', 'Weak Tilt (WTG)'),
                               ("Normal", '#95A5A6', 'Normal'),
                               ("STG", '#E74C3C', 'Strong Tilt (STG)')]:
        grp_data = df_valid[df_valid["group"] == grp]
        ax2.scatter(grp_data["speed_m_s"], grp_data["mean_tilt"], 
                   c=color, s=60, alpha=0.7, label=label, edgecolors='white', linewidth=0.5)
    
    # 阈值线
    ax2.axhline(y=threshold_high, color='#E74C3C', linestyle='--', alpha=0.7, 
                label=f'STG threshold ({threshold_high:.1f}°)')
    ax2.axhline(y=threshold_low, color='#3498DB', linestyle='--', alpha=0.7,
                label=f'WTG threshold ({threshold_low:.1f}°)')
    ax2.axhline(y=tilt_mean, color='gray', linestyle='-', alpha=0.5)
    
    ax2.set_xlabel("Phase Speed (m/s)", fontsize=12)
    ax2.set_ylabel("Mean Tilt (°)", fontsize=12)
    ax2.set_title("MJO Events by Tilt Group", fontsize=14)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f"  Saved figure: {OUT_FIG}")
    plt.close()
    
    # ---- 7. 保存结果 ----
    print("\n[7] Saving results...")
    
    # 汇总统计表
    summary_data = {
        "Group": ["STG (Strong Tilt)", "WTG (Weak Tilt)", "All Events"],
        "N": [len(stg), len(wtg), len(df_valid)],
        "Tilt_Mean": [stg["mean_tilt"].mean(), wtg["mean_tilt"].mean(), tilt_mean],
        "Tilt_Std": [stg["mean_tilt"].std(), wtg["mean_tilt"].std(), tilt_std],
        "Speed_Mean": [mean_stg, mean_wtg, df_valid["speed_m_s"].mean()],
        "Speed_Std": [std_stg, std_wtg, df_valid["speed_m_s"].std()],
        "Speed_SEM": [sem_stg, sem_wtg, df_valid["speed_m_s"].std() / np.sqrt(len(df_valid))],
    }
    df_summary = pd.DataFrame(summary_data)
    
    # 添加t检验结果
    df_summary["t_statistic"] = t_stat
    df_summary["p_value"] = p_value
    df_summary["cohens_d"] = cohens_d
    df_summary["threshold_high"] = threshold_high
    df_summary["threshold_low"] = threshold_low
    
    df_summary.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  Saved summary: {OUT_CSV}")
    
    # 保存分组后的完整数据
    out_full = OUT_DIR / "phase_speed_tilt_with_groups.csv"
    df_valid.to_csv(out_full, index=False, encoding="utf-8-sig")
    print(f"  Saved full data: {out_full}")
    
    # ---- 8. 结果汇总 ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nGroup Definitions (threshold = ±{TILT_THRESHOLD_SIGMA}σ):")
    print(f"  STG: tilt > {threshold_high:.2f}° ({len(stg)} events)")
    print(f"  WTG: tilt < {threshold_low:.2f}° ({len(wtg)} events)")
    print(f"\nPhase Speed Comparison:")
    print(f"  STG: {mean_stg:.3f} ± {std_stg:.3f} m/s")
    print(f"  WTG: {mean_wtg:.3f} ± {std_wtg:.3f} m/s")
    print(f"  Difference: {mean_stg - mean_wtg:.3f} m/s")
    print(f"\nWelch's t-test:")
    print(f"  t = {t_stat:.4f}, p = {p_value:.4e}")
    print(f"  Cohen's d = {cohens_d:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
