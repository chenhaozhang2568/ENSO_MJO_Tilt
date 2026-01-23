# -*- coding: utf-8 -*-
"""
test5.py: 分析高层边界 (Upper-layer boundaries) 的分布及其长尾效应。
用于诊断为什么 Tilt 指数会出现极大值。
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置
TILT_NC = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\checks_step4")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def analyze_tails():
    print(f"Loading data from {TILT_NC}...")
    ds = xr.open_dataset(TILT_NC)
    
    # 提取变量并去除无效值
    df = ds[["up_west_rel", "up_east_rel", "low_west_rel", "low_east_rel", "tilt"]].to_dataframe().dropna()
    
    print(f"Total valid samples: {len(df)}")
    
    # --- 图 1: 上层西部 vs 下层西部的对比 ---
    plt.figure(figsize=(10, 6), dpi=150)
    sns.histplot(df["low_west_rel"], bins=50, kde=True, color="blue", label="Lower West")
    sns.histplot(df["up_west_rel"], bins=50, kde=True, color="red", label="Upper West")
    plt.title("West Boundary Comparison: Lower vs Upper")
    plt.xlabel("Relative Longitude (deg)")
    plt.legend()
    out_fig1 = FIG_DIR / "west_boundary_comparison_step4_1979-2022.png"
    plt.savefig(out_fig1, bbox_inches='tight')
    print(f"Figure saved to: {out_fig1}")
    plt.close()

    # --- 图 2: 散点图：上层西部位置 vs Tilt 值 ---
    plt.figure(figsize=(10, 6), dpi=150)
    sns.scatterplot(data=df, x="up_west_rel", y="tilt", alpha=0.4, s=15, color="purple")
    plt.title("Correlation: Upper West vs Tilt Index")
    plt.xlabel("Upper West Relative Longitude (deg)")
    plt.ylabel("Daily Tilt Index (deg)")
    out_fig2 = FIG_DIR / "upper_west_vs_tilt_scatter_step4_1979-2022.png"
    plt.savefig(out_fig2, bbox_inches='tight')
    print(f"Figure saved to: {out_fig2}")
    plt.close()

    # --- 图 3: 箱线图查看异常值 (长尾) ---
    plt.figure(figsize=(10, 6), dpi=150)
    melted_df = df.melt(value_vars=["up_west_rel", "low_west_rel", "tilt"], var_name="Variable", value_name="Value")
    sns.boxplot(data=melted_df, x="Variable", y="Value", palette="vlag")
    plt.title("Outliers (Tail Effects) Visualization")
    out_fig3 = FIG_DIR / "outlier_boxplots_step4_1979-2022.png"
    plt.savefig(out_fig3, bbox_inches='tight')
    print(f"Figure saved to: {out_fig3}")
    plt.close()
    
    # 打印一些统计数据
    print("\nDescriptive Statistics for West Boundaries:")
    stats = df[["up_west_rel", "low_west_rel", "tilt"]].describe(percentiles=[.01, .05, .5, .95, .99])
    print(stats)
    
    # 统计长尾
    tail_99 = stats.loc["1%", "up_west_rel"]
    print(f"\nPotential Long Tail Threshold (Upper West 1%): {tail_99:.2f} deg")

if __name__ == "__main__":
    analyze_tails()
