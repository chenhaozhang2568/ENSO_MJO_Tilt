# -*- coding: utf-8 -*-
"""
enso_tilt_both_sides.py — 东西两侧 Tilt ENSO 分组对比

功能：
    分别计算 MJO 上升区域西侧和东侧的 Tilt，
    比较不同 ENSO 相位的差异及两侧的对称性。
输入：
    tilt_daily_step4_layermean_1979-2022.nc, mjo_events_step3_1979-2022.csv,
    oni.ascii.txt
输出：
    figures/both_sides/tilt_both_sides_comparison.png
用法：
    python tests/enso_tilt_both_sides.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
TILT_NC = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\both_sides")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENSO_ORDER = ["El Nino", "La Nina", "Neutral"]
ENSO_COLORS = {"El Nino": "#E74C3C", "La Nina": "#3498DB", "Neutral": "#95A5A6"}


def main():
    print("="*70)
    print("东西两侧 Tilt 对比分析")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1] Loading data...")
    ds = xr.open_dataset(TILT_NC, engine="netcdf4")
    time = pd.to_datetime(ds["time"].values)
    
    tilt_west = ds["tilt"].values       # 西侧 tilt
    tilt_east = ds["tilt_east"].values  # 东侧 tilt
    
    events_df = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    enso_stats = pd.read_csv(ENSO_STATS_CSV)
    enso_map = dict(zip(enso_stats['event_id'], enso_stats['enso_phase']))
    print(f"  Events: {len(events_df)}")
    
    # 2. 计算事件级统计
    print("\n[2] Computing event-level statistics...")
    
    event_stats = []
    for _, ev in events_df.iterrows():
        eid = ev['event_id']
        phase = enso_map.get(eid)
        if phase is None:
            continue
        
        start = pd.Timestamp(ev['start_date'])
        end = pd.Timestamp(ev['end_date'])
        mask = (time >= start) & (time <= end)
        
        tw = tilt_west[mask]
        te = tilt_east[mask]
        
        valid_w = np.isfinite(tw)
        valid_e = np.isfinite(te)
        
        if valid_w.sum() == 0 or valid_e.sum() == 0:
            continue
        
        event_stats.append({
            'event_id': eid,
            'enso_phase': phase,
            'mean_tilt_west': np.nanmean(tw[valid_w]),
            'mean_tilt_east': np.nanmean(te[valid_e]),
        })
    
    df = pd.DataFrame(event_stats)
    print(f"  Valid events: {len(df)}")
    
    # 3. 分组统计
    print("\n" + "="*70)
    print("分析结果")
    print("="*70)
    
    print("\n【西侧 Tilt vs 东侧 Tilt 对比】")
    print("-"*60)
    print(f"{'Phase':<12} {'西侧Tilt':>12} {'东侧Tilt':>12} {'差值(西-东)':>14}")
    print("-"*60)
    
    for phase in ENSO_ORDER:
        grp = df[df['enso_phase'] == phase]
        tw = grp['mean_tilt_west'].mean()
        te = grp['mean_tilt_east'].mean()
        diff = tw - te
        print(f"{phase:<12} {tw:>12.2f}° {te:>12.2f}° {diff:>+14.2f}°")
    
    # 4. 统计检验
    print("\n【ENSO 分组 T-test】")
    print("-"*60)
    
    en = df[df['enso_phase'] == 'El Nino']
    ln = df[df['enso_phase'] == 'La Nina']
    neu = df[df['enso_phase'] == 'Neutral']
    
    print("\n西侧 Tilt:")
    for name, (g1, g2) in [("El Nino vs La Nina", (en, ln)),
                           ("La Nina vs Neutral", (ln, neu))]:
        t, p = stats.ttest_ind(g1['mean_tilt_west'], g2['mean_tilt_west'], equal_var=False)
        diff = g1['mean_tilt_west'].mean() - g2['mean_tilt_west'].mean()
        sig = "✓" if p < 0.05 else ""
        print(f"  {name:<22}: diff={diff:+.2f}°, p={p:.4f} {sig}")
    
    print("\n东侧 Tilt:")
    for name, (g1, g2) in [("El Nino vs La Nina", (en, ln)),
                           ("La Nina vs Neutral", (ln, neu))]:
        t, p = stats.ttest_ind(g1['mean_tilt_east'], g2['mean_tilt_east'], equal_var=False)
        diff = g1['mean_tilt_east'].mean() - g2['mean_tilt_east'].mean()
        sig = "✓" if p < 0.05 else ""
        print(f"  {name:<22}: diff={diff:+.2f}°, p={p:.4f} {sig}")
    
    # 5. 可视化
    print("\n[3] Generating plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 图1: 西侧 Tilt 箱线图
    ax1 = axes[0]
    data1 = [df[df['enso_phase'] == p]['mean_tilt_west'].values for p in ENSO_ORDER]
    bp1 = ax1.boxplot(data1, patch_artist=True, widths=0.6)
    for patch, phase in zip(bp1['boxes'], ENSO_ORDER):
        patch.set_facecolor(ENSO_COLORS[phase])
        patch.set_alpha(0.7)
    ax1.set_xticklabels(ENSO_ORDER)
    ax1.set_ylabel('Tilt (°)', fontsize=11)
    ax1.set_title('西侧 Tilt (后侧)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: 东侧 Tilt 箱线图
    ax2 = axes[1]
    data2 = [df[df['enso_phase'] == p]['mean_tilt_east'].values for p in ENSO_ORDER]
    bp2 = ax2.boxplot(data2, patch_artist=True, widths=0.6)
    for patch, phase in zip(bp2['boxes'], ENSO_ORDER):
        patch.set_facecolor(ENSO_COLORS[phase])
        patch.set_alpha(0.7)
    ax2.set_xticklabels(ENSO_ORDER)
    ax2.set_ylabel('Tilt (°)', fontsize=11)
    ax2.set_title('东侧 Tilt (前侧)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 图3: 西侧 vs 东侧 散点图
    ax3 = axes[2]
    for phase in ENSO_ORDER:
        grp = df[df['enso_phase'] == phase]
        ax3.scatter(grp['mean_tilt_west'], grp['mean_tilt_east'], 
                   c=ENSO_COLORS[phase], s=60, alpha=0.7, label=phase, edgecolors='white')
    
    # 添加对角线
    lims = [min(ax3.get_xlim()[0], ax3.get_ylim()[0]),
            max(ax3.get_xlim()[1], ax3.get_ylim()[1])]
    ax3.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
    
    ax3.set_xlabel('西侧 Tilt (°)', fontsize=11)
    ax3.set_ylabel('东侧 Tilt (°)', fontsize=11)
    ax3.set_title('西侧 vs 东侧 Tilt', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = OUT_DIR / "tilt_both_sides_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()
    
    # 保存数据
    df.to_csv(OUT_DIR / "tilt_both_sides_data.csv", index=False)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
