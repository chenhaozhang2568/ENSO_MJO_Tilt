# -*- coding: utf-8 -*-
"""
boundary_position_by_enso.py — ENSO 分组边界位置分析

功能：
    按 ENSO 相位分组，检验低层/高层西边界位置差异，
    验证 La Niña 期间高层西边界是否更偏西的假设。
输入：
    tilt_daily_step4_layermean_1979-2022.nc, mjo_events_step3_1979-2022.csv,
    oni.ascii.txt
输出：
    figures/boundary_analysis/boundary_position_by_enso.png
用法：
    python tests/boundary_position_by_enso.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
TILT_NC = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\boundary")
OUT_DIR.mkdir(parents=True, exist_ok=True)




def main():
    print("="*60)
    print("ENSO 分组边界位置分析")
    print("="*60)
    
    # 1. 加载数据
    print("\n[1] Loading data...")
    
    ds = xr.open_dataset(TILT_NC, engine="netcdf4")
    time = pd.to_datetime(ds["time"].values)
    tilt = ds["tilt"].values
    low_west = ds["low_west_rel"].values
    up_west = ds["up_west_rel"].values
    
    # 加载事件 + ENSO 分类
    events_df = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    enso_stats = pd.read_csv(ENSO_STATS_CSV)
    enso_map = dict(zip(enso_stats['event_id'], enso_stats['enso_phase']))
    
    print(f"  Tilt data: {len(time)} days")
    print(f"  Events: {len(events_df)}")
    
    # 2. 计算每个事件的平均边界
    print("\n[2] Computing event-level boundary statistics...")
    
    event_stats = []
    for _, ev in events_df.iterrows():
        eid = ev['event_id']
        phase = enso_map.get(eid)
        if phase is None:
            continue
        
        start = pd.Timestamp(ev['start_date'])
        end = pd.Timestamp(ev['end_date'])
        mask = (time >= start) & (time <= end)
        
        tilt_ev = tilt[mask]
        lw_ev = low_west[mask]
        uw_ev = up_west[mask]
        
        valid_tilt = tilt_ev[np.isfinite(tilt_ev)]
        valid_lw = lw_ev[np.isfinite(lw_ev)]
        valid_uw = uw_ev[np.isfinite(uw_ev)]
        
        if len(valid_tilt) > 0:
            event_stats.append({
                'event_id': eid,
                'start_date': start,
                'end_date': end,
                'center_date': start + (end - start) / 2,
                'enso_phase': phase,
                'mean_tilt': np.mean(valid_tilt),
                'mean_low_west': np.mean(valid_lw) if len(valid_lw) > 0 else np.nan,
                'mean_up_west': np.mean(valid_uw) if len(valid_uw) > 0 else np.nan,
                'n_days': len(valid_tilt)
            })
    
    df = pd.DataFrame(event_stats)
    df = df.dropna(subset=['enso_phase', 'mean_low_west', 'mean_up_west'])
    print(f"  Valid events: {len(df)}")
    
    # 3. 分组统计
    print("\n[3] ENSO group statistics...")
    print("\n" + "="*70)
    print(f"{'Phase':<12} {'N':>4} {'Low_West':>12} {'Up_West':>12} {'Tilt':>10}")
    print("="*70)
    
    for phase in ['El Nino', 'La Nina', 'Neutral']:
        grp = df[df['enso_phase'] == phase]
        if len(grp) > 0:
            lw = grp['mean_low_west'].mean()
            uw = grp['mean_up_west'].mean()
            tlt = grp['mean_tilt'].mean()
            print(f"{phase:<12} {len(grp):>4} {lw:>+12.2f}° {uw:>+12.2f}° {tlt:>+10.2f}°")
    print("="*70)
    
    # 4. 统计检验
    print("\n[4] Statistical tests (T-test)...")
    
    en = df[df['enso_phase'] == 'El Nino']
    ln = df[df['enso_phase'] == 'La Nina']
    neu = df[df['enso_phase'] == 'Neutral']
    
    print("\n--- Low-level West Boundary ---")
    for name, (g1, g2) in [("El Nino vs La Nina", (en, ln)),
                           ("El Nino vs Neutral", (en, neu)),
                           ("La Nina vs Neutral", (ln, neu))]:
        t, p = stats.ttest_ind(g1['mean_low_west'], g2['mean_low_west'], equal_var=False)
        diff = g1['mean_low_west'].mean() - g2['mean_low_west'].mean()
        sig = "✓" if p < 0.05 else ""
        print(f"  {name:<22}: diff={diff:+.2f}°, p={p:.4f} {sig}")
    
    print("\n--- Upper-level West Boundary ---")
    for name, (g1, g2) in [("El Nino vs La Nina", (en, ln)),
                           ("El Nino vs Neutral", (en, neu)),
                           ("La Nina vs Neutral", (ln, neu))]:
        t, p = stats.ttest_ind(g1['mean_up_west'], g2['mean_up_west'], equal_var=False)
        diff = g1['mean_up_west'].mean() - g2['mean_up_west'].mean()
        sig = "✓" if p < 0.05 else ""
        print(f"  {name:<22}: diff={diff:+.2f}°, p={p:.4f} {sig}")
    
    print("\n--- Tilt (Low_West - Up_West) ---")
    for name, (g1, g2) in [("El Nino vs La Nina", (en, ln)),
                           ("El Nino vs Neutral", (en, neu)),
                           ("La Nina vs Neutral", (ln, neu))]:
        t, p = stats.ttest_ind(g1['mean_tilt'], g2['mean_tilt'], equal_var=False)
        diff = g1['mean_tilt'].mean() - g2['mean_tilt'].mean()
        sig = "✓" if p < 0.05 else ""
        print(f"  {name:<22}: diff={diff:+.2f}°, p={p:.4f} {sig}")
    
    # 5. 可视化
    print("\n[5] Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {'El Nino': '#E74C3C', 'La Nina': '#3498DB', 'Neutral': '#95A5A6'}
    order = ['El Nino', 'La Nina', 'Neutral']
    
    # 图1: 低层西边界箱线图
    ax1 = axes[0, 0]
    data1 = [df[df['enso_phase'] == p]['mean_low_west'].values for p in order]
    bp1 = ax1.boxplot(data1, patch_artist=True, widths=0.6)
    for patch, phase in zip(bp1['boxes'], order):
        patch.set_facecolor(colors[phase])
        patch.set_alpha(0.7)
    ax1.set_xticklabels(order)
    ax1.set_ylabel('低层西边界位置 (相对经度°)', fontsize=11)
    ax1.set_title('低层 (1000-600 hPa) 西边界', fontsize=13, fontweight='bold')
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: 高层西边界箱线图
    ax2 = axes[0, 1]
    data2 = [df[df['enso_phase'] == p]['mean_up_west'].values for p in order]
    bp2 = ax2.boxplot(data2, patch_artist=True, widths=0.6)
    for patch, phase in zip(bp2['boxes'], order):
        patch.set_facecolor(colors[phase])
        patch.set_alpha(0.7)
    ax2.set_xticklabels(order)
    ax2.set_ylabel('高层西边界位置 (相对经度°)', fontsize=11)
    ax2.set_title('高层 (400-200 hPa) 西边界', fontsize=13, fontweight='bold')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # 图3: Tilt 箱线图
    ax3 = axes[1, 0]
    data3 = [df[df['enso_phase'] == p]['mean_tilt'].values for p in order]
    bp3 = ax3.boxplot(data3, patch_artist=True, widths=0.6)
    for patch, phase in zip(bp3['boxes'], order):
        patch.set_facecolor(colors[phase])
        patch.set_alpha(0.7)
    ax3.set_xticklabels(order)
    ax3.set_ylabel('Tilt (°)', fontsize=11)
    ax3.set_title('Tilt = 低层西边界 - 高层西边界', fontsize=13, fontweight='bold')
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.grid(axis='y', alpha=0.3)
    
    # 图4: 散点图 - 低层 vs 高层
    ax4 = axes[1, 1]
    for phase in order:
        grp = df[df['enso_phase'] == phase]
        ax4.scatter(grp['mean_up_west'], grp['mean_low_west'], 
                   c=colors[phase], s=60, alpha=0.7, label=phase, edgecolors='white')
    
    # 1:1 线
    lims = [min(ax4.get_xlim()[0], ax4.get_ylim()[0]),
            max(ax4.get_xlim()[1], ax4.get_ylim()[1])]
    ax4.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
    
    ax4.set_xlabel('高层西边界 (°)', fontsize=11)
    ax4.set_ylabel('低层西边界 (°)', fontsize=11)
    ax4.set_title('低层 vs 高层西边界位置', fontsize=13, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = OUT_DIR / "boundary_position_by_enso.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()
    
    # 保存数据
    csv_path = OUT_DIR / "boundary_position_by_enso.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # 6. 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\n各 ENSO 相的边界位置均值：")
    for phase in order:
        grp = df[df['enso_phase'] == phase]
        print(f"  {phase}:")
        print(f"    低层西边界: {grp['mean_low_west'].mean():+.2f}° (偏西为负)")
        print(f"    高层西边界: {grp['mean_up_west'].mean():+.2f}° (偏西为负)")
        print(f"    Tilt:      {grp['mean_tilt'].mean():+.2f}°")
    
    print("\n假设检验：La Nina 高层是否推得更远（更负）？")
    ln_uw = ln['mean_up_west'].mean()
    en_uw = en['mean_up_west'].mean()
    neu_uw = neu['mean_up_west'].mean()
    print(f"  El Nino 高层西边界: {en_uw:+.2f}°")
    print(f"  La Nina 高层西边界: {ln_uw:+.2f}°")
    print(f"  Neutral 高层西边界: {neu_uw:+.2f}°")
    if ln_uw < en_uw:
        print(f"  → La Nina 高层比 El Nino 更西 {en_uw - ln_uw:.2f}°")
    else:
        print(f"  → La Nina 高层比 El Nino 更东 {ln_uw - en_uw:.2f}°")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
