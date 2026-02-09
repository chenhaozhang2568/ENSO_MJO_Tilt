# -*- coding: utf-8 -*-
"""
enso_tilt_mechanism_analysis.py: ENSO-Tilt 差异物理机制分析

分析内容:
1. 边界宽度对比（高层/低层）
2. omega 峰值强度对比
3. 西边界 vs 东边界分离分析
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
ONI_FILE = r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt"
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\mechanism_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ONI_THRESHOLD = 0.5


def load_oni():
    oni = pd.read_csv(ONI_FILE, sep=r'\s+', header=0, engine='python')
    month_map = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
                'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}
    records = []
    for _, row in oni.iterrows():
        seas = row['SEAS']
        year = int(row['YR'])
        anom = row['ANOM']
        if seas in month_map:
            month = month_map[seas]
            records.append({'year': year, 'month': month, 'oni': anom})
    oni_df = pd.DataFrame(records)
    oni_df['date'] = pd.to_datetime(oni_df[['year', 'month']].assign(day=1))
    return oni_df.set_index('date')['oni']


def classify_enso(event_center_date, oni_series):
    year = event_center_date.year
    month = event_center_date.month
    target = pd.Timestamp(year=year, month=month, day=1)
    if target in oni_series.index:
        oni_val = oni_series.loc[target]
    else:
        idx = oni_series.index.get_indexer([target], method='nearest')[0]
        if idx >= 0 and idx < len(oni_series):
            oni_val = oni_series.iloc[idx]
        else:
            return None
    if oni_val >= ONI_THRESHOLD:
        return 'El Nino'
    elif oni_val <= -ONI_THRESHOLD:
        return 'La Nina'
    else:
        return 'Neutral'


def main():
    print("="*70)
    print("ENSO-Tilt 差异物理机制分析")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1] Loading data...")
    ds = xr.open_dataset(TILT_NC, engine="netcdf4")
    time = pd.to_datetime(ds["time"].values)
    
    # 提取所有需要的变量
    tilt = ds["tilt"].values
    low_west = ds["low_west_rel"].values
    low_east = ds["low_east_rel"].values
    up_west = ds["up_west_rel"].values
    up_east = ds["up_east_rel"].values
    low_wmin = ds["low_wmin"].values  # 低层最强上升
    up_wmin = ds["up_wmin"].values    # 高层最强上升
    
    events_df = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    oni_series = load_oni()
    print(f"  Events: {len(events_df)}")
    
    # 2. 计算事件级统计
    print("\n[2] Computing event-level statistics...")
    
    event_stats = []
    for _, ev in events_df.iterrows():
        start = pd.Timestamp(ev['start_date'])
        end = pd.Timestamp(ev['end_date'])
        center = start + (end - start) / 2
        mask = (time >= start) & (time <= end)
        
        # 提取该事件的数据
        tilt_ev = tilt[mask]
        lw_ev = low_west[mask]
        le_ev = low_east[mask]
        uw_ev = up_west[mask]
        ue_ev = up_east[mask]
        low_wmin_ev = low_wmin[mask]
        up_wmin_ev = up_wmin[mask]
        
        # 有效数据
        valid_idx = np.isfinite(tilt_ev)
        if valid_idx.sum() == 0:
            continue
        
        enso = classify_enso(center, oni_series)
        if enso is None:
            continue
        
        # 计算边界宽度
        low_width = le_ev[valid_idx] - lw_ev[valid_idx]
        up_width = ue_ev[valid_idx] - uw_ev[valid_idx]
        
        event_stats.append({
            'event_id': ev['event_id'],
            'enso_phase': enso,
            'mean_tilt': np.nanmean(tilt_ev[valid_idx]),
            # 边界位置
            'mean_low_west': np.nanmean(lw_ev[valid_idx]),
            'mean_low_east': np.nanmean(le_ev[valid_idx]),
            'mean_up_west': np.nanmean(uw_ev[valid_idx]),
            'mean_up_east': np.nanmean(ue_ev[valid_idx]),
            # 边界宽度
            'mean_low_width': np.nanmean(low_width),
            'mean_up_width': np.nanmean(up_width),
            # omega 峰值强度（注意是负值，更负=更强上升）
            'mean_low_wmin': np.nanmean(low_wmin_ev[valid_idx]),
            'mean_up_wmin': np.nanmean(up_wmin_ev[valid_idx]),
        })
    
    df = pd.DataFrame(event_stats)
    print(f"  Valid events: {len(df)}")
    
    # 3. 分组统计
    print("\n" + "="*70)
    print("分析结果")
    print("="*70)
    
    en = df[df['enso_phase'] == 'El Nino']
    ln = df[df['enso_phase'] == 'La Nina']
    neu = df[df['enso_phase'] == 'Neutral']
    
    # -------- 分析 1: 边界宽度 --------
    print("\n【分析 1】边界宽度对比")
    print("-"*50)
    print(f"{'Phase':<12} {'低层宽度':>12} {'高层宽度':>12} {'宽度差':>10}")
    print("-"*50)
    for phase, grp in [('El Nino', en), ('La Nina', ln), ('Neutral', neu)]:
        lw = grp['mean_low_width'].mean()
        uw = grp['mean_up_width'].mean()
        print(f"{phase:<12} {lw:>12.2f}° {uw:>12.2f}° {uw-lw:>+10.2f}°")
    
    print("\n高层宽度 T-test:")
    for name, (g1, g2) in [("El Nino vs La Nina", (en, ln)),
                           ("El Nino vs Neutral", (en, neu)),
                           ("La Nina vs Neutral", (ln, neu))]:
        t, p = stats.ttest_ind(g1['mean_up_width'], g2['mean_up_width'], equal_var=False)
        diff = g1['mean_up_width'].mean() - g2['mean_up_width'].mean()
        sig = "✓" if p < 0.05 else ""
        print(f"  {name:<22}: diff={diff:+.2f}°, p={p:.4f} {sig}")
    
    # -------- 分析 2: omega 峰值强度 --------
    print("\n【分析 2】Omega 峰值强度对比 (更负=对流更强)")
    print("-"*50)
    print(f"{'Phase':<12} {'低层ω_min':>14} {'高层ω_min':>14}")
    print("-"*50)
    for phase, grp in [('El Nino', en), ('La Nina', ln), ('Neutral', neu)]:
        lw = grp['mean_low_wmin'].mean()
        uw = grp['mean_up_wmin'].mean()
        print(f"{phase:<12} {lw:>14.4f} {uw:>14.4f}")
    
    print("\n高层 ω_min T-test:")
    for name, (g1, g2) in [("El Nino vs La Nina", (en, ln)),
                           ("El Nino vs Neutral", (en, neu)),
                           ("La Nina vs Neutral", (ln, neu))]:
        t, p = stats.ttest_ind(g1['mean_up_wmin'], g2['mean_up_wmin'], equal_var=False)
        diff = g1['mean_up_wmin'].mean() - g2['mean_up_wmin'].mean()
        sig = "✓" if p < 0.05 else ""
        print(f"  {name:<22}: diff={diff:+.4f}, p={p:.4f} {sig}")
    
    # -------- 分析 3: 西边界 vs 东边界 --------
    print("\n【分析 3】西边界 vs 东边界分离分析")
    print("-"*50)
    print("高层西边界:")
    for phase, grp in [('El Nino', en), ('La Nina', ln), ('Neutral', neu)]:
        val = grp['mean_up_west'].mean()
        print(f"  {phase:<12}: {val:+.2f}°")
    
    print("\n高层东边界:")
    for phase, grp in [('El Nino', en), ('La Nina', ln), ('Neutral', neu)]:
        val = grp['mean_up_east'].mean()
        print(f"  {phase:<12}: {val:+.2f}°")
    
    print("\n高层西边界 T-test (La Nina 是否更西?):")
    t, p = stats.ttest_ind(ln['mean_up_west'], en['mean_up_west'], equal_var=False)
    print(f"  La Nina vs El Nino: diff={ln['mean_up_west'].mean() - en['mean_up_west'].mean():+.2f}°, p={p:.4f}")
    
    t, p = stats.ttest_ind(ln['mean_up_west'], neu['mean_up_west'], equal_var=False)
    print(f"  La Nina vs Neutral: diff={ln['mean_up_west'].mean() - neu['mean_up_west'].mean():+.2f}°, p={p:.4f}")
    
    # -------- 可视化 --------
    print("\n[3] Generating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    colors = {'El Nino': '#E74C3C', 'La Nina': '#3498DB', 'Neutral': '#95A5A6'}
    order = ['El Nino', 'La Nina', 'Neutral']
    
    # 图1: 高层宽度箱线图
    ax1 = axes[0, 0]
    data1 = [df[df['enso_phase'] == p]['mean_up_width'].values for p in order]
    bp1 = ax1.boxplot(data1, patch_artist=True, widths=0.6)
    for patch, phase in zip(bp1['boxes'], order):
        patch.set_facecolor(colors[phase])
        patch.set_alpha(0.7)
    ax1.set_xticklabels(order)
    ax1.set_ylabel('高层边界宽度 (°)', fontsize=11)
    ax1.set_title('高层 (400-200 hPa) 边界宽度', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2: 高层 omega 峰值箱线图
    ax2 = axes[0, 1]
    data2 = [df[df['enso_phase'] == p]['mean_up_wmin'].values for p in order]
    bp2 = ax2.boxplot(data2, patch_artist=True, widths=0.6)
    for patch, phase in zip(bp2['boxes'], order):
        patch.set_facecolor(colors[phase])
        patch.set_alpha(0.7)
    ax2.set_xticklabels(order)
    ax2.set_ylabel('高层 ω_min (Pa/s)', fontsize=11)
    ax2.set_title('高层 Omega 峰值强度 (更负=更强)', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 图3: 西边界 vs 东边界散点图
    ax3 = axes[1, 0]
    for phase in order:
        grp = df[df['enso_phase'] == phase]
        ax3.scatter(grp['mean_up_west'], grp['mean_up_east'], 
                   c=colors[phase], s=60, alpha=0.7, label=phase, edgecolors='white')
    ax3.set_xlabel('高层西边界 (°)', fontsize=11)
    ax3.set_ylabel('高层东边界 (°)', fontsize=11)
    ax3.set_title('高层西边界 vs 东边界', fontsize=13, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 宽度 vs Tilt 散点图
    ax4 = axes[1, 1]
    for phase in order:
        grp = df[df['enso_phase'] == phase]
        ax4.scatter(grp['mean_up_width'], grp['mean_tilt'], 
                   c=colors[phase], s=60, alpha=0.7, label=phase, edgecolors='white')
    ax4.set_xlabel('高层边界宽度 (°)', fontsize=11)
    ax4.set_ylabel('Tilt (°)', fontsize=11)
    ax4.set_title('高层宽度 vs Tilt', fontsize=13, fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = OUT_DIR / "enso_tilt_mechanism.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()
    
    # 保存数据
    df.to_csv(OUT_DIR / "enso_tilt_mechanism_data.csv", index=False)
    
    # -------- 结论 --------
    print("\n" + "="*70)
    print("结论")
    print("="*70)
    
    # 检查假设
    ln_width = ln['mean_up_width'].mean()
    en_width = en['mean_up_width'].mean()
    neu_width = neu['mean_up_width'].mean()
    
    print(f"\n假设验证: La Nina 高层边界宽度是否更大?")
    print(f"  La Nina 高层宽度:  {ln_width:.2f}°")
    print(f"  El Nino 高层宽度:  {en_width:.2f}°")
    print(f"  Neutral 高层宽度:  {neu_width:.2f}°")
    
    if ln_width > en_width and ln_width > neu_width:
        print(f"\n  ✓ La Nina 高层边界宽度最大，支持暖池扩大假设")
    else:
        max_phase = max([('La Nina', ln_width), ('El Nino', en_width), ('Neutral', neu_width)], key=lambda x: x[1])
        print(f"\n  ✗ {max_phase[0]} 高层边界宽度最大 ({max_phase[1]:.2f}°)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
