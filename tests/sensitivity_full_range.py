# -*- coding: utf-8 -*-
"""
sensitivity_full_range.py: 完整范围敏感性分析（西到东）

阈值从 0%→100%→0% 对应 西边界→核心→东边界
输出 8 张独立的图
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
W_NORM_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ONI_FILE = r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt"
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\sensitivity_full")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
THRESHOLDS_HALF = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 
                   0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
ONI_THRESHOLD = 0.5
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
LOW_LAYER = (1000.0, 600.0)
UP_LAYER = (400.0, 200.0)
PIVOT_DELTA_DEG = 10.0
MIN_VALID_POINTS = 7

ENSO_ORDER = ["El Nino", "La Nina", "Neutral"]
ENSO_COLORS = {"El Nino": "#E74C3C", "La Nina": "#3498DB", "Neutral": "#95A5A6"}


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
            records.append({'year': year, 'month': month_map[seas], 'oni': anom})
    oni_df = pd.DataFrame(records)
    oni_df['date'] = pd.to_datetime(oni_df[['year', 'month']].assign(day=1))
    return oni_df.set_index('date')['oni']


def classify_enso(date, oni_series):
    target = pd.Timestamp(year=date.year, month=date.month, day=1)
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
    return 'Neutral'


def calc_boundary(rel_lon, w, threshold_frac):
    """计算边界"""
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return np.nan, np.nan
    
    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)
    
    win = (rr >= -PIVOT_DELTA_DEG) & (rr <= PIVOT_DELTA_DEG)
    if win.any():
        j0 = int(np.nanargmin(ww[win]))
        pivot_idx = np.where(win)[0][j0]
    else:
        pivot_idx = int(np.nanargmin(ww))
    
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return np.nan, np.nan
    
    thr = threshold_frac * wmin
    
    # West boundary
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:
            west_idx = i + 1
            west_idx = min(west_idx, pivot_idx)
            break
    if west_idx is None:
        west_idx = 0
    
    # East boundary
    east_idx = None
    for i in range(pivot_idx, len(ww)):
        if ww[i] >= thr:
            east_idx = i - 1
            east_idx = max(east_idx, pivot_idx)
            break
    if east_idx is None:
        east_idx = len(ww) - 1
    
    west = float(rr[west_idx])
    east = float(rr[east_idx])
    
    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return np.nan, np.nan
    
    return west, east


def main():
    print("="*70)
    print("完整范围敏感性分析（西→核心→东）")
    print("="*70)
    
    # Load data
    print("\n[1] Loading data...")
    dsw = xr.open_dataset(W_NORM_NC, engine="netcdf4")
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4")
    events_df = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    oni_series = load_oni()
    
    w = dsw["w_mjo_recon_norm"]
    if "pressure_level" in w.dims:
        w = w.rename({"pressure_level": "level"})
    w = w.sel(lon=slice(0, 180))
    
    w_low = w.sel(level=slice(LOW_LAYER[0], LOW_LAYER[1])).mean("level", skipna=True)
    w_up = w.sel(level=slice(UP_LAYER[0], UP_LAYER[1])).mean("level", skipna=True)
    
    time = pd.to_datetime(w_low["time"].values)
    lon = w_low["lon"].values
    center_lon = ds3["center_lon_track"].values
    amp = ds3["amp"].values
    
    w_low_np = w_low.values
    w_up_np = w_up.values
    
    print(f"  Events: {len(events_df)}")
    
    # 构建完整的 x 轴: 西侧(0→100) + 东侧(100→0)
    # x 轴标签: "W0" ... "W100/E100" ... "E0"
    n_half = len(THRESHOLDS_HALF)  # 21
    full_x = list(range(n_half)) + list(range(n_half, 2*n_half - 1))
    full_labels = [f"{int(t*100)}" for t in THRESHOLDS_HALF]  # 西侧
    full_labels += [f"{int(t*100)}" for t in THRESHOLDS_HALF[-2::-1]]  # 东侧 (倒序，去掉重复的100%)
    
    # 计算每个阈值下每个事件的 tilt
    results = []
    
    print("\n[2] Computing tilts...")
    for side in ['west', 'east']:
        for thr in THRESHOLDS_HALF:
            event_data = []
            for _, ev in events_df.iterrows():
                start = pd.Timestamp(ev['start_date'])
                end = pd.Timestamp(ev['end_date'])
                center_date = start + (end - start) / 2
                
                mask = (time >= start) & (time <= end)
                day_indices = np.where(mask)[0]
                
                tilt_list = []
                
                for idx in day_indices:
                    if time[idx].month not in WINTER_MONTHS:
                        continue
                    c = center_lon[idx]
                    a = amp[idx]
                    if not np.isfinite(c) or not np.isfinite(a) or a < 0.5:
                        continue
                    
                    rel = lon - c
                    wl = w_low_np[idx, :]
                    wu = w_up_np[idx, :]
                    
                    lw, le = calc_boundary(rel, wl, thr)
                    uw, ue = calc_boundary(rel, wu, thr)
                    
                    if side == 'west':
                        if np.isfinite(lw) and np.isfinite(uw):
                            tilt_list.append(lw - uw)
                    else:
                        if np.isfinite(le) and np.isfinite(ue):
                            tilt_list.append(le - ue)
                
                if len(tilt_list) > 0:
                    enso = classify_enso(center_date, oni_series)
                    if enso:
                        event_data.append({
                            'enso': enso,
                            'tilt': np.mean(tilt_list)
                        })
            
            df = pd.DataFrame(event_data)
            
            for phase in ENSO_ORDER:
                grp = df[df['enso'] == phase]
                results.append({
                    'side': side,
                    'threshold_pct': thr * 100,
                    'enso_phase': phase,
                    'n': len(grp),
                    'mean_tilt': grp['tilt'].mean() if len(grp) > 0 else np.nan,
                    'std_tilt': grp['tilt'].std() if len(grp) > 0 else np.nan,
                })
    
    results_df = pd.DataFrame(results)
    
    # 计算 p 值
    print("\n[3] Computing p-values...")
    pvalue_results = []
    
    for side in ['west', 'east']:
        for thr in THRESHOLDS_HALF:
            subset = results_df[(results_df['side'] == side) & (results_df['threshold_pct'] == thr * 100)]
            
            # 获取各组 tilt 列表
            en_tilts = []
            ln_tilts = []
            neu_tilts = []
            
            # 需要重新计算事件级数据
            event_tilts = {'El Nino': [], 'La Nina': [], 'Neutral': []}
            
            for _, ev in events_df.iterrows():
                start = pd.Timestamp(ev['start_date'])
                end = pd.Timestamp(ev['end_date'])
                center_date = start + (end - start) / 2
                
                mask = (time >= start) & (time <= end)
                day_indices = np.where(mask)[0]
                
                tilt_list = []
                
                for idx in day_indices:
                    if time[idx].month not in WINTER_MONTHS:
                        continue
                    c = center_lon[idx]
                    a = amp[idx]
                    if not np.isfinite(c) or not np.isfinite(a) or a < 0.5:
                        continue
                    
                    rel = lon - c
                    wl = w_low_np[idx, :]
                    wu = w_up_np[idx, :]
                    
                    lw, le = calc_boundary(rel, wl, thr)
                    uw, ue = calc_boundary(rel, wu, thr)
                    
                    if side == 'west':
                        if np.isfinite(lw) and np.isfinite(uw):
                            tilt_list.append(lw - uw)
                    else:
                        if np.isfinite(le) and np.isfinite(ue):
                            tilt_list.append(le - ue)
                
                if len(tilt_list) > 0:
                    enso = classify_enso(center_date, oni_series)
                    if enso:
                        event_tilts[enso].append(np.mean(tilt_list))
            
            # 计算 p 值
            p_en_ln = stats.ttest_ind(event_tilts['El Nino'], event_tilts['La Nina'], equal_var=False).pvalue if len(event_tilts['El Nino']) > 1 and len(event_tilts['La Nina']) > 1 else np.nan
            p_en_neu = stats.ttest_ind(event_tilts['El Nino'], event_tilts['Neutral'], equal_var=False).pvalue if len(event_tilts['El Nino']) > 1 and len(event_tilts['Neutral']) > 1 else np.nan
            p_ln_neu = stats.ttest_ind(event_tilts['La Nina'], event_tilts['Neutral'], equal_var=False).pvalue if len(event_tilts['La Nina']) > 1 and len(event_tilts['Neutral']) > 1 else np.nan
            
            pvalue_results.append({
                'side': side,
                'threshold_pct': thr * 100,
                'p_ElNino_vs_LaNina': p_en_ln,
                'p_ElNino_vs_Neutral': p_en_neu,
                'p_LaNina_vs_Neutral': p_ln_neu,
            })
    
    pv_df = pd.DataFrame(pvalue_results)
    
    # 构建完整 x 轴数据
    def get_full_series(df, column, phase=None):
        """合并西侧和东侧数据"""
        if phase:
            west_data = df[(df['side'] == 'west') & (df['enso_phase'] == phase)].sort_values('threshold_pct')[column].values
            east_data = df[(df['side'] == 'east') & (df['enso_phase'] == phase)].sort_values('threshold_pct', ascending=False)[column].values[1:]  # 去掉重复的100%
        else:
            west_data = df[df['side'] == 'west'].sort_values('threshold_pct')[column].values
            east_data = df[df['side'] == 'east'].sort_values('threshold_pct', ascending=False)[column].values[1:]
        return np.concatenate([west_data, east_data])
    
    def get_full_pvalue_series(df, column):
        west_data = df[df['side'] == 'west'].sort_values('threshold_pct')[column].values
        east_data = df[df['side'] == 'east'].sort_values('threshold_pct', ascending=False)[column].values[1:]
        return np.concatenate([west_data, east_data])
    
    # x 轴
    x_full = np.arange(len(THRESHOLDS_HALF) + len(THRESHOLDS_HALF) - 1)  # 41 点
    x_labels = [f"{int(t*100)}" for t in THRESHOLDS_HALF] + [f"{int(t*100)}" for t in THRESHOLDS_HALF[-2::-1]]
    
    # 生成 8 张图
    print("\n[4] Generating 8 figures...")
    
    # === 图 1: Tilt 均值 vs 阈值（各 ENSO 相）===
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    for phase in ENSO_ORDER:
        y = get_full_series(results_df, 'mean_tilt', phase)
        ax1.plot(x_full, y, 'o-', label=phase, color=ENSO_COLORS[phase], markersize=4, linewidth=2)
    ax1.axvline(len(THRESHOLDS_HALF)-1, color='purple', linestyle='--', alpha=0.7, linewidth=2)
    ax1.annotate('核心\n(100%)', xy=(len(THRESHOLDS_HALF)-1, ax1.get_ylim()[1]*0.9), fontsize=10, ha='center', color='purple')
    ax1.set_xticks(x_full[::2])
    ax1.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax1.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax1.set_ylabel('Tilt 均值 (°)', fontsize=11)
    ax1.set_title('Tilt 均值随阈值变化（西→核心→东）', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_tilt_mean_vs_threshold.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  1/8: 01_tilt_mean_vs_threshold.png")
    
    # === 图 2: 哪个 ENSO 相 Tilt 最大 ===
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    max_phases = []
    for i in range(len(x_full)):
        vals = {}
        for phase in ENSO_ORDER:
            y = get_full_series(results_df, 'mean_tilt', phase)
            vals[phase] = y[i]
        max_phases.append(max(vals, key=vals.get))
    
    bar_colors = [ENSO_COLORS[p] for p in max_phases]
    ax2.bar(x_full, [1]*len(x_full), color=bar_colors, width=1.0, alpha=0.8)
    ax2.axvline(len(THRESHOLDS_HALF)-1, color='black', linestyle='-', alpha=0.5)
    ax2.set_xticks(x_full[::2])
    ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax2.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax2.set_yticks([])
    ax2.set_title('Tilt 最大的 ENSO 相', fontsize=13, fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=ENSO_COLORS[p], label=p) for p in ENSO_ORDER]
    ax2.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_tilt_max_phase.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  2/8: 02_tilt_max_phase.png")
    
    # === 图 3: El Nino vs La Nina p 值 ===
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    y = get_full_pvalue_series(pv_df, 'p_ElNino_vs_LaNina')
    ax3.plot(x_full, y, 'o-', color='purple', markersize=4, linewidth=2, label='El Nino vs La Nina')
    ax3.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax3.axvline(len(THRESHOLDS_HALF)-1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticks(x_full[::2])
    ax3.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax3.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax3.set_ylabel('p 值', fontsize=11)
    ax3.set_title('El Nino vs La Nina p值', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_pvalue_en_vs_ln.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  3/8: 03_pvalue_en_vs_ln.png")
    
    # === 图 4: El Nino vs Neutral p 值 ===
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    y = get_full_pvalue_series(pv_df, 'p_ElNino_vs_Neutral')
    ax4.plot(x_full, y, 'o-', color='orange', markersize=4, linewidth=2, label='El Nino vs Neutral')
    ax4.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax4.axvline(len(THRESHOLDS_HALF)-1, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xticks(x_full[::2])
    ax4.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax4.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax4.set_ylabel('p 值', fontsize=11)
    ax4.set_title('El Nino vs Neutral p值', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_pvalue_en_vs_neu.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  4/8: 04_pvalue_en_vs_neu.png")
    
    # === 图 5: La Nina vs Neutral p 值 ===
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    y = get_full_pvalue_series(pv_df, 'p_LaNina_vs_Neutral')
    ax5.plot(x_full, y, 'o-', color='green', markersize=4, linewidth=2, label='La Nina vs Neutral')
    ax5.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax5.axvline(len(THRESHOLDS_HALF)-1, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xticks(x_full[::2])
    ax5.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax5.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax5.set_ylabel('p 值', fontsize=11)
    ax5.set_title('La Nina vs Neutral p值', fontsize=13, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_pvalue_ln_vs_neu.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  5/8: 05_pvalue_ln_vs_neu.png")
    
    # === 图 6: Tilt 标准差 ===
    fig6, ax6 = plt.subplots(figsize=(12, 5))
    for phase in ENSO_ORDER:
        y = get_full_series(results_df, 'std_tilt', phase)
        ax6.plot(x_full, y, 'o-', label=phase, color=ENSO_COLORS[phase], markersize=4, linewidth=2)
    ax6.axvline(len(THRESHOLDS_HALF)-1, color='purple', linestyle='--', alpha=0.7)
    ax6.set_xticks(x_full[::2])
    ax6.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax6.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax6.set_ylabel('Tilt 标准差 (°)', fontsize=11)
    ax6.set_title('Tilt 标准差随阈值变化', fontsize=13, fontweight='bold')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_tilt_std_vs_threshold.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  6/8: 06_tilt_std_vs_threshold.png")
    
    # === 图 7: Tilt 差值 (El Nino - La Nina) ===
    fig7, ax7 = plt.subplots(figsize=(12, 5))
    y_en = get_full_series(results_df, 'mean_tilt', 'El Nino')
    y_ln = get_full_series(results_df, 'mean_tilt', 'La Nina')
    y_diff = y_en - y_ln
    ax7.plot(x_full, y_diff, 'o-', color='purple', markersize=4, linewidth=2)
    ax7.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax7.axvline(len(THRESHOLDS_HALF)-1, color='purple', linestyle='--', alpha=0.7)
    ax7.fill_between(x_full, 0, y_diff, where=(y_diff > 0), alpha=0.3, color='red', label='El Nino > La Nina')
    ax7.fill_between(x_full, 0, y_diff, where=(y_diff < 0), alpha=0.3, color='blue', label='La Nina > El Nino')
    ax7.set_xticks(x_full[::2])
    ax7.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax7.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax7.set_ylabel('Tilt 差值 (°)', fontsize=11)
    ax7.set_title('El Nino - La Nina Tilt 差值', fontsize=13, fontweight='bold')
    ax7.legend(loc='upper right')
    ax7.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_tilt_diff_en_ln.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  7/8: 07_tilt_diff_en_ln.png")
    
    # === 图 8: 所有 p 值合并 ===
    fig8, ax8 = plt.subplots(figsize=(12, 5))
    ax8.plot(x_full, get_full_pvalue_series(pv_df, 'p_ElNino_vs_LaNina'), 'o-', label='El Nino vs La Nina', color='purple', markersize=4, linewidth=2)
    ax8.plot(x_full, get_full_pvalue_series(pv_df, 'p_ElNino_vs_Neutral'), 'o-', label='El Nino vs Neutral', color='orange', markersize=4, linewidth=2)
    ax8.plot(x_full, get_full_pvalue_series(pv_df, 'p_LaNina_vs_Neutral'), 'o-', label='La Nina vs Neutral', color='green', markersize=4, linewidth=2)
    ax8.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax8.axvline(len(THRESHOLDS_HALF)-1, color='gray', linestyle='--', alpha=0.5)
    ax8.set_xticks(x_full[::2])
    ax8.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax8.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax8.set_ylabel('p 值', fontsize=11)
    ax8.set_title('所有 ENSO 组对比 p值', fontsize=13, fontweight='bold')
    ax8.legend(loc='upper right')
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_pvalue_all.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  8/8: 08_pvalue_all.png")
    
    # 保存数据
    results_df.to_csv(OUT_DIR / "sensitivity_full_range_data.csv", index=False)
    pv_df.to_csv(OUT_DIR / "sensitivity_full_range_pvalues.csv", index=False)
    
    print("\n" + "="*70)
    print("Done! 8 figures saved to:", OUT_DIR)
    print("="*70)


if __name__ == "__main__":
    main()
