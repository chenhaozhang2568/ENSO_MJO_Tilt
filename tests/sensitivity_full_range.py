# -*- coding: utf-8 -*-
"""
sensitivity_full_range.py — Tilt 敏感性完整范围分析

功能：
    在西→核心→东全范围内扫描不同半高宽阈值的 Tilt 统计量、
    ENSO 分组 p 值和东/西两侧 Tilt 差异。
输入：
    era5_mjo_recon_w_norm_1979-2022.nc, mjo_mvEOF_step3_1979-2022.nc,
    mjo_events_step3_1979-2022.csv, oni.ascii.txt
输出：
    figures/sensitivity/ 下的 8 张图 + CSV 数据
用法：
    python tests/sensitivity_full_range.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy import stats

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
W_RECON_NC  = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"
OUT_DIR    = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# SETTINGS
# ======================
THRESHOLDS_HALF = [i * 0.05 for i in range(21)]   # 0% ~ 100%
WINTER_MONTHS   = {11, 12, 1, 2, 3, 4}
LOW_LAYER       = (1000.0, 600.0)
UP_LAYER        = (400.0, 200.0)
PIVOT_DELTA_DEG = 10.0
MIN_VALID_POINTS = 7
SMOOTH_WINDOW   = 1    # 1=不平滑，与 03 一致
AMP_EPS         = 1e-6

ENSO_ORDER  = ["El Nino", "La Nina", "Neutral"]
ENSO_COLORS = {"El Nino": "#E74C3C", "La Nina": "#3498DB", "Neutral": "#95A5A6"}


# ======================
# HELPERS
# ======================



def calc_boundary(rel_lon, w, threshold_frac):
    """计算上升区西/东边界"""
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return np.nan, np.nan

    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)

    # 滑动平均
    if SMOOTH_WINDOW > 1 and len(ww) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        ww = np.convolve(ww, kernel, mode='same')

    # pivot: 对流中心附近最强上升
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

    # West
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:
            west_idx = min(i + 1, pivot_idx)
            break
    if west_idx is None:
        return np.nan, np.nan  # 没找到边界 → 无效

    # East
    east_idx = None
    for i in range(pivot_idx, len(ww)):
        if ww[i] >= thr:
            east_idx = max(i - 1, pivot_idx)
            break
    if east_idx is None:
        return np.nan, np.nan  # 没找到边界 → 无效

    west = float(rr[west_idx])
    east = float(rr[east_idx])

    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return np.nan, np.nan

    return west, east


def compute_all_tilts(events_df, time, lon, center_lon, amp, w_low_np, w_up_np, 
                      enso_map, thresholds):
    """
    一次性计算所有阈值下、每个事件的西/东 tilt。
    返回: results_df (阈值×ENSO 统计), event_tilts_dict (阈值→side→phase→list)
    """
    amp_ok = np.isfinite(amp) & (amp > AMP_EPS)
    
    results = []
    event_tilts_dict = {}  # key: (side, thr) → {phase: [tilts]}
    
    for thr in thresholds:
        # 每个阈值：收集所有事件的西/东 tilt
        event_west = {p: [] for p in ENSO_ORDER}
        event_east = {p: [] for p in ENSO_ORDER}
        
        for _, ev in events_df.iterrows():
            eid = ev['event_id']
            enso = enso_map.get(eid)
            if enso is None or enso not in ENSO_ORDER:
                continue
            
            start = pd.Timestamp(ev['start_date'])
            end = pd.Timestamp(ev['end_date'])
            mask = (time >= start) & (time <= end)
            day_indices = np.where(mask)[0]
            
            tw_list, te_list = [], []
            
            for idx in day_indices:
                c = center_lon[idx]
                if not np.isfinite(c) or not amp_ok[idx]:
                    continue
                
                rel = lon - c
                lw, le = calc_boundary(rel, w_low_np[idx, :], thr)
                uw, ue = calc_boundary(rel, w_up_np[idx, :], thr)
                
                if np.isfinite(lw) and np.isfinite(uw):
                    tw_list.append(lw - uw)
                if np.isfinite(le) and np.isfinite(ue):
                    te_list.append(le - ue)
            
            if len(tw_list) > 0:
                event_west[enso].append(np.mean(tw_list))
            if len(te_list) > 0:
                event_east[enso].append(np.mean(te_list))
        
        # 保存事件级数据
        event_tilts_dict[('west', thr)] = event_west
        event_tilts_dict[('east', thr)] = event_east
        
        # 汇总统计
        for side, data in [('west', event_west), ('east', event_east)]:
            for phase in ENSO_ORDER:
                arr = np.array(data[phase]) if data[phase] else np.array([])
                results.append({
                    'side': side,
                    'threshold_pct': thr * 100,
                    'enso_phase': phase,
                    'n': len(arr),
                    'mean_tilt': np.mean(arr) if len(arr) > 0 else np.nan,
                    'std_tilt': np.std(arr) if len(arr) > 0 else np.nan,
                })
    
    return pd.DataFrame(results), event_tilts_dict


def compute_pvalues(event_tilts_dict, thresholds):
    """从已有的事件级 tilt 数据计算 p 值"""
    pvalue_results = []
    
    for side in ['west', 'east']:
        for thr in thresholds:
            data = event_tilts_dict[(side, thr)]
            en = data['El Nino']
            ln = data['La Nina']
            neu = data['Neutral']
            
            p_en_ln = stats.ttest_ind(en, ln, equal_var=False).pvalue if len(en) > 1 and len(ln) > 1 else np.nan
            p_en_neu = stats.ttest_ind(en, neu, equal_var=False).pvalue if len(en) > 1 and len(neu) > 1 else np.nan
            p_ln_neu = stats.ttest_ind(ln, neu, equal_var=False).pvalue if len(ln) > 1 and len(neu) > 1 else np.nan
            
            pvalue_results.append({
                'side': side,
                'threshold_pct': thr * 100,
                'p_ElNino_vs_LaNina': p_en_ln,
                'p_ElNino_vs_Neutral': p_en_neu,
                'p_LaNina_vs_Neutral': p_ln_neu,
            })
    
    return pd.DataFrame(pvalue_results)


# ======================
# 绘图辅助
# ======================
def get_full_series(df, column, phase=None):
    """合并西侧(0→100)和东侧(100→0)成连续序列"""
    if phase:
        west = df[(df['side'] == 'west') & (df['enso_phase'] == phase)].sort_values('threshold_pct')[column].values
        east = df[(df['side'] == 'east') & (df['enso_phase'] == phase)].sort_values('threshold_pct', ascending=False)[column].values[1:]
    else:
        west = df[df['side'] == 'west'].sort_values('threshold_pct')[column].values
        east = df[df['side'] == 'east'].sort_values('threshold_pct', ascending=False)[column].values[1:]
    return np.concatenate([west, east])


def get_full_pv_series(df, column):
    west = df[df['side'] == 'west'].sort_values('threshold_pct')[column].values
    east = df[df['side'] == 'east'].sort_values('threshold_pct', ascending=False)[column].values[1:]
    return np.concatenate([west, east])


# ======================
# MAIN
# ======================
def main():
    print("=" * 70)
    print("完整范围敏感性分析（西→核心→东）")
    print("=" * 70)

    # ---- 1. Load ----
    print("\n[1] Loading data...")
    dsw = xr.open_dataset(W_RECON_NC, engine="netcdf4")
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4")
    events_df = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    enso_stats = pd.read_csv(ENSO_STATS_CSV)
    enso_map = dict(zip(enso_stats['event_id'], enso_stats['enso_phase']))

    w = dsw["w_mjo_recon_norm"]
    if "pressure_level" in w.dims:
        w = w.rename({"pressure_level": "level"})

    # 将经度从 -180~180 转为 0~360，以支持 lon > 180
    lon_vals = w["lon"].values
    if lon_vals.min() < 0:
        new_lon = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
        w = w.assign_coords(lon=new_lon).sortby("lon")

    w = w.sel(lon=slice(0, 240))

    w_low = w.sel(level=slice(LOW_LAYER[0], LOW_LAYER[1])).mean("level", skipna=True)
    w_up = w.sel(level=slice(UP_LAYER[0], UP_LAYER[1])).mean("level", skipna=True)

    # Align
    center_da = ds3["center_lon_track"].astype(float)
    amp_da = ds3["amp"].astype(float)
    center_a, amp_a, w_low_a, w_up_a = xr.align(
        center_da, amp_da, w_low, w_up, join="inner"
    )

    time = pd.to_datetime(center_a["time"].values)
    lon = w_low_a["lon"].values
    center_lon = center_a.values
    amp = amp_a.values.astype(float)
    w_low_np = w_low_a.values
    w_up_np = w_up_a.values

    print(f"  Events: {len(events_df)}")
    print(f"  SMOOTH_WINDOW={SMOOTH_WINDOW}")

    # ---- 2. Compute ----
    print("\n[2] Computing tilts for all thresholds...")
    results_df, event_tilts_dict = compute_all_tilts(
        events_df, time, lon, center_lon, amp, w_low_np, w_up_np,
        enso_map, THRESHOLDS_HALF
    )

    print("\n[3] Computing p-values...")
    pv_df = compute_pvalues(event_tilts_dict, THRESHOLDS_HALF)

    # 保存 CSV
    results_df.to_csv(OUT_DIR / "sensitivity_full_range_data.csv", index=False)
    pv_df.to_csv(OUT_DIR / "sensitivity_full_range_pvalues.csv", index=False)

    # ---- 3. 绘图 ----
    print("\n[4] Generating 8 figures...")

    n_half = len(THRESHOLDS_HALF)
    x_full = np.arange(2 * n_half - 1)
    x_labels = [f"{int(t*100)}" for t in THRESHOLDS_HALF] + \
               [f"{int(t*100)}" for t in THRESHOLDS_HALF[-2::-1]]
    x_ticks = x_full[::2]
    x_tick_labels = [x_labels[i] for i in range(0, len(x_labels), 2)]
    center_x = n_half - 1

    def setup_ax(ax, ylabel, title):
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, fontsize=8)
        ax.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # --- 图1: Tilt 均值 ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for phase in ENSO_ORDER:
        y = get_full_series(results_df, 'mean_tilt', phase)
        ax.plot(x_full, y, 'o-', label=phase, color=ENSO_COLORS[phase], markersize=4, linewidth=2)
    ax.axvline(center_x, color='purple', linestyle='--', alpha=0.7, linewidth=2)
    setup_ax(ax, 'Tilt 均值 (°)', 'Tilt 均值随阈值变化（西→核心→东）')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_tilt_mean_vs_threshold.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  1/8: 01_tilt_mean_vs_threshold.png")

    # --- 图2: Tilt 最大的 ENSO 相 ---
    fig, ax = plt.subplots(figsize=(12, 3))
    max_phases = []
    for i in range(len(x_full)):
        vals = {p: get_full_series(results_df, 'mean_tilt', p)[i] for p in ENSO_ORDER}
        max_phases.append(max(vals, key=vals.get))
    ax.bar(x_full, [1]*len(x_full), color=[ENSO_COLORS[p] for p in max_phases], width=1.0, alpha=0.8)
    ax.axvline(center_x, color='black', linestyle='-', alpha=0.5)
    setup_ax(ax, '', 'Tilt 最大的 ENSO 相')
    ax.set_yticks([])
    ax.legend(handles=[Patch(facecolor=ENSO_COLORS[p], label=p) for p in ENSO_ORDER], loc='upper right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_tilt_max_phase.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  2/8: 02_tilt_max_phase.png")

    # --- 图3-5: p 值（分别 + 合并）---
    pv_pairs = [
        ('p_ElNino_vs_LaNina',  'El Nino vs La Nina',  'purple', '03'),
        ('p_ElNino_vs_Neutral', 'El Nino vs Neutral',  'orange', '04'),
        ('p_LaNina_vs_Neutral', 'La Nina vs Neutral',  'green',  '05'),
    ]
    for col, label, color, num in pv_pairs:
        fig, ax = plt.subplots(figsize=(12, 5))
        y = get_full_pv_series(pv_df, col)
        ax.plot(x_full, y, 'o-', color=color, markersize=4, linewidth=2, label=label)
        ax.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax.axvline(center_x, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(x_full, 0, 0.05, alpha=0.1, color='green')
        setup_ax(ax, 'p 值', f'{label} p值')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{num}_pvalue_{col.lower()}.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {num[1]}/8: {num}_pvalue_{col.lower()}.png")

    # --- 图6: Tilt 标准差 ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for phase in ENSO_ORDER:
        y = get_full_series(results_df, 'std_tilt', phase)
        ax.plot(x_full, y, 'o-', label=phase, color=ENSO_COLORS[phase], markersize=4, linewidth=2)
    ax.axvline(center_x, color='purple', linestyle='--', alpha=0.7)
    setup_ax(ax, 'Tilt 标准差 (°)', 'Tilt 标准差随阈值变化')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "06_tilt_std_vs_threshold.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  6/8: 06_tilt_std_vs_threshold.png")

    # --- 图7: El Nino - La Nina 差值 ---
    fig, ax = plt.subplots(figsize=(12, 5))
    y_en = get_full_series(results_df, 'mean_tilt', 'El Nino')
    y_ln = get_full_series(results_df, 'mean_tilt', 'La Nina')
    y_diff = y_en - y_ln
    ax.plot(x_full, y_diff, 'o-', color='purple', markersize=4, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(center_x, color='purple', linestyle='--', alpha=0.7)
    ax.fill_between(x_full, 0, y_diff, where=(y_diff > 0), alpha=0.3, color='red', label='El Nino > La Nina')
    ax.fill_between(x_full, 0, y_diff, where=(y_diff < 0), alpha=0.3, color='blue', label='La Nina > El Nino')
    setup_ax(ax, 'Tilt 差值 (°)', 'El Nino − La Nina Tilt 差值')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_tilt_diff_en_ln.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  7/8: 07_tilt_diff_en_ln.png")

    # --- 图8: 所有 p 值合并 ---
    fig, ax = plt.subplots(figsize=(12, 5))
    for col, label, color, _ in pv_pairs:
        ax.plot(x_full, get_full_pv_series(pv_df, col), 'o-', label=label, color=color, markersize=4, linewidth=2)
    ax.axhline(0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
    ax.axvline(center_x, color='gray', linestyle='--', alpha=0.5)
    setup_ax(ax, 'p 值', '所有 ENSO 组对比 p值')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_pvalue_all.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  8/8: 08_pvalue_all.png")

    print(f"\n{'='*70}")
    print(f"Done! 8 figures + 2 CSVs saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
