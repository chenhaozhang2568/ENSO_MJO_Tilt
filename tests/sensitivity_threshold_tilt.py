# -*- coding: utf-8 -*-
"""
sensitivity_threshold_tilt.py: 边界阈值敏感性分析

分析不同边界阈值 (0, 0.05, 0.1, ..., 0.7) 对 tilt 计算结果的影响。
生成可视化图表展示 tilt 统计量随阈值的变化趋势。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

# 中文字体设置
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
W_NORM_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# SETTINGS
# ======================
START_DATE = "1979-01-01"
END_DATE = "2022-12-31"
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
TRACK_LON_MIN = 0.0
TRACK_LON_MAX = 180.0
LOW_LAYER = (1000.0, 600.0)
UP_LAYER = (400.0, 200.0)

# 测试的阈值列表：0% ~ 100%，每隔 5%
THRESHOLDS = [i * 0.05 for i in range(21)]  # [0.0, 0.05, 0.10, ..., 0.95, 1.0]

MIN_VALID_POINTS = 7
PIVOT_DELTA_DEG = 10.0
EDGE_N_CONSEC = 1
OLR_MIN_THRESH = -10.0
AMP_EPS = 1e-6

# ======================
# HELPERS
# ======================
def _winter_np(time_index: pd.DatetimeIndex) -> np.ndarray:
    return np.isin(time_index.month, list(WINTER_MONTHS)).astype(bool)

def _mask_event_days(time: pd.DatetimeIndex, events_csv: str) -> np.ndarray:
    ev = pd.read_csv(events_csv, parse_dates=["start_date", "end_date"])
    m = np.zeros(len(time), dtype=bool)
    if ev.empty:
        return m
    tv = time.values.astype("datetime64[ns]")
    for _, r in ev.iterrows():
        s = np.datetime64(pd.Timestamp(r["start_date"]).normalize().to_datetime64())
        e = np.datetime64(pd.Timestamp(r["end_date"]).normalize().to_datetime64())
        i0 = int(np.searchsorted(tv, s, side="left"))
        i1 = int(np.searchsorted(tv, e, side="right")) - 1
        if i1 >= i0:
            m[i0:i1+1] = True
    return m

def _ascent_boundary_by_threshold(
    rel_lon: np.ndarray,
    w: np.ndarray,
    threshold_fraction: float,
    pivot_delta: float,
    n_consec: int
):
    """
    通用边界计算函数：
      threshold_fraction = 0.0 -> zero-crossing (ω >= 0)
      threshold_fraction = 0.5 -> half-max (ω >= 0.5 × ω_min)
    """
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return (np.nan, np.nan)
    
    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)
    
    # 找 pivot
    win = (rr >= -pivot_delta) & (rr <= pivot_delta)
    if win.any():
        j0 = int(np.nanargmin(ww[win]))
        pivot_idx = np.where(win)[0][j0]
    else:
        pivot_idx = int(np.nanargmin(ww))
    
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return (np.nan, np.nan)
    
    # 阈值
    thr = float(threshold_fraction) * wmin  # wmin<0
    
    # west edge
    outside = 0
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:
            outside += 1
        else:
            outside = 0
        if outside >= n_consec:
            cand = i + n_consec
            cand = min(cand, pivot_idx)
            west_idx = cand
            break
    if west_idx is None:
        west_idx = 0
    
    # east edge
    outside = 0
    east_idx = None
    for i in range(pivot_idx, len(ww)):
        if ww[i] >= thr:
            outside += 1
        else:
            outside = 0
        if outside >= n_consec:
            cand = i - n_consec
            cand = max(cand, pivot_idx)
            east_idx = cand
            break
    if east_idx is None:
        east_idx = len(ww) - 1
    
    west = float(rr[west_idx])
    east = float(rr[east_idx])
    
    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return (np.nan, np.nan)
    
    return (west, east)


def compute_tilt_for_threshold(threshold: float, center_a, w_low_a, w_up_a, 
                                amp_a, lon, time, winter, eventmask, active):
    """计算给定阈值下的所有日 tilt 值"""
    amp_ok = np.isfinite(amp_a.values) & (amp_a.values > AMP_EPS)
    w_low_norm = w_low_a.values.astype(float)
    w_up_norm = w_up_a.values.astype(float)
    c_np = center_a.values.astype(float)
    
    n = len(time)
    tilt_vals = []
    
    for i in range(n):
        if not winter[i]:
            continue
        if not eventmask[i]:
            continue
        c = c_np[i]
        if not np.isfinite(c):
            continue
        if not amp_ok[i]:
            continue
        
        rel = lon - float(c)
        wl = w_low_norm[i, :]
        wu = w_up_norm[i, :]
        
        lw, le = _ascent_boundary_by_threshold(rel, wl, threshold, PIVOT_DELTA_DEG, EDGE_N_CONSEC)
        uw, ue = _ascent_boundary_by_threshold(rel, wu, threshold, PIVOT_DELTA_DEG, EDGE_N_CONSEC)
        
        if np.isfinite(lw) and np.isfinite(uw):
            tilt_vals.append(float(lw - uw))
    
    return np.array(tilt_vals)


def main():
    print("Loading data...")
    
    # Load Step3
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    center = ds3["center_lon_track"].astype(float)
    olr_center = ds3["olr_center_track"].astype(float)
    amp = ds3["amp"].astype(float)
    
    # Load normalized omega
    dsw = xr.open_dataset(W_NORM_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    w_var = "w_mjo_recon_norm"
    w = dsw[w_var]
    if "pressure_level" in w.dims:
        w = w.rename({"pressure_level": "level"})
    w = w.sel(lon=slice(TRACK_LON_MIN, TRACK_LON_MAX))
    
    w_low = w.sel(level=slice(LOW_LAYER[0], LOW_LAYER[1])).mean("level", skipna=True)
    w_up = w.sel(level=slice(UP_LAYER[0], UP_LAYER[1])).mean("level", skipna=True)
    w_low = w_low.transpose("time", "lon")
    w_up = w_up.transpose("time", "lon")
    
    # Align
    center_a, olr_center_a, amp_a, w_low_a, w_up_a = xr.align(
        center, olr_center, amp, w_low, w_up, join="inner"
    )
    time = pd.to_datetime(center_a["time"].values)
    winter = _winter_np(time)
    active = (olr_center_a.values.astype(float) <= OLR_MIN_THRESH) & np.isfinite(olr_center_a.values)
    eventmask = _mask_event_days(time, EVENTS_CSV)
    lon = w_low_a["lon"].values.astype(float)
    
    print(f"Analyzing {len(THRESHOLDS)} thresholds...")
    
    # 收集结果
    results = []
    all_tilt_data = {}
    
    for thr in THRESHOLDS:
        print(f"  Threshold = {thr:.2f}...")
        tilt_arr = compute_tilt_for_threshold(
            thr, center_a, w_low_a, w_up_a, amp_a, lon, time, winter, eventmask, active
        )
        
        if len(tilt_arr) > 0:
            results.append({
                'threshold': thr,
                'n_valid': len(tilt_arr),
                'mean': np.nanmean(tilt_arr),
                'std': np.nanstd(tilt_arr),
                'median': np.nanmedian(tilt_arr),
                'p5': np.nanpercentile(tilt_arr, 5),
                'p25': np.nanpercentile(tilt_arr, 25),
                'p75': np.nanpercentile(tilt_arr, 75),
                'p95': np.nanpercentile(tilt_arr, 95),
                'min': np.nanmin(tilt_arr),
                'max': np.nanmax(tilt_arr),
                'sem': np.nanstd(tilt_arr) / np.sqrt(len(tilt_arr))
            })
            all_tilt_data[thr] = tilt_arr
    
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    
    # 保存 CSV
    csv_path = OUT_DIR / "threshold_sensitivity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    # ===================
    # 可视化
    # ===================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- 图1: 均值 ± 标准误 ---
    ax1 = axes[0, 0]
    ax1.errorbar(df['threshold'] * 100, df['mean'], yerr=df['sem'], 
                 fmt='o-', capsize=3, capthick=1.5, markersize=6, 
                 color='steelblue', linewidth=2)
    ax1.fill_between(df['threshold'] * 100, df['mean'] - df['std'], df['mean'] + df['std'],
                     alpha=0.2, color='steelblue', label='±1 std')
    ax1.set_xlabel('边界阈值 (%)', fontsize=12)
    ax1.set_ylabel('Tilt 均值 (°)', fontsize=12)
    ax1.set_title('Tilt 均值随阈值变化 (每隔5%)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    ax1.set_xticks(range(0, 105, 10))  # 主刻度每隔10%
    ax1.set_xlim(-2, 102)
    
    # 标注所有点的数值，交替上下偏移避免重叠
    for i, row in df.iterrows():
        thr_pct = row['threshold'] * 100
        offset_y = 12 if i % 2 == 0 else -18  # 交替上下
        ax1.annotate(f"{row['mean']:.1f}", 
                    (thr_pct, row['mean']),
                    textcoords="offset points", xytext=(0, offset_y),
                    ha='center', fontsize=7, color='darkblue')
    
    # --- 图2: 标准差变化 ---
    ax2 = axes[0, 1]
    ax2.plot(df['threshold'], df['std'], 'o-', color='coral', 
             markersize=8, linewidth=2, label='标准差')
    ax2.set_xlabel('边界阈值 (× ω_min)', fontsize=12)
    ax2.set_ylabel('Tilt 标准差 (°)', fontsize=12)
    ax2.set_title('Tilt 离散程度随阈值变化', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # --- 图3: 百分位范围 ---
    ax3 = axes[1, 0]
    ax3.fill_between(df['threshold'], df['p5'], df['p95'], 
                     alpha=0.3, color='green', label='5-95%')
    ax3.fill_between(df['threshold'], df['p25'], df['p75'], 
                     alpha=0.5, color='green', label='25-75%')
    ax3.plot(df['threshold'], df['median'], 'o-', color='darkgreen', 
             markersize=6, linewidth=2, label='中位数')
    ax3.set_xlabel('边界阈值 (× ω_min)', fontsize=12)
    ax3.set_ylabel('Tilt (°)', fontsize=12)
    ax3.set_title('Tilt 分布范围随阈值变化', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.legend(loc='upper right')
    
    # --- 图4: 箱线图 ---
    ax4 = axes[1, 1]
    box_data = [all_tilt_data[thr] for thr in THRESHOLDS if thr in all_tilt_data]
    box_labels = [f'{thr:.2f}' for thr in THRESHOLDS if thr in all_tilt_data]
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # 设置颜色渐变
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.set_xlabel('边界阈值 (× ω_min)', fontsize=12)
    ax4.set_ylabel('Tilt (°)', fontsize=12)
    ax4.set_title('Tilt 分布箱线图', fontsize=14, fontweight='bold')
    ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax4.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = OUT_DIR / "threshold_sensitivity_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()
    
    # ===================
    # 额外：单独的分布曲线图
    # ===================
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    selected = [0.0, 0.05, 0.10, 0.30, 0.50]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(selected)))
    
    for thr, color in zip(selected, colors):
        if thr in all_tilt_data:
            data = all_tilt_data[thr]
            bins = np.arange(-60, 70, 5)
            ax.hist(data, bins=bins, alpha=0.4, color=color, 
                   label=f'阈值={thr:.2f} (μ={np.mean(data):.1f}°)', 
                   density=True, histtype='stepfilled')
    
    ax.set_xlabel('Tilt (°)', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title('不同阈值下 Tilt 分布对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    fig2_path = OUT_DIR / "threshold_sensitivity_distributions.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig2_path}")
    plt.close()
    
    print("\nDone!")


if __name__ == "__main__":
    main()
