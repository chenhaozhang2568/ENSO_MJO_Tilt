# -*- coding: utf-8 -*-
"""
sensitivity_threshold_enso_pvalue.py: 阈值敏感性分析 - ENSO分组p值变化

分析不同边界阈值 (0~100%, 每隔5%) 对 ENSO 分组显著性检验结果的影响。
可视化 p 值随阈值的变化趋势。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# 中文字体设置（使用不含特殊字符的标签）
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial', 'DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# 注：图中使用 "El Nino" 和 "La Nina" 避免 ñ 字符显示问题

# ======================
# PATHS
# ======================
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
W_NORM_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
ONI_FILE = r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt"
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
THRESHOLDS = [i * 0.05 for i in range(21)]

MIN_VALID_POINTS = 7
PIVOT_DELTA_DEG = 10.0
EDGE_N_CONSEC = 1
OLR_MIN_THRESH = -10.0
AMP_EPS = 1e-6

# ENSO 阈值
ONI_THRESHOLD = 0.5

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

def _ascent_boundary_by_threshold(rel_lon, w, threshold_fraction, pivot_delta, n_consec):
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return (np.nan, np.nan)
    
    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)
    
    win = (rr >= -pivot_delta) & (rr <= pivot_delta)
    if win.any():
        j0 = int(np.nanargmin(ww[win]))
        pivot_idx = np.where(win)[0][j0]
    else:
        pivot_idx = int(np.nanargmin(ww))
    
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return (np.nan, np.nan)
    
    thr = float(threshold_fraction) * wmin
    
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


def load_oni():
    """加载 ONI 指数"""
    # 文件格式: SEAS YR TOTAL ANOM
    oni = pd.read_csv(ONI_FILE, sep=r'\s+', header=0)
    
    month_map = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
                'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}
    
    records = []
    for _, row in oni.iterrows():
        seas = row['SEAS']
        year = int(row['YR'])
        anom = row['ANOM']
        if seas in month_map and anom != -99.9:
            month = month_map[seas]
            records.append({'year': year, 'month': month, 'oni': anom})
    
    oni_df = pd.DataFrame(records)
    oni_df['date'] = pd.to_datetime(oni_df[['year', 'month']].assign(day=1))
    return oni_df.set_index('date')['oni']


def classify_enso(event_center_date, oni_series):
    """根据事件中心时间分类 ENSO 相位"""
    month_start = event_center_date.replace(day=1)
    if month_start in oni_series.index:
        oni_val = oni_series.loc[month_start]
        if oni_val >= ONI_THRESHOLD:
            return 'El Nino'
        elif oni_val <= -ONI_THRESHOLD:
            return 'La Nina'
        else:
            return 'Neutral'
    return None


def compute_event_tilts_for_threshold(threshold, center_a, w_low_a, w_up_a, amp_a, 
                                       lon, time, winter, eventmask, events_df):
    """计算给定阈值下每个事件的平均 tilt"""
    amp_ok = np.isfinite(amp_a.values) & (amp_a.values > AMP_EPS)
    w_low_norm = w_low_a.values.astype(float)
    w_up_norm = w_up_a.values.astype(float)
    c_np = center_a.values.astype(float)
    
    n = len(time)
    daily_tilts = np.full(n, np.nan)
    
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
            daily_tilts[i] = float(lw - uw)
    
    # 汇总到事件级别
    event_tilts = []
    for _, ev in events_df.iterrows():
        start = pd.Timestamp(ev['start_date'])
        end = pd.Timestamp(ev['end_date'])
        mask = (time >= start) & (time <= end)
        tilts = daily_tilts[mask]
        valid = tilts[np.isfinite(tilts)]
        if len(valid) > 0:
            event_tilts.append({
                'event_id': ev['event_id'],
                'mean_tilt': np.mean(valid),
                'start_date': start,
                'end_date': end
            })
    
    return pd.DataFrame(event_tilts)


def main():
    print("Loading data...")
    
    # Load Step3
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    center = ds3["center_lon_track"].astype(float)
    olr_center = ds3["olr_center_track"].astype(float)
    amp = ds3["amp"].astype(float)
    
    # Load events
    events_df = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    
    # Load ONI
    oni_series = load_oni()
    
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
    eventmask = _mask_event_days(time, EVENTS_CSV)
    lon = w_low_a["lon"].values.astype(float)
    
    print(f"Analyzing {len(THRESHOLDS)} thresholds for ENSO p-values...")
    
    results = []
    
    for thr in THRESHOLDS:
        print(f"  Threshold = {thr:.2f}...")
        
        # 计算事件级 tilt
        event_tilts = compute_event_tilts_for_threshold(
            thr, center_a, w_low_a, w_up_a, amp_a, lon, time, winter, eventmask, events_df
        )
        
        if event_tilts.empty:
            continue
        
        # 分类 ENSO
        event_tilts['center_date'] = event_tilts['start_date'] + (event_tilts['end_date'] - event_tilts['start_date']) / 2
        event_tilts['enso_phase'] = event_tilts['center_date'].apply(lambda x: classify_enso(x, oni_series))
        event_tilts = event_tilts.dropna(subset=['enso_phase'])
        
        # 分组
        el_nino = event_tilts[event_tilts['enso_phase'] == 'El Nino']['mean_tilt'].values
        la_nina = event_tilts[event_tilts['enso_phase'] == 'La Nina']['mean_tilt'].values
        neutral = event_tilts[event_tilts['enso_phase'] == 'Neutral']['mean_tilt'].values
        
        # T-tests
        p_en_ln = stats.ttest_ind(el_nino, la_nina, equal_var=False).pvalue if len(el_nino) > 1 and len(la_nina) > 1 else np.nan
        p_en_neu = stats.ttest_ind(el_nino, neutral, equal_var=False).pvalue if len(el_nino) > 1 and len(neutral) > 1 else np.nan
        p_ln_neu = stats.ttest_ind(la_nina, neutral, equal_var=False).pvalue if len(la_nina) > 1 and len(neutral) > 1 else np.nan
        
        results.append({
            'threshold': thr,
            'threshold_pct': thr * 100,
            'n_events': len(event_tilts),
            'n_el_nino': len(el_nino),
            'n_la_nina': len(la_nina),
            'n_neutral': len(neutral),
            'mean_el_nino': np.mean(el_nino) if len(el_nino) > 0 else np.nan,
            'mean_la_nina': np.mean(la_nina) if len(la_nina) > 0 else np.nan,
            'mean_neutral': np.mean(neutral) if len(neutral) > 0 else np.nan,
            'p_ElNino_vs_LaNina': p_en_ln,
            'p_ElNino_vs_Neutral': p_en_neu,
            'p_LaNina_vs_Neutral': p_ln_neu
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("ENSO P-VALUE SENSITIVITY ANALYSIS")
    print("="*80)
    print(df[['threshold_pct', 'mean_el_nino', 'mean_la_nina', 'mean_neutral', 
              'p_ElNino_vs_LaNina', 'p_ElNino_vs_Neutral', 'p_LaNina_vs_Neutral']].to_string(index=False))
    
    # 保存 CSV
    csv_path = OUT_DIR / "threshold_enso_pvalue_sensitivity.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    
    # ===================
    # 可视化
    # ===================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- 图1: p值随阈值变化 ---
    ax1 = axes[0, 0]
    ax1.plot(df['threshold_pct'], df['p_ElNino_vs_LaNina'], 'o-', color='red', 
             markersize=6, linewidth=2, label='El Nino vs La Nina')
    ax1.plot(df['threshold_pct'], df['p_ElNino_vs_Neutral'], 's-', color='blue', 
             markersize=6, linewidth=2, label='El Nino vs Neutral')
    ax1.plot(df['threshold_pct'], df['p_LaNina_vs_Neutral'], '^-', color='green', 
             markersize=6, linewidth=2, label='La Nina vs Neutral')
    
    # 显著性水平线
    ax1.axhline(0.05, color='orange', linestyle='--', linewidth=2, label='p=0.05')
    ax1.axhline(0.10, color='gray', linestyle=':', linewidth=1.5, label='p=0.10')
    
    ax1.set_xlabel('边界阈值 (%)', fontsize=12)
    ax1.set_ylabel('p 值', fontsize=12)
    ax1.set_title('ENSO 分组 T-test p值随阈值变化', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-2, 102)
    ax1.set_xticks(range(0, 105, 10))
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)
    
    # --- 图2: 各组均值随阈值变化 ---
    ax2 = axes[0, 1]
    ax2.plot(df['threshold_pct'], df['mean_el_nino'], 'o-', color='red', 
             markersize=6, linewidth=2, label='El Nino')
    ax2.plot(df['threshold_pct'], df['mean_la_nina'], 's-', color='blue', 
             markersize=6, linewidth=2, label='La Nina')
    ax2.plot(df['threshold_pct'], df['mean_neutral'], '^-', color='gray', 
             markersize=6, linewidth=2, label='Neutral')
    
    ax2.set_xlabel('边界阈值 (%)', fontsize=12)
    ax2.set_ylabel('Tilt 均值 (°)', fontsize=12)
    ax2.set_title('各 ENSO 分组 Tilt 均值随阈值变化', fontsize=14, fontweight='bold')
    ax2.set_xlim(-2, 102)
    ax2.set_xticks(range(0, 105, 10))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # --- 图3: 各组均值差异随阈值变化 ---
    ax3 = axes[1, 0]
    diff_en_ln = df['mean_el_nino'] - df['mean_la_nina']
    diff_en_neu = df['mean_el_nino'] - df['mean_neutral']
    diff_ln_neu = df['mean_la_nina'] - df['mean_neutral']
    
    ax3.plot(df['threshold_pct'], diff_en_ln, 'o-', color='red', 
             markersize=6, linewidth=2, label='El Nino - La Nina')
    ax3.plot(df['threshold_pct'], diff_en_neu, 's-', color='blue', 
             markersize=6, linewidth=2, label='El Nino - Neutral')
    ax3.plot(df['threshold_pct'], diff_ln_neu, '^-', color='green', 
             markersize=6, linewidth=2, label='La Nina - Neutral')
    
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('边界阈值 (%)', fontsize=12)
    ax3.set_ylabel('Tilt 均值差 (°)', fontsize=12)
    ax3.set_title('ENSO 分组间 Tilt 均值差异', fontsize=14, fontweight='bold')
    ax3.set_xlim(-2, 102)
    ax3.set_xticks(range(0, 105, 10))
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # --- 图4: La Nina vs Neutral 详细（之前显著的组）---
    ax4 = axes[1, 1]
    ax4.plot(df['threshold_pct'], df['p_LaNina_vs_Neutral'], 'o-', color='green', 
             markersize=8, linewidth=2.5, label='La Nina vs Neutral p值')
    
    # 标注所有点的 p 值
    for i, row in df.iterrows():
        offset_y = 0.05 if i % 2 == 0 else -0.08
        ax4.annotate(f"{row['p_LaNina_vs_Neutral']:.3f}", 
                    (row['threshold_pct'], row['p_LaNina_vs_Neutral']),
                    textcoords="offset points", xytext=(0, offset_y * 100),
                    ha='center', fontsize=7, color='darkgreen')
    
    ax4.axhline(0.05, color='orange', linestyle='--', linewidth=2, label='p=0.05 (显著)')
    ax4.axhline(0.10, color='gray', linestyle=':', linewidth=1.5, label='p=0.10')
    
    # 填充显著区域
    ax4.fill_between(df['threshold_pct'], 0, 0.05, alpha=0.2, color='green', label='显著区域')
    
    ax4.set_xlabel('边界阈值 (%)', fontsize=12)
    ax4.set_ylabel('p 值', fontsize=12)
    ax4.set_title('La Nina vs Neutral 显著性检验', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 0.8)
    ax4.set_xlim(-2, 102)
    ax4.set_xticks(range(0, 105, 10))
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    fig_path = OUT_DIR / "threshold_enso_pvalue_sensitivity.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close()
    
    # ===================
    # 找最显著的阈值
    # ===================
    print("\n" + "="*60)
    print("SUMMARY: 最显著的阈值区间")
    print("="*60)
    
    min_p_ln_neu = df['p_LaNina_vs_Neutral'].min()
    best_thr_ln_neu = df.loc[df['p_LaNina_vs_Neutral'].idxmin(), 'threshold_pct']
    print(f"La Nina vs Neutral: 最低 p={min_p_ln_neu:.4f} @ 阈值={best_thr_ln_neu:.0f}%")
    
    sig_range = df[df['p_LaNina_vs_Neutral'] < 0.05]['threshold_pct']
    if len(sig_range) > 0:
        print(f"  显著 (p<0.05) 的阈值范围: {sig_range.min():.0f}% ~ {sig_range.max():.0f}%")
    else:
        print("  无显著 (p<0.05) 的阈值")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
