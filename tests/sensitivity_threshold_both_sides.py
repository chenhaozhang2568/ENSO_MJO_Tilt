# -*- coding: utf-8 -*-
"""
sensitivity_threshold_both_sides.py: 东西两侧 Tilt 阈值敏感性分析

分析不同阈值下东西两侧 Tilt 的 ENSO 分组差异
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
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 
              0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
ONI_THRESHOLD = 0.5
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
LOW_LAYER = (1000.0, 600.0)
UP_LAYER = (400.0, 200.0)
PIVOT_DELTA_DEG = 10.0
MIN_VALID_POINTS = 7

ENSO_ORDER = ["El Nino", "La Nina", "Neutral"]


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
    print("东西两侧 Tilt 阈值敏感性分析")
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
    
    # Align
    w_low_np = w_low.values
    w_up_np = w_up.values
    
    print(f"  Events: {len(events_df)}")
    
    # Compute event-level tilt for each threshold
    results = []
    
    for thr in THRESHOLDS:
        print(f"\n[Threshold {thr*100:.0f}%]", end=" ")
        
        event_data = []
        for _, ev in events_df.iterrows():
            start = pd.Timestamp(ev['start_date'])
            end = pd.Timestamp(ev['end_date'])
            center_date = start + (end - start) / 2
            
            mask = (time >= start) & (time <= end)
            day_indices = np.where(mask)[0]
            
            tilt_w_list = []
            tilt_e_list = []
            
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
                
                if np.isfinite(lw) and np.isfinite(uw):
                    tilt_w_list.append(lw - uw)
                if np.isfinite(le) and np.isfinite(ue):
                    tilt_e_list.append(le - ue)
            
            if len(tilt_w_list) > 0 and len(tilt_e_list) > 0:
                enso = classify_enso(center_date, oni_series)
                if enso:
                    event_data.append({
                        'enso': enso,
                        'tilt_west': np.mean(tilt_w_list),
                        'tilt_east': np.mean(tilt_e_list)
                    })
        
        df = pd.DataFrame(event_data)
        print(f"events={len(df)}")
        
        for phase in ENSO_ORDER:
            grp = df[df['enso'] == phase]
            results.append({
                'threshold_pct': thr * 100,
                'enso_phase': phase,
                'n': len(grp),
                'mean_tilt_west': grp['tilt_west'].mean() if len(grp) > 0 else np.nan,
                'mean_tilt_east': grp['tilt_east'].mean() if len(grp) > 0 else np.nan,
            })
    
    results_df = pd.DataFrame(results)
    
    # 可视化
    print("\n[2] Generating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图: 西侧 Tilt
    ax1 = axes[0]
    for phase in ENSO_ORDER:
        data = results_df[results_df['enso_phase'] == phase]
        ax1.plot(data['threshold_pct'], data['mean_tilt_west'], 'o-', 
                label=phase, color={'El Nino': 'red', 'La Nina': 'blue', 'Neutral': 'gray'}[phase], 
                markersize=5, linewidth=2)
    
    ax1.set_xlabel('边界阈值 (%)', fontsize=12)
    ax1.set_ylabel('西侧 Tilt 均值 (°)', fontsize=12)
    ax1.set_title('西侧 Tilt (后侧) vs 阈值', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-2, 102)
    
    # 右图: 东侧 Tilt
    ax2 = axes[1]
    for phase in ENSO_ORDER:
        data = results_df[results_df['enso_phase'] == phase]
        ax2.plot(data['threshold_pct'], data['mean_tilt_east'], 'o-', 
                label=phase, color={'El Nino': 'red', 'La Nina': 'blue', 'Neutral': 'gray'}[phase], 
                markersize=5, linewidth=2)
    
    ax2.set_xlabel('边界阈值 (%)', fontsize=12)
    ax2.set_ylabel('东侧 Tilt 均值 (°)', fontsize=12)
    ax2.set_title('东侧 Tilt (前侧) vs 阈值', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-2, 102)
    
    plt.tight_layout()
    fig_path = OUT_DIR / "threshold_sensitivity_both_sides.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()
    
    # 保存数据
    results_df.to_csv(OUT_DIR / "threshold_sensitivity_both_sides.csv", index=False)
    
    # 汇总表
    print("\n" + "="*70)
    print("阈值敏感性汇总")
    print("="*70)
    print("\n西侧 Tilt 最大者:")
    for thr in [0, 20, 50, 70]:
        subset = results_df[results_df['threshold_pct'] == thr]
        max_phase = subset.loc[subset['mean_tilt_west'].idxmax(), 'enso_phase']
        print(f"  {thr}%: {max_phase}")
    
    print("\n东侧 Tilt 最大者:")
    for thr in [0, 20, 50, 70]:
        subset = results_df[results_df['threshold_pct'] == thr]
        max_phase = subset.loc[subset['mean_tilt_east'].idxmax(), 'enso_phase']
        print(f"  {thr}%: {max_phase}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
