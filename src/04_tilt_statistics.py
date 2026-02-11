# -*- coding: utf-8 -*-
"""
04_tilt_statistics.py: Step5 - 逐事件 Tilt 统计分析

================================================================================
功能描述：
    本脚本计算每个 MJO 事件的 Tilt 统计量（均值、标准差、中位数等），
    并生成 Tilt 分布直方图。

统计内容：
    - 每事件 Tilt 均值、标准差、中位数、最小/最大值
    - 有效 Tilt 天数统计
    - 全样本 Tilt 分布直方图

输入数据：
    - 逐日 Tilt：tilt_daily_step4_layermean_1979-2022.nc
    - MJO 事件列表：mjo_events_step3_1979-2022.csv

输出：
    - tilt_event_stats_1979-2022.csv：逐事件统计表
    - tilt_event_distribution.png：Tilt 分布直方图
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# PATHS
# ======================
TILT_NC = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
OUT_CSV = r"E:\Datas\Derived\tilt_event_stats_1979-2022.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_stats")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = FIG_DIR / "tilt_event_distribution.png"


def main():
    # 1. Load Data
    print(f"Loading tilt data: {TILT_NC}")
    if not Path(TILT_NC).exists():
        raise FileNotFoundError(f"File not found: {TILT_NC}")
    
    ds = xr.open_dataset(TILT_NC)
    tilt_da = ds["tilt"]
    
    print(f"Loading events: {EVENTS_CSV}")
    if not Path(EVENTS_CSV).exists():
        raise FileNotFoundError(f"File not found: {EVENTS_CSV}")
        
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    
    # 2. Iterate Events
    results = []
    
    # Pre-select time to optimize access
    # Convert 'time' coord to pd.DatetimeIndex for easy slicing
    times = pd.to_datetime(tilt_da["time"].values)
    tilt_vals = tilt_da.values
    
    # Ensure sorted by time for optimized search (usually true for NetCDF)
    # If not sorted, we might need a different approach, but usually climate data is sorted.
    
    print(f"Processing {len(events)} events...")
    
    for _, row in events.iterrows():
        eid = row["event_id"]
        s_date = row["start_date"]
        e_date = row["end_date"]
        
        # Find indices
        # searchsorted requires sorted array
        ts = np.datetime64(s_date)
        te = np.datetime64(e_date)
        
        # Simple boolean mask (robust)
        mask = (times >= ts) & (times <= te)
        
        if not np.any(mask):
            # Try converting to date-only if mismatch (though timestamps should match if from same pipeline)
            # But let's assume strict match first as per previous scripts
            print(f"Warning: No data found for event {eid} ({s_date} to {e_date})")
            continue
            
        event_tilts = tilt_vals[mask]
        
        # Valid values
        valid_tilts = event_tilts[np.isfinite(event_tilts)]
        
        if len(valid_tilts) > 0:
            mean_tilt = np.mean(valid_tilts)
            std_tilt = np.std(valid_tilts)
            median_tilt = np.median(valid_tilts)
            count = len(valid_tilts)
            min_tilt = np.min(valid_tilts)
            max_tilt = np.max(valid_tilts)
        else:
            mean_tilt = np.nan
            std_tilt = np.nan
            median_tilt = np.nan
            count = 0
            min_tilt = np.nan
            max_tilt = np.nan
            
        results.append({
            "event_id": eid,
            "start_date": s_date,
            "end_date": e_date,
            "mean_tilt": mean_tilt,
            "std_tilt": std_tilt,
            "median_tilt": median_tilt,
            "count": count,
            "min_tilt": min_tilt,
            "max_tilt": max_tilt,
            "duration_days": row.get("duration_days", (e_date - s_date).days + 1)
        })

    df_res = pd.DataFrame(results)
    
    # 3. Overall Statistics (Event-based)
    # We filter out events that had 0 valid tilt days (if any)
    df_valid = df_res.dropna(subset=["mean_tilt"])
    
    print("\n" + "="*40)
    print("OVERALL TILT STATISTICS (Event-Mean)")
    print("="*40)
    print(f"Total events analyzed: {len(df_res)}")
    print(f"Events with valid tilt: {len(df_valid)}")
    
    if len(df_valid) == 0:
        print("No valid events found.")
        return

    stats = df_valid["mean_tilt"].describe()
    print(stats)
    print("-" * 20)
    print(f"Overall Mean of Event Means: {df_valid['mean_tilt'].mean():.4f} deg")
    print(f"Standard Deviation of Means: {df_valid['mean_tilt'].std():.4f} deg")
    print(f"Standard Error of Mean:      {df_valid['mean_tilt'].sem():.4f} deg")
    
    # 4. Save CSV
    df_res.to_csv(OUT_CSV, index=False)
    print(f"\nSaved per-event stats to: {OUT_CSV}")
    
    # 5. Plot Distribution
    plt.figure(figsize=(10, 6))
    
    # Histogram of Event Means
    vals = df_valid["mean_tilt"]
    n, bins, patches = plt.hist(vals, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add vertical line for overall mean
    overall_mean = vals.mean()
    plt.axvline(overall_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {overall_mean:.2f}°')
    plt.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    plt.title(f"Distribution of Mean Tilt per MJO Event (N={len(df_valid)})")
    plt.xlabel("Mean Tilt (degrees, positive = bottom-east/top-west)")
    plt.ylabel("Number of Events")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Add text stats
    stats_text = (
        f"Mean: {overall_mean:.2f}\n"
        f"Std:  {vals.std():.2f}\n"
        f"Min:  {vals.min():.2f}\n"
        f"Max:  {vals.max():.2f}"
    )
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150)
    print(f"Saved distribution figure to: {OUT_FIG}")
    plt.close()

if __name__ == "__main__":
    main()
