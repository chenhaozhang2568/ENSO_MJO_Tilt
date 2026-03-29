# -*- coding: utf-8 -*-
"""
03b_verify_tilt_q.py: 验证新 tilt_q 数据，生成可视化对比图

================================================================================
功能描述：
    1. 逐事件平均 tilt_q，与旧 tilt（omega 西边界定义）对比直方图
    2. 散点图：旧 tilt vs 新 tilt_q
    3. q 低层平均剖面示例：标注 q 最大值位置
    4. 典型事件逐日对比时间序列

输入数据：
    - 新 tilt_q：tilt_q_daily_1979-2022.nc
    - 旧 tilt：tilt_daily_step4_layermean_1979-2022.nc
    - MJO 事件列表：mjo_events_step3_1979-2022.csv

输出：
    - 验证图：outputs/figures/tilt_q/tilt_q_*.png
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# PATHS
# ======================
TILT_Q_NC = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"
TILT_OLD_NC = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_event_mean(ds, var_name, events):
    """计算逐事件平均 tilt。"""
    times = pd.to_datetime(ds["time"].values)
    vals = ds[var_name].values.astype(float)
    results = []

    for _, row in events.iterrows():
        eid = row["event_id"]
        ts = np.datetime64(row["start_date"])
        te = np.datetime64(row["end_date"])
        mask = (times >= ts) & (times <= te)

        if not np.any(mask):
            continue
        event_vals = vals[mask]
        valid = event_vals[np.isfinite(event_vals)]

        if len(valid) > 0:
            results.append({
                "event_id": eid,
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                f"mean_{var_name}": np.mean(valid),
                "count": len(valid),
            })
    return pd.DataFrame(results)


def plot_histogram_comparison(df_q, df_old, fig_dir):
    """逐事件平均 tilt 直方图对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: 新 tilt_q 直方图
    ax1 = axes[0]
    v1 = df_q["mean_tilt_q"].dropna()
    ax1.hist(v1, bins=25, color='#3498DB', edgecolor='black', alpha=0.7)
    mean1 = v1.mean()
    ax1.axvline(mean1, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean1:.2f}°')
    ax1.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_title(f"New Tilt_q (N={len(v1)})")
    ax1.set_xlabel("Event Mean Tilt_q (deg)")
    ax1.set_ylabel("Number of Events")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: 旧 tilt 直方图
    ax2 = axes[1]
    v2 = df_old["mean_tilt"].dropna()
    ax2.hist(v2, bins=25, color='#E74C3C', edgecolor='black', alpha=0.7)
    mean2 = v2.mean()
    ax2.axvline(mean2, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean2:.2f}°')
    ax2.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_title(f"Old Tilt (N={len(v2)})")
    ax2.set_xlabel("Event Mean Tilt (deg)")
    ax2.set_ylabel("Number of Events")
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle("Event-Mean Tilt Distribution: New (q-max) vs Old (ω west boundary)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = fig_dir / "tilt_q_histogram_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_scatter(df_merged, fig_dir):
    """散点图：旧 tilt vs 新 tilt_q。"""
    fig, ax = plt.subplots(figsize=(7, 7))

    x = df_merged["mean_tilt"]
    y = df_merged["mean_tilt_q"]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    ax.scatter(x, y, c='#2980B9', alpha=0.6, s=40, edgecolors='k', linewidths=0.5)

    # 1:1 line
    lim_min = min(x.min(), y.min()) - 5
    lim_max = max(x.max(), y.max()) + 5
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.4, label='1:1 line')

    # regression
    from scipy import stats
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2,
            label=f'r={r_val:.3f}, p={p_val:.4f}\ny={slope:.2f}x+{intercept:.2f}')

    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel("Old Tilt (ω west boundary, deg)", fontsize=12)
    ax.set_ylabel("New Tilt_q (q max position, deg)", fontsize=12)
    ax.set_title(f"Old Tilt vs New Tilt_q (N={len(x)})", fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = fig_dir / "tilt_q_vs_old_scatter.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_event_timeseries(ds_q, ds_old, events, fig_dir, n_events=4):
    """选取典型事件绘制逐日 tilt 对比时间序列。"""
    times_q = pd.to_datetime(ds_q["time"].values)
    times_old = pd.to_datetime(ds_old["time"].values)

    tilt_q_vals = ds_q["tilt_q"].values.astype(float)
    tilt_old_vals = ds_old["tilt"].values.astype(float)

    # 选取持续时间较长的事件作展示
    events_sorted = events.copy()
    events_sorted["duration_days"] = (
        pd.to_datetime(events_sorted["end_date"]) - pd.to_datetime(events_sorted["start_date"])
    ).dt.days + 1
    events_sorted = events_sorted.sort_values("duration_days", ascending=False).head(n_events)

    fig, axes = plt.subplots(n_events, 1, figsize=(12, 3 * n_events), sharex=False)
    if n_events == 1:
        axes = [axes]

    for idx, (_, row) in enumerate(events_sorted.iterrows()):
        ax = axes[idx]
        ts = pd.Timestamp(row["start_date"])
        te = pd.Timestamp(row["end_date"])

        # new tilt_q
        mask_q = (times_q >= ts) & (times_q <= te)
        t_q = times_q[mask_q]
        v_q = tilt_q_vals[mask_q]

        # old tilt
        mask_old = (times_old >= ts) & (times_old <= te)
        t_old = times_old[mask_old]
        v_old = tilt_old_vals[mask_old]

        ax.plot(t_q, v_q, 'b-o', markersize=3, label='New Tilt_q (q max)')
        ax.plot(t_old, v_old, 'r-s', markersize=3, label='Old Tilt (ω west)')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel("Tilt (deg)")
        ax.set_title(f"Event {row['event_id']}: {ts.strftime('%Y-%m-%d')} ~ {te.strftime('%Y-%m-%d')}")
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Event Time Series: New Tilt_q vs Old Tilt", fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = fig_dir / "tilt_q_event_timeseries.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def plot_q_max_position_histogram(ds_q, fig_dir):
    """q 最大值相对经度位置分布直方图。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    qr = ds_q["q_max_rel"].values.astype(float)
    qok = qr[np.isfinite(qr)]

    ax.hist(qok, bins=40, color='#27AE60', edgecolor='black', alpha=0.7)
    mean_val = np.mean(qok)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.2f}°')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5,
               label='Convective Center')

    ax.set_xlabel("q Max Relative Longitude (deg)", fontsize=12)
    ax.set_ylabel("Count (days)", fontsize=12)
    ax.set_title(f"Distribution of q Low-Level Max Position (N={len(qok)})",
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    stats_text = (
        f"Mean: {mean_val:.2f}°\n"
        f"Median: {np.median(qok):.2f}°\n"
        f"Std: {np.std(qok):.2f}°\n"
        f"Min: {np.min(qok):.2f}°\n"
        f"Max: {np.max(qok):.2f}°"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

    plt.tight_layout()
    out = fig_dir / "q_max_position_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()


def main():
    print("Loading data...")
    ds_q = xr.open_dataset(TILT_Q_NC)
    ds_old = xr.open_dataset(TILT_OLD_NC)
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])

    print(f"  tilt_q: {sum(np.isfinite(ds_q['tilt_q'].values))} valid days")
    print(f"  old tilt: {sum(np.isfinite(ds_old['tilt'].values))} valid days")
    print(f"  events: {len(events)}")

    # 1. Compute event means
    print("\nComputing event mean tilt_q...")
    df_q = compute_event_mean(ds_q, "tilt_q", events)
    print(f"  Events with valid tilt_q: {len(df_q)}")

    print("Computing event mean old tilt...")
    df_old = compute_event_mean(ds_old, "tilt", events)
    print(f"  Events with valid tilt: {len(df_old)}")

    # 2. Histogram comparison
    print("\nPlotting histogram comparison...")
    plot_histogram_comparison(df_q, df_old, FIG_DIR)

    # 3. Scatter: old vs new
    print("Plotting scatter comparison...")
    df_merged = df_q.merge(df_old, on="event_id", suffixes=("_q", "_old"))
    plot_scatter(df_merged, FIG_DIR)

    # 4. Event time series
    print("Plotting event time series...")
    plot_event_timeseries(ds_q, ds_old, events, FIG_DIR, n_events=4)

    # 5. q max position distribution
    print("Plotting q max position distribution...")
    plot_q_max_position_histogram(ds_q, FIG_DIR)

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    v_q = df_q["mean_tilt_q"].dropna()
    v_old = df_old["mean_tilt"].dropna()
    print(f"New tilt_q: mean={v_q.mean():.2f}°, std={v_q.std():.2f}°, N={len(v_q)}")
    print(f"Old tilt:   mean={v_old.mean():.2f}°, std={v_old.std():.2f}°, N={len(v_old)}")

    if len(df_merged) > 2:
        from scipy import stats
        x = df_merged["mean_tilt"].dropna()
        y = df_merged["mean_tilt_q"].dropna()
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 2:
            r, p = stats.pearsonr(x[valid], y[valid])
            print(f"Correlation: r={r:.3f}, p={p:.4f}")

    print(f"\nAll figures saved to: {FIG_DIR}")
    print("DONE")


if __name__ == "__main__":
    main()
