# -*- coding: utf-8 -*-
"""
plot_daily_q_centroid_distribution.py
绘制逐日水汽重心（q centroid）相对经度分布直方图。

与 event_mean_q_centroid_distribution.png（事件平均场）对应，
本脚本对每一天独立计算低层 q 重心，再汇总绘制分布。

输出：
    outputs/figures/tilt_q_diagnose/daily_q_centroid_distribution.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import Akima1DInterpolator
from pathlib import Path

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS (与 03b_diagnose_tilt_q.py 保持一致)
# ======================
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
Q_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_diagnose")

# ======================
# SETTINGS (与主脚本保持一致)
# ======================
SMOOTH_WINDOW = 10
REL_LON_RANGE = (-90, 90)
Q_LOW_LAYER = (1000.0, 850.0)


def _smooth_1d(profile, window):
    if window <= 1:
        return profile
    kernel = np.ones(window) / window
    valid = np.isfinite(profile).astype(float)
    filled = np.where(np.isfinite(profile), profile, 0.0)
    smoothed = np.convolve(filled, kernel, mode='same')
    count = np.convolve(valid, kernel, mode='same')
    count[count < 1e-10] = np.nan
    return smoothed / count


def _cubic_spline_interp_1d(src_lon, profile, target_lon):
    valid = np.isfinite(profile)
    if valid.sum() < 4:
        return np.full(len(target_lon), np.nan)
    f = Akima1DInterpolator(src_lon[valid], profile[valid])
    return f(target_lon)


def _find_q_centroid(rel_lon, q_profile):
    """计算 q>0 区域的水汽重心位置。"""
    m = np.isfinite(q_profile) & np.isfinite(rel_lon) & (q_profile > 0)
    if m.sum() < 3:
        return np.nan
    rr = rel_lon[m].astype(float)
    qq = q_profile[m].astype(float)
    q_sum = np.sum(qq)
    if q_sum < 1e-20:
        return np.nan
    return float(np.sum(qq * rr) / q_sum)


def main():
    print("=" * 60)
    print("绘制逐日水汽重心分布直方图")
    print("=" * 60)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- 加载数据 ---
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ds_q = xr.open_dataset(Q_NORM_NC)
    ds3 = xr.open_dataset(STEP3_NC)

    q_raw = ds_q['q_mjo_recon_norm'].values          # (time, level, lon)
    levels_q = (ds_q['pressure_level'].values
                if 'pressure_level' in ds_q else ds_q['level'].values)
    lon_q = ds_q['lon'].values
    time_arr = pd.to_datetime(ds_q['time'].values)
    center_lon = ds3['center_lon_track'].values.astype(float)

    # 转 0-360 并排序
    lon_q_360 = np.where(lon_q < 0, lon_q + 360, lon_q)
    q_sort = np.argsort(lon_q_360)
    lon_q_360 = lon_q_360[q_sort]

    # 低层选取
    low_mask = (levels_q >= Q_LOW_LAYER[1]) & (levels_q <= Q_LOW_LAYER[0])
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + 0.25, 0.25)

    # --- 逐日计算 centroid ---
    daily_centroids = []
    n_events = len(events)

    for ev_idx, row in events.iterrows():
        eid = int(row['event_id'])
        ts = pd.Timestamp(row['start_date'])
        te = pd.Timestamp(row['end_date'])

        event_mask = (time_arr >= ts) & (time_arr <= te)
        event_indices = np.where(event_mask)[0]

        for idx in event_indices:
            c = center_lon[idx]
            if not np.isfinite(c):
                continue

            rel_lon = lon_q_360 - c
            mask = (rel_lon >= REL_LON_RANGE[0]) & (rel_lon <= REL_LON_RANGE[1])
            rel_lons = rel_lon[mask]

            q_day = q_raw[idx, :, :][:, q_sort][:, mask]
            q_low_mean = np.nanmean(q_day[low_mask, :], axis=0)

            # 插值 + 平滑
            q_interp = _cubic_spline_interp_1d(rel_lons, q_low_mean, target_rel)
            q_smooth = _smooth_1d(q_interp, SMOOTH_WINDOW)

            centroid = _find_q_centroid(target_rel, q_smooth)
            if np.isfinite(centroid):
                daily_centroids.append(centroid)

        if (ev_idx + 1) % 20 == 0:
            print(f"  已处理 {ev_idx + 1}/{n_events} 个事件...")

    vals = np.array(daily_centroids)
    print(f"  有效天数 = {len(vals)}")

    # --- 绘图（风格与 event_mean_q_centroid_distribution 一致）---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=50, color='#2980B9', edgecolor='black', alpha=0.7)
    mean_val = np.mean(vals)
    median_val = np.median(vals)
    ax.axvline(mean_val, color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}°')
    ax.axvline(median_val, color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.1f}°')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Convective Center')

    ax.set_xlabel("Lower-Level q Centroid (Relative Longitude, °)", fontsize=12)
    ax.set_ylabel("Count (days)", fontsize=12)
    ax.set_title(f"Daily Lower-Level q Centroid Distribution (N={len(vals)})",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    stats_text = (
        f"Mean: {mean_val:.2f}°\n"
        f"Median: {median_val:.2f}°\n"
        f"Std: {np.std(vals):.2f}°\n"
        f"Min: {np.min(vals):.2f}°\n"
        f"Max: {np.max(vals):.2f}°"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

    plt.tight_layout()
    out = FIG_DIR / "daily_q_centroid_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
