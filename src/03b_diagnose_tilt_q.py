# -*- coding: utf-8 -*-
"""
03b_diagnose_tilt_q.py: Tilt_q 诊断可视化

================================================================================
功能描述：
    1. 图1: 所有事件日的上层 omega 西边界相对经度分布直方图
    2. 图2: 所有事件日的下层 q 最大值相对经度分布直方图
    3. 图3: 逐事件平均 omega/q 场合并剖面图 + 散点标注（115 张）
    4. 图4: 逐事件逐日上层/下层经度分布折线图（115 张）

输入数据：
    - tilt_q_daily_1979-2022.nc（预计算的逐日 tilt_q、up_west_rel、q_max_rel）
    - mjo_events_step3_1979-2022.csv（事件列表）
    - era5_mjo_recon_w_norm_1979-2022.nc（归一化 omega 场）
    - era5_mjo_recon_q_norm_1979-2022.nc（归一化 q 场）
    - mjo_mvEOF_step3_1979-2022.nc（对流中心轨迹）

输出：
    - outputs/figures/tilt_q_diagnose/daily_up_west_distribution.png
    - outputs/figures/tilt_q_diagnose/daily_q_max_distribution.png
    - outputs/figures/tilt_q_diagnose/event_profile/event_NNN_profile.png  (×115)
    - outputs/figures/tilt_q_diagnose/event_lon_series/event_NNN_lon_series.png  (×115)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import Akima1DInterpolator, interp1d
from pathlib import Path

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
TILT_Q_NC  = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
W_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
Q_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"

FIG_DIR    = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_diagnose")
PROFILE_DIR = FIG_DIR / "event_profile"
LON_SERIES_DIR = FIG_DIR / "event_lon_series"
CENTROID_PROFILE_DIR = FIG_DIR / "centroid_profile"

# ======================
# SETTINGS
# ======================
SMOOTH_WINDOW = 10
CSA_TARGET_DLON = 0.25
REL_LON_RANGE = (-90, 90)

# q 最大值搜索范围：对流中心前后 90°（共 180°）
Q_MAX_SEARCH_RANGE = (-90, 90)

# 层次定义
UP_LAYER = (400.0, 200.0)       # omega 高层
Q_LOW_LAYER = (1000.0, 850.0)   # q 低层

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}


# ======================
# 辅助函数
# ======================
def _smooth_1d(profile, window):
    """沿经度方向做滑动平均"""
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
    """单条剖面做 Akima 插值"""
    valid = np.isfinite(profile)
    if valid.sum() < 4:
        return np.full(len(target_lon), np.nan)
    f = Akima1DInterpolator(src_lon[valid], profile[valid])
    return f(target_lon)


def _ascent_boundary_zero(rel_lon, w_profile):
    """
    在高层平均 omega 剖面中，找 omega=0 的西边界点。
    从对流中心(rel=0)向西搜索，找到 omega >= 0 的第一个点。
    """
    m = np.isfinite(w_profile) & np.isfinite(rel_lon)
    if m.sum() < 7:
        return np.nan

    rr = rel_lon[m].astype(float)
    ww = w_profile[m].astype(float)

    pivot_idx = int(np.argmin(np.abs(rr)))
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return np.nan

    # 向西搜索 omega >= 0
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= 0:
            return float(rr[i])
    return np.nan


def _find_q_max(rel_lon, q_profile, search_min=-90, search_max=90):
    """
    在 q 低层剖面中找最大值。搜索范围限制在 [search_min, search_max]。
    """
    m = np.isfinite(q_profile) & np.isfinite(rel_lon)
    m = m & (rel_lon >= search_min) & (rel_lon <= search_max)
    if m.sum() < 7:
        return np.nan
    rr = rel_lon[m].astype(float)
    qq = q_profile[m].astype(float)
    max_idx = int(np.argmax(qq))
    return float(rr[max_idx])


def _find_q_centroid(rel_lon, q_profile):
    """
    计算 q>0 区域的水汽重心位置。
    centroid = Σ(q * rel_lon) / Σ(q)，仅对 q > 0 的格点。
    """
    m = np.isfinite(q_profile) & np.isfinite(rel_lon) & (q_profile > 0)
    if m.sum() < 3:
        return np.nan
    rr = rel_lon[m].astype(float)
    qq = q_profile[m].astype(float)
    q_sum = np.sum(qq)
    if q_sum < 1e-20:
        return np.nan
    return float(np.sum(qq * rr) / q_sum)


# ======================
# 图1: 每日上层西边界经度分布
# ======================
def plot_daily_up_west_distribution(ds_tilt_q, events, fig_dir):
    """绘制所有事件日的上层 omega 西边界相对经度分布直方图"""
    times = pd.to_datetime(ds_tilt_q["time"].values)
    up_west = ds_tilt_q["up_west_rel"].values.astype(float)

    # 筛选事件日
    vals = []
    for _, row in events.iterrows():
        ts = np.datetime64(row["start_date"])
        te = np.datetime64(row["end_date"])
        mask = (times >= ts) & (times <= te)
        v = up_west[mask]
        vals.extend(v[np.isfinite(v)].tolist())

    vals = np.array(vals)
    print(f"  图1: 有效点数 = {len(vals)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=50, color='#E74C3C', edgecolor='black', alpha=0.7)
    mean_val = np.mean(vals)
    median_val = np.median(vals)
    ax.axvline(mean_val, color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}°')
    ax.axvline(median_val, color='darkgreen', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.1f}°')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Convective Center')

    ax.set_xlabel("Upper-Level ω West Boundary (Relative Longitude, °)", fontsize=12)
    ax.set_ylabel("Count (days)", fontsize=12)
    ax.set_title(f"Daily Upper-Level ω West Boundary Distribution (N={len(vals)})",
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
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

    plt.tight_layout()
    out = fig_dir / "daily_up_west_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ======================
# 图2: 每日下层水汽最大值经度分布
# ======================
def plot_daily_q_max_distribution(ds_tilt_q, events, fig_dir):
    """绘制所有事件日的下层 q 最大值相对经度分布直方图"""
    times = pd.to_datetime(ds_tilt_q["time"].values)
    q_max_rel = ds_tilt_q["q_max_rel"].values.astype(float)

    vals = []
    for _, row in events.iterrows():
        ts = np.datetime64(row["start_date"])
        te = np.datetime64(row["end_date"])
        mask = (times >= ts) & (times <= te)
        v = q_max_rel[mask]
        vals.extend(v[np.isfinite(v)].tolist())

    vals = np.array(vals)
    print(f"  图2: 有效点数 = {len(vals)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=50, color='#27AE60', edgecolor='black', alpha=0.7)
    mean_val = np.mean(vals)
    median_val = np.median(vals)
    ax.axvline(mean_val, color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}°')
    ax.axvline(median_val, color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.1f}°')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Convective Center')

    ax.set_xlabel("Lower-Level q Max Position (Relative Longitude, °)", fontsize=12)
    ax.set_ylabel("Count (days)", fontsize=12)
    ax.set_title(f"Daily Lower-Level q Max Distribution (N={len(vals)})",
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
    out = fig_dir / "daily_q_max_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ======================
# 图1b: 每个事件平均上层西边界经度分布
# ======================
def plot_event_mean_up_west_distribution(ds_tilt_q, events, fig_dir):
    """以事件为单位，每个事件取平均 up_west_rel，绘制 115 个事件均值的分布直方图"""
    times = pd.to_datetime(ds_tilt_q["time"].values)
    up_west = ds_tilt_q["up_west_rel"].values.astype(float)

    event_means = []
    for _, row in events.iterrows():
        ts = np.datetime64(row["start_date"])
        te = np.datetime64(row["end_date"])
        mask = (times >= ts) & (times <= te)
        v = up_west[mask]
        v_valid = v[np.isfinite(v)]
        if len(v_valid) > 0:
            event_means.append(np.mean(v_valid))

    vals = np.array(event_means)
    print(f"  图1b: 有效事件数 = {len(vals)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=30, color='#E74C3C', edgecolor='black', alpha=0.7)
    mean_val = np.mean(vals)
    median_val = np.median(vals)
    ax.axvline(mean_val, color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}°')
    ax.axvline(median_val, color='darkgreen', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.1f}°')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Convective Center')

    ax.set_xlabel("Event-Mean Upper-Level ω West Boundary (Relative Longitude, °)", fontsize=12)
    ax.set_ylabel("Count (events)", fontsize=12)
    ax.set_title(f"Event-Mean Upper-Level ω West Boundary Distribution (N={len(vals)})",
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
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

    plt.tight_layout()
    out = fig_dir / "event_mean_up_west_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ======================
# 图2b: 每个事件平均下层水汽最大值经度分布
# ======================
def plot_event_mean_q_max_distribution(ds_tilt_q, events, fig_dir):
    """以事件为单位，每个事件取平均 q_max_rel，绘制 115 个事件均值的分布直方图"""
    times = pd.to_datetime(ds_tilt_q["time"].values)
    q_max_rel = ds_tilt_q["q_max_rel"].values.astype(float)

    event_means = []
    for _, row in events.iterrows():
        ts = np.datetime64(row["start_date"])
        te = np.datetime64(row["end_date"])
        mask = (times >= ts) & (times <= te)
        v = q_max_rel[mask]
        v_valid = v[np.isfinite(v)]
        if len(v_valid) > 0:
            event_means.append(np.mean(v_valid))

    vals = np.array(event_means)
    print(f"  图2b: 有效事件数 = {len(vals)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=30, color='#27AE60', edgecolor='black', alpha=0.7)
    mean_val = np.mean(vals)
    median_val = np.median(vals)
    ax.axvline(mean_val, color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}°')
    ax.axvline(median_val, color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.1f}°')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Convective Center')

    ax.set_xlabel("Event-Mean Lower-Level q Max Position (Relative Longitude, °)", fontsize=12)
    ax.set_ylabel("Count (events)", fontsize=12)
    ax.set_title(f"Event-Mean Lower-Level q Max Distribution (N={len(vals)})",
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
    out = fig_dir / "event_mean_q_max_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ======================
# 图: 事件均值水汽重心（centroid）经度分布
# ======================
def plot_event_centroid_distribution(centroid_vals, fig_dir):
    """绘制 115 个事件的水汽重心经度分布直方图"""
    vals = centroid_vals[np.isfinite(centroid_vals)]
    print(f"  图centroid分布: 有效事件数 = {len(vals)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=30, color='#2980B9', edgecolor='black', alpha=0.7)
    mean_val = np.mean(vals)
    median_val = np.median(vals)
    ax.axvline(mean_val, color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}°')
    ax.axvline(median_val, color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.1f}°')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Convective Center')

    ax.set_xlabel("Event-Mean q Centroid (Relative Longitude, °)", fontsize=12)
    ax.set_ylabel("Count (events)", fontsize=12)
    ax.set_title(f"Event-Mean Lower-Level q Centroid Distribution (N={len(vals)})",
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
    out = fig_dir / "event_mean_q_centroid_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ======================
# 图: 事件平均场剖面 + centroid 标注
# ======================
def plot_centroid_profile(event_id, w_mean, q_mean, target_rel, target_h,
                          mean_up_west, mean_q_centroid,
                          event_row, out_dir):
    """绘制事件平均场剖面图，下层用 centroid 替代 q max"""
    up_h_min = LEVEL_TO_HEIGHT[400]
    up_h_max = LEVEL_TO_HEIGHT[200]
    low_h_min = LEVEL_TO_HEIGHT[1000]
    low_h_max = LEVEL_TO_HEIGHT[850]
    up_h_mid = (up_h_min + up_h_max) / 2.0
    low_h_mid = (low_h_min + low_h_max) / 2.0

    fig, ax = plt.subplots(figsize=(14, 7))
    X, Y = np.meshgrid(target_rel, target_h)

    # omega 填色（400 hPa 以上）
    w_display = np.where((target_h >= up_h_min)[:, None], w_mean, np.nan)
    vmax_w = np.nanmax(np.abs(w_display)) * 0.8
    if vmax_w < 1e-6 or not np.isfinite(vmax_w):
        vmax_w = 0.01
    norm_w = TwoSlopeNorm(vmin=-vmax_w, vcenter=0, vmax=vmax_w)
    cf_w = ax.contourf(X, Y, w_display,
                       levels=np.linspace(-vmax_w, vmax_w, 21),
                       cmap='RdBu_r', norm=norm_w, extend='both', alpha=0.7)

    # q 填色（850 hPa 以下）
    q_display = np.where((target_h <= low_h_max)[:, None], q_mean, np.nan)
    vmax_q = np.nanmax(np.abs(q_display)) * 0.8
    if vmax_q < 1e-10 or not np.isfinite(vmax_q):
        vmax_q = 1e-5
    norm_q = TwoSlopeNorm(vmin=-vmax_q, vcenter=0, vmax=vmax_q)
    cf_q = ax.contourf(X, Y, q_display,
                       levels=np.linspace(-vmax_q, vmax_q, 21),
                       cmap='BrBG', norm=norm_q, extend='both', alpha=0.9)

    # omega=0 等值线
    w_contour = np.where((target_h >= up_h_min)[:, None], w_mean, np.nan)
    ax.contour(X, Y, w_contour, levels=[0], colors='black', linewidths=2.0)

    # 水平分隔线
    for h in [low_h_max, up_h_min]:
        ax.axhline(h, color='gray', lw=1.2, ls='-', alpha=0.6)

    # 层标签
    ax.text(REL_LON_RANGE[1] - 2, (low_h_min + low_h_max) / 2,
            'q (1000–850 hPa)', fontsize=9, color='darkgreen',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.text(REL_LON_RANGE[1] - 2, (up_h_min + up_h_max) / 2,
            'ω (400–200 hPa)', fontsize=9, color='darkred',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # 均值点 + 连线（centroid）
    if np.isfinite(mean_up_west) and np.isfinite(mean_q_centroid):
        tilt_val = mean_q_centroid - mean_up_west
        ax.plot([mean_up_west, mean_q_centroid], [up_h_mid, low_h_mid],
                'o-', color='gold', markersize=14, markeredgecolor='black',
                markeredgewidth=2, lw=3.5, zorder=10,
                label=f'Mean Tilt (centroid) = {tilt_val:.1f}°')

        ax.annotate(f'Mean ω west: {mean_up_west:.1f}°',
                    (mean_up_west, up_h_mid),
                    textcoords='offset points', xytext=(15, 10),
                    fontsize=10, color='darkgoldenrod', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=1.5))

        ax.annotate(f'q centroid: {mean_q_centroid:.1f}°',
                    (mean_q_centroid, low_h_mid),
                    textcoords='offset points', xytext=(15, -20),
                    fontsize=10, color='#2471A3', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#EBF5FB', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#2471A3', lw=1.5))

        mid_x = (mean_up_west + mean_q_centroid) / 2
        mid_y = (up_h_mid + low_h_mid) / 2
        ax.text(mid_x + 5, mid_y, f'Δlon = {tilt_val:.1f}°',
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                          edgecolor='gold', alpha=0.9))

    # 对流中心线
    ax.axvline(0, color='limegreen', lw=2.5, alpha=0.8, label='Convective Center')

    ax.set_ylim(0, 12)
    ax.set_xlim(REL_LON_RANGE)
    ax.set_ylabel('Height (km)', fontsize=12)
    ax.set_xlabel('Relative Longitude (°)', fontsize=12)

    # 右轴
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in pticks])
    ax2.set_yticklabels([str(p) for p in pticks])
    ax2.set_ylabel('Pressure (hPa)', fontsize=12)

    # colorbars
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax_w = inset_axes(ax, width='2%', height='45%', loc='upper right',
                       bbox_to_anchor=(0.14, 0.0, 1, 1), bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar_w = fig.colorbar(cf_w, cax=cax_w, orientation='vertical')
    cbar_w.set_label('omega (norm)', fontsize=8)

    cax_q = inset_axes(ax, width='2%', height='45%', loc='lower right',
                       bbox_to_anchor=(0.14, 0.0, 1, 1), bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar_q = fig.colorbar(cf_q, cax=cax_q, orientation='vertical')
    cbar_q.set_label('q (norm)', fontsize=8)

    eid = int(event_row['event_id'])
    title = (f"Event #{eid}: {event_row['start_date']} ~ {event_row['end_date']} "
             f"({int(event_row['duration_days'])}d) — Event-Mean Field (Centroid)")
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    out = out_dir / f"event_{eid:03d}_centroid_profile.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


# ======================
# 图3 & 图4: 逐事件剖面图 + 逐日经度分布
# ======================
def _prepare_field_data(ds_w, ds_q, ds3):
    """
    加载并预处理 omega 和 q 场数据。
    返回原始数据字典：raw omega (time, level, lon)、raw q (time, level, lon)、
    center_lon_track、时间坐标等。
    """
    w_raw = ds_w['w_mjo_recon_norm'].values     # (time, level, lon)
    q_raw = ds_q['q_mjo_recon_norm'].values
    levels_w = ds_w['pressure_level'].values if 'pressure_level' in ds_w else ds_w['level'].values
    levels_q = ds_q['pressure_level'].values if 'pressure_level' in ds_q else ds_q['level'].values
    lon_w = ds_w['lon'].values
    lon_q = ds_q['lon'].values
    time_w = pd.to_datetime(ds_w['time'].values)
    center_lon = ds3['center_lon_track'].values.astype(float)

    # 转 0-360
    lon_w_360 = np.where(lon_w < 0, lon_w + 360, lon_w)
    w_sort = np.argsort(lon_w_360)
    lon_w_360 = lon_w_360[w_sort]

    lon_q_360 = np.where(lon_q < 0, lon_q + 360, lon_q)
    q_sort = np.argsort(lon_q_360)
    lon_q_360 = lon_q_360[q_sort]

    return {
        'w_raw': w_raw, 'q_raw': q_raw,
        'levels_w': levels_w, 'levels_q': levels_q,
        'lon_w_360': lon_w_360, 'lon_q_360': lon_q_360,
        'w_sort': w_sort, 'q_sort': q_sort,
        'time': time_w,
        'center_lon': center_lon,
    }


def _process_single_day(data, idx, center):
    """
    处理单日的 omega 和 q 场：截取 ±90° 相对经度范围，插值 + 平滑，
    插值到均匀高度网格。

    返回: (w_h_interp, q_h_interp, target_rel, target_h) 或 None
    """
    rel_lon_w = data['lon_w_360'] - center
    mask_w = (rel_lon_w >= REL_LON_RANGE[0]) & (rel_lon_w <= REL_LON_RANGE[1])
    rel_lons_w = rel_lon_w[mask_w]

    w_day = data['w_raw'][idx, :, :][:, data['w_sort']][:, mask_w]
    heights_w = np.array([LEVEL_TO_HEIGHT[int(p)] for p in data['levels_w']])

    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    w_interp = np.full((len(data['levels_w']), len(target_rel)), np.nan)
    for k in range(len(data['levels_w'])):
        interped = _cubic_spline_interp_1d(rel_lons_w, w_day[k, :], target_rel)
        w_interp[k, :] = _smooth_1d(interped, SMOOTH_WINDOW)

    target_h = np.linspace(0.0, 12.0, 120)
    w_h = np.full((len(target_h), len(target_rel)), np.nan)
    for j in range(len(target_rel)):
        col = w_interp[:, j]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            f = interp1d(heights_w[valid], col[valid], kind='linear',
                         bounds_error=False, fill_value=np.nan)
            w_h[:, j] = f(target_h)

    # q
    rel_lon_q = data['lon_q_360'] - center
    mask_q = (rel_lon_q >= REL_LON_RANGE[0]) & (rel_lon_q <= REL_LON_RANGE[1])
    rel_lons_q = rel_lon_q[mask_q]

    q_day = data['q_raw'][idx, :, :][:, data['q_sort']][:, mask_q]
    heights_q = np.array([LEVEL_TO_HEIGHT[int(p)] for p in data['levels_q']])

    q_interp = np.full((len(data['levels_q']), len(target_rel)), np.nan)
    for k in range(len(data['levels_q'])):
        interped = _cubic_spline_interp_1d(rel_lons_q, q_day[k, :], target_rel)
        q_interp[k, :] = _smooth_1d(interped, SMOOTH_WINDOW)

    q_h = np.full((len(target_h), len(target_rel)), np.nan)
    for j in range(len(target_rel)):
        col = q_interp[:, j]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            f = interp1d(heights_q[valid], col[valid], kind='linear',
                         bounds_error=False, fill_value=np.nan)
            q_h[:, j] = f(target_h)

    return w_h, q_h, target_rel, target_h


def _compute_event_daily_points(data, event_indices, centers):
    """
    计算事件内每日的上层西边界点和下层 q 最大值点。
    使用事件平均场来获取层平均剖面，然后逐日计算。

    返回: up_west_rels, q_max_rels (列表，与 event_indices 对齐)
    """
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)

    up_west_rels = []
    q_max_rels = []

    for i, (idx, c) in enumerate(zip(event_indices, centers)):
        if not np.isfinite(c):
            up_west_rels.append(np.nan)
            q_max_rels.append(np.nan)
            continue

        # omega 高层层平均
        rel_lon_w = data['lon_w_360'] - c
        mask_w = (rel_lon_w >= REL_LON_RANGE[0]) & (rel_lon_w <= REL_LON_RANGE[1])
        rel_lons_w = rel_lon_w[mask_w]

        w_day = data['w_raw'][idx, :, :][:, data['w_sort']][:, mask_w]
        heights_w = np.array([LEVEL_TO_HEIGHT[int(p)] for p in data['levels_w']])

        # 选取 400-200 hPa 层
        up_mask = (data['levels_w'] >= UP_LAYER[1]) & (data['levels_w'] <= UP_LAYER[0])
        w_up_mean = np.nanmean(w_day[up_mask, :], axis=0)

        # 插值 + 平滑
        w_up_interp = _cubic_spline_interp_1d(rel_lons_w, w_up_mean, target_rel)
        w_up_smooth = _smooth_1d(w_up_interp, SMOOTH_WINDOW)

        uw = _ascent_boundary_zero(target_rel, w_up_smooth)
        up_west_rels.append(uw)

        # q 低层层平均
        rel_lon_q = data['lon_q_360'] - c
        mask_q = (rel_lon_q >= REL_LON_RANGE[0]) & (rel_lon_q <= REL_LON_RANGE[1])
        rel_lons_q = rel_lon_q[mask_q]

        q_day = data['q_raw'][idx, :, :][:, data['q_sort']][:, mask_q]
        low_mask = (data['levels_q'] >= Q_LOW_LAYER[1]) & (data['levels_q'] <= Q_LOW_LAYER[0])
        q_low_mean = np.nanmean(q_day[low_mask, :], axis=0)

        q_low_interp = _cubic_spline_interp_1d(rel_lons_q, q_low_mean, target_rel)
        q_low_smooth = _smooth_1d(q_low_interp, SMOOTH_WINDOW)

        qm = _find_q_max(target_rel, q_low_smooth,
                         Q_MAX_SEARCH_RANGE[0], Q_MAX_SEARCH_RANGE[1])
        q_max_rels.append(qm)

    return np.array(up_west_rels), np.array(q_max_rels)


def plot_event_profile(event_id, w_mean, q_mean, target_rel, target_h,
                       up_west_rels, q_max_rels, mean_up_west, mean_q_max,
                       event_row, out_dir):
    """
    图3: 绘制事件平均场合并剖面图，标注逐日散点和均值点。
    """
    up_h_min = LEVEL_TO_HEIGHT[400]   # 7.2
    up_h_max = LEVEL_TO_HEIGHT[200]   # 12.0
    low_h_min = LEVEL_TO_HEIGHT[1000] # 0.1
    low_h_max = LEVEL_TO_HEIGHT[850]  # 1.5
    up_h_mid = (up_h_min + up_h_max) / 2.0
    low_h_mid = (low_h_min + low_h_max) / 2.0

    fig, ax = plt.subplots(figsize=(14, 7))
    X, Y = np.meshgrid(target_rel, target_h)

    # omega 填色（400 hPa 以上）
    w_display = np.where((target_h >= up_h_min)[:, None], w_mean, np.nan)
    vmax_w = np.nanmax(np.abs(w_display)) * 0.8
    if vmax_w < 1e-6 or not np.isfinite(vmax_w):
        vmax_w = 0.01
    norm_w = TwoSlopeNorm(vmin=-vmax_w, vcenter=0, vmax=vmax_w)
    cf_w = ax.contourf(X, Y, w_display,
                       levels=np.linspace(-vmax_w, vmax_w, 21),
                       cmap='RdBu_r', norm=norm_w, extend='both', alpha=0.7)

    # q 填色（850 hPa 以下）
    q_display = np.where((target_h <= low_h_max)[:, None], q_mean, np.nan)
    vmax_q = np.nanmax(np.abs(q_display)) * 0.8
    if vmax_q < 1e-10 or not np.isfinite(vmax_q):
        vmax_q = 1e-5
    norm_q = TwoSlopeNorm(vmin=-vmax_q, vcenter=0, vmax=vmax_q)
    cf_q = ax.contourf(X, Y, q_display,
                       levels=np.linspace(-vmax_q, vmax_q, 21),
                       cmap='BrBG', norm=norm_q, extend='both', alpha=0.9)

    # omega=0 等值线
    w_contour = np.where((target_h >= up_h_min)[:, None], w_mean, np.nan)
    ax.contour(X, Y, w_contour, levels=[0], colors='black', linewidths=2.0)

    # 水平分隔线
    for h in [low_h_max, up_h_min]:
        ax.axhline(h, color='gray', lw=1.2, ls='-', alpha=0.6)

    # 层标签
    ax.text(REL_LON_RANGE[1] - 2, (low_h_min + low_h_max) / 2,
            'q (1000–850 hPa)', fontsize=9, color='darkgreen',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.text(REL_LON_RANGE[1] - 2, (up_h_min + up_h_max) / 2,
            'ω (400–200 hPa)', fontsize=9, color='darkred',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # 逐日散点：上层西边界（在 up_h_mid 高度处散布）
    valid_uw = up_west_rels[np.isfinite(up_west_rels)]
    if len(valid_uw) > 0:
        # 给散点纵向加小扰动，避免重叠
        jitter_up = np.random.default_rng(42).uniform(-0.3, 0.3, len(valid_uw))
        ax.scatter(valid_uw, up_h_mid + jitter_up,
                   c='red', s=30, alpha=0.5, zorder=8, edgecolors='darkred',
                   linewidths=0.5, label=f'Daily ω west (N={len(valid_uw)})')

    # 逐日散点：下层 q 最大值（在 low_h_mid 高度处散布）
    valid_qm = q_max_rels[np.isfinite(q_max_rels)]
    if len(valid_qm) > 0:
        jitter_low = np.random.default_rng(43).uniform(-0.2, 0.2, len(valid_qm))
        ax.scatter(valid_qm, low_h_mid + jitter_low,
                   c='limegreen', s=30, alpha=0.5, zorder=8, edgecolors='darkgreen',
                   linewidths=0.5, label=f'Daily q max (N={len(valid_qm)})')

    # 均值点 + 连线
    if np.isfinite(mean_up_west) and np.isfinite(mean_q_max):
        tilt_val = mean_q_max - mean_up_west
        ax.plot([mean_up_west, mean_q_max], [up_h_mid, low_h_mid],
                'o-', color='gold', markersize=14, markeredgecolor='black',
                markeredgewidth=2, lw=3.5, zorder=10,
                label=f'Mean Tilt_q = {tilt_val:.1f}°')

        ax.annotate(f'Mean ω west: {mean_up_west:.1f}°',
                    (mean_up_west, up_h_mid),
                    textcoords='offset points', xytext=(15, 10),
                    fontsize=10, color='darkgoldenrod', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=1.5))

        ax.annotate(f'Mean q max: {mean_q_max:.1f}°',
                    (mean_q_max, low_h_mid),
                    textcoords='offset points', xytext=(15, -20),
                    fontsize=10, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='honeydew', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

        mid_x = (mean_up_west + mean_q_max) / 2
        mid_y = (up_h_mid + low_h_mid) / 2
        ax.text(mid_x + 5, mid_y, f'Δlon = {tilt_val:.1f}°',
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                          edgecolor='gold', alpha=0.9))

    # 对流中心线
    ax.axvline(0, color='limegreen', lw=2.5, alpha=0.8, label='Convective Center')

    ax.set_ylim(0, 12)
    ax.set_xlim(REL_LON_RANGE)
    ax.set_ylabel('Height (km)', fontsize=12)
    ax.set_xlabel('Relative Longitude (°)', fontsize=12)

    # 右轴
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in pticks])
    ax2.set_yticklabels([str(p) for p in pticks])
    ax2.set_ylabel('Pressure (hPa)', fontsize=12)

    # colorbars
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax_w = inset_axes(ax, width='2%', height='45%', loc='upper right',
                       bbox_to_anchor=(0.14, 0.0, 1, 1), bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar_w = fig.colorbar(cf_w, cax=cax_w, orientation='vertical')
    cbar_w.set_label('omega (norm)', fontsize=8)

    cax_q = inset_axes(ax, width='2%', height='45%', loc='lower right',
                       bbox_to_anchor=(0.14, 0.0, 1, 1), bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar_q = fig.colorbar(cf_q, cax=cax_q, orientation='vertical')
    cbar_q.set_label('q (norm)', fontsize=8)

    eid = int(event_row['event_id'])
    title = (f"Event #{eid}: {event_row['start_date']} ~ {event_row['end_date']} "
             f"({int(event_row['duration_days'])}d) — Event-Mean Field\n"
             f"Window={SMOOTH_WINDOW}  |  q search: [{Q_MAX_SEARCH_RANGE[0]}, {Q_MAX_SEARCH_RANGE[1]}]°")
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    out = out_dir / f"event_{eid:03d}_profile.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


def plot_event_lon_series(event_id, up_west_rels, q_max_rels, dates, event_row, out_dir):
    """
    图4: 绘制逐日上层/下层经度随时间变化的折线图。
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    day_idx = np.arange(len(dates))
    date_labels = [d.strftime('%m-%d') for d in dates]

    # 上层西边界
    ax.plot(day_idx, up_west_rels, 'rs-', markersize=5, linewidth=1.5,
            label='Upper ω west boundary', alpha=0.8)
    # 下层 q 最大值
    ax.plot(day_idx, q_max_rels, 'g^-', markersize=5, linewidth=1.5,
            label='Lower q max', alpha=0.8)

    # 均值线
    mean_uw = np.nanmean(up_west_rels)
    mean_qm = np.nanmean(q_max_rels)
    if np.isfinite(mean_uw):
        ax.axhline(mean_uw, color='red', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'Mean ω west: {mean_uw:.1f}°')
    if np.isfinite(mean_qm):
        ax.axhline(mean_qm, color='green', linestyle='--', linewidth=1, alpha=0.5,
                   label=f'Mean q max: {mean_qm:.1f}°')

    ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.4)

    # x 轴设置
    if len(day_idx) > 20:
        step = max(1, len(day_idx) // 15)
        ax.set_xticks(day_idx[::step])
        ax.set_xticklabels(date_labels[::step], rotation=45, ha='right')
    else:
        ax.set_xticks(day_idx)
        ax.set_xticklabels(date_labels, rotation=45, ha='right')

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Relative Longitude (°)", fontsize=12)

    eid = int(event_row['event_id'])
    tilt_mean = mean_qm - mean_uw if (np.isfinite(mean_qm) and np.isfinite(mean_uw)) else np.nan
    title_str = (f"Event #{eid}: {event_row['start_date']} ~ {event_row['end_date']} "
                 f"({int(event_row['duration_days'])}d)")
    if np.isfinite(tilt_mean):
        title_str += f"  |  Mean Tilt_q = {tilt_mean:.1f}°"
    ax.set_title(title_str, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / f"event_{eid:03d}_lon_series.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()


# ======================
# 主函数
# ======================
def main():
    print("=" * 60)
    print("03b_diagnose_tilt_q.py: Tilt_q 诊断可视化")
    print("=" * 60)

    # 创建图片输出目录
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    LON_SERIES_DIR.mkdir(parents=True, exist_ok=True)
    CENTROID_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    # --- 加载数据 ---
    print("Loading data...")
    ds_tilt_q = xr.open_dataset(TILT_Q_NC)
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    print(f"  Events: {len(events)}")

    # ========================
    # 图1: 每日上层西边界分布
    # ========================
    print("\n[图1] 每日上层西边界经度分布...")
    plot_daily_up_west_distribution(ds_tilt_q, events, FIG_DIR)

    # ========================
    # 图2: 每日下层 q 最大值分布
    # ========================
    print("\n[图2] 每日下层 q 最大值经度分布...")
    plot_daily_q_max_distribution(ds_tilt_q, events, FIG_DIR)

    # ========================
    # 图1b: 事件均值上层西边界分布
    # ========================
    print("\n[图1b] 事件均值上层西边界经度分布...")
    plot_event_mean_up_west_distribution(ds_tilt_q, events, FIG_DIR)

    # ========================
    # 图2b: 事件均值下层 q 最大值分布
    # ========================
    print("\n[图2b] 事件均值下层 q 最大值经度分布...")
    plot_event_mean_q_max_distribution(ds_tilt_q, events, FIG_DIR)

    # ========================
    # 图3 & 图4: 逐事件剖面 + 逐日分布
    # ========================
    print("\n[图3-5] 加载原始场数据...")
    ds_w = xr.open_dataset(W_NORM_NC)
    ds_q = xr.open_dataset(Q_NORM_NC)
    ds3 = xr.open_dataset(STEP3_NC)

    data = _prepare_field_data(ds_w, ds_q, ds3)
    time_arr = data['time']

    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    target_h = np.linspace(0.0, 12.0, 120)

    # 收集每个事件的平均场值
    all_centroids = []
    all_mean_up_wests = []

    for ev_idx, row in events.iterrows():
        eid = int(row['event_id'])
        ts = pd.Timestamp(row['start_date'])
        te = pd.Timestamp(row['end_date'])

        # 找到事件对应的时间索引
        event_time_mask = (time_arr >= ts) & (time_arr <= te)
        event_indices = np.where(event_time_mask)[0]
        event_dates = time_arr[event_time_mask]

        if len(event_indices) < 2:
            print(f"  Event #{eid}: too short ({len(event_indices)} days), skip")
            all_centroids.append(np.nan)
            all_mean_up_wests.append(np.nan)
            continue

        centers = data['center_lon'][event_indices]
        valid_center = np.isfinite(centers)

        if valid_center.sum() < 2:
            print(f"  Event #{eid}: insufficient valid centers, skip")
            all_centroids.append(np.nan)
            all_mean_up_wests.append(np.nan)
            continue

        # --- 计算事件平均场 ---
        w_h_list = []
        q_h_list = []
        for i, idx in enumerate(event_indices):
            c = centers[i]
            if not np.isfinite(c):
                continue
            result = _process_single_day(data, idx, c)
            if result is not None:
                w_h, q_h, _, _ = result
                w_h_list.append(w_h)
                q_h_list.append(q_h)

        if len(w_h_list) < 2:
            print(f"  Event #{eid}: insufficient valid days for mean field, skip")
            all_centroids.append(np.nan)
            all_mean_up_wests.append(np.nan)
            continue

        w_mean = np.nanmean(np.stack(w_h_list), axis=0)
        q_mean = np.nanmean(np.stack(q_h_list), axis=0)

        # --- 计算逐日上层/下层经度 ---
        up_west_rels, q_max_rels = _compute_event_daily_points(
            data, event_indices, centers
        )

        mean_q_max = float(np.nanmean(q_max_rels))

        # --- 从平均场上计算上层西边界 ---
        up_h_min = LEVEL_TO_HEIGHT[400]   # 7.2
        up_h_mask = target_h >= up_h_min
        w_up_profile = np.nanmean(w_mean[up_h_mask, :], axis=0)  # (lon,)
        mean_up_west = _ascent_boundary_zero(target_rel, w_up_profile)
        all_mean_up_wests.append(mean_up_west)

        # --- 从平均场上计算 q centroid ---
        low_h_max = LEVEL_TO_HEIGHT[850]  # 1.5
        h_mask = target_h <= low_h_max
        q_low_profile = np.nanmean(q_mean[h_mask, :], axis=0)  # (lon,)
        q_centroid = _find_q_centroid(target_rel, q_low_profile)
        all_centroids.append(q_centroid)

        # --- 从平均场上计算 q max ---
        mean_q_max_field = _find_q_max(target_rel, q_low_profile,
                                        Q_MAX_SEARCH_RANGE[0], Q_MAX_SEARCH_RANGE[1])

        # --- 图3: 剖面图（q_max 版） ---
        plot_event_profile(eid, w_mean, q_mean, target_rel, target_h,
                           up_west_rels, q_max_rels, mean_up_west, mean_q_max_field,
                           row, PROFILE_DIR)

        # --- 图4: 逐日经度分布 ---
        plot_event_lon_series(eid, up_west_rels, q_max_rels,
                              event_dates, row, LON_SERIES_DIR)

        # --- 图5: centroid 剖面图 ---
        plot_centroid_profile(eid, w_mean, q_mean, target_rel, target_h,
                              mean_up_west, q_centroid,
                              row, CENTROID_PROFILE_DIR)

        print(f"  Event #{eid} done (days={len(event_indices)}, "
              f"field_uw={mean_up_west:.1f}°, field_qm={mean_q_max_field:.1f}°, "
              f"centroid={q_centroid:.1f}°)")

    # --- centroid 分布直方图 ---
    all_centroids = np.array(all_centroids)
    print("\n[图Centroid] 水汽重心经度分布...")
    plot_event_centroid_distribution(all_centroids, FIG_DIR)

    # --- 平均场 up_west 分布直方图 ---
    all_mean_up_wests = np.array(all_mean_up_wests)
    vals_uw = all_mean_up_wests[np.isfinite(all_mean_up_wests)]
    print(f"\n[图UpWest平均场] 平均场上层西边界分布 (N={len(vals_uw)})...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals_uw, bins=30, color='#E74C3C', edgecolor='black', alpha=0.7)
    mean_val = np.mean(vals_uw)
    median_val = np.median(vals_uw)
    ax.axvline(mean_val, color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}°')
    ax.axvline(median_val, color='darkgreen', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.1f}°')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Convective Center')
    ax.set_xlabel("Mean-Field Upper ω West Boundary (Relative Longitude, °)", fontsize=12)
    ax.set_ylabel("Count (events)", fontsize=12)
    ax.set_title(f"Mean-Field Upper ω West Boundary Distribution (N={len(vals_uw)})",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    stats_text = (
        f"Mean: {mean_val:.2f}°\n"
        f"Median: {median_val:.2f}°\n"
        f"Std: {np.std(vals_uw):.2f}°\n"
        f"Min: {np.min(vals_uw):.2f}°\n"
        f"Max: {np.max(vals_uw):.2f}°"
    )
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)
    plt.tight_layout()
    out = FIG_DIR / "meanfield_up_west_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # --- 保存平均场值到 CSV 供相关性脚本使用 ---
    df_field = pd.DataFrame({
        'event_id': events['event_id'].values,
        'field_up_west': all_mean_up_wests,
        'field_centroid': all_centroids,
    })
    csv_out = FIG_DIR / "event_mean_field_values.csv"
    df_field.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    print(f"\n{'='*60}")
    print(f"All figures saved to: {FIG_DIR}")
    print("DONE")


if __name__ == "__main__":
    main()
