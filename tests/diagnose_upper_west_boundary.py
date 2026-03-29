# -*- coding: utf-8 -*-
"""
diagnose_upper_west_boundary.py — 高层omega西边界与相速度正相关问题诊断
（已改用逐日 up_west_rel 的事件内均值代替平均场 field_up_west）

目的：
    高层omega西边界经度理论上应与相速度负相关（西边界越偏西 → 倾斜越强 → 相速度越快），
    但实际结果显示正相关(p=0.0003)。本脚本通过6个猜想的系统性验证，找出原因。

猜想清单：
    H1: 离散聚类导致虚假相关（-81.25°极端值的杠杆效应）
    H2: MJO振幅混淆效应
    H3: 事件持续时间混淆效应
    H4: 不同经度位置的MJO系统性差异
    H5: EOF重构的非物理伪影(需原始数据,仅打印提示)
    H6: 边界检测的跳跃不连续 → 用上升区宽度替代

输出：
    outputs/figures/upper_west_diagnose/ 下的所有诊断图表

用法：
    python tests/diagnose_upper_west_boundary.py
"""

from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
FIELD_CSV   = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_diagnose\both_meanfield\event_mean_field_values.csv"
SPEED_CSV   = r"E:\Datas\Derived\phase_speed_q_events.csv"
TILT_Q_NC   = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"
EVENTS_CSV  = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
STEP3_NC    = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\upper_west_diagnose")


# ======================
# 数据加载
# ======================
def load_data():
    """加载并合并所有需要的事件级数据。"""
    df_field = pd.read_csv(FIELD_CSV)
    df_speed = pd.read_csv(SPEED_CSV)
    df_events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])

    # 合并
    df = df_speed.merge(df_field, on="event_id", how="left")
    df = df.merge(df_events[["event_id", "lon_start", "lon_end"]],
                  on="event_id", how="left")

    # 事件平均振幅
    ds3 = xr.open_dataset(STEP3_NC)
    amp_all = ds3["amp"].values.astype(float)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    time_all = pd.to_datetime(ds3["time"].values)

    # 逐事件平均 tilt_q
    ds_tilt_q = xr.open_dataset(TILT_Q_NC)
    tilt_q_all = ds_tilt_q["tilt_q"].values.astype(float)
    up_west_daily = ds_tilt_q["up_west_rel"].values.astype(float)
    up_east_daily = ds_tilt_q["up_east_rel"].values.astype(float)
    tilt_q_time = pd.to_datetime(ds_tilt_q["time"].values)

    event_amps = []
    event_center_lons = []
    event_tilt_qs = []
    event_up_west_daily_means = []
    event_ascent_widths = []

    for _, ev in df_events.iterrows():
        ts = pd.Timestamp(ev["start_date"])
        te = pd.Timestamp(ev["end_date"])

        # 振幅
        mask_s3 = (time_all >= ts) & (time_all <= te)
        a = amp_all[mask_s3]
        a_valid = a[np.isfinite(a)]
        event_amps.append(np.mean(a_valid) if len(a_valid) > 0 else np.nan)

        # 中心经度
        c = center_lon_all[mask_s3]
        c_valid = c[np.isfinite(c)]
        event_center_lons.append(np.mean(c_valid) if len(c_valid) > 0 else np.nan)

        # tilt_q
        mask_tq = (tilt_q_time >= ts) & (tilt_q_time <= te)
        tq = tilt_q_all[mask_tq]
        tq_valid = tq[np.isfinite(tq)]
        event_tilt_qs.append(np.mean(tq_valid) if len(tq_valid) > 0 else np.nan)

        # 逐日 up_west 均值
        uw = up_west_daily[mask_tq]
        uw_valid = uw[np.isfinite(uw)]
        event_up_west_daily_means.append(np.mean(uw_valid) if len(uw_valid) > 0 else np.nan)

        # 上升区宽度 = east - west
        ue = up_east_daily[mask_tq]
        width = ue - uw
        w_valid = width[np.isfinite(width)]
        event_ascent_widths.append(np.mean(w_valid) if len(w_valid) > 0 else np.nan)

    df_extra = pd.DataFrame({
        "event_id": df_events["event_id"].values,
        "mean_amp": event_amps,
        "mean_center_lon": event_center_lons,
        "mean_tilt_q": event_tilt_qs,
        "daily_up_west_mean": event_up_west_daily_means,
        "ascent_width": event_ascent_widths,
    })

    df = df.merge(df_extra, on="event_id", how="left")

    # 过滤有效行（改用逐日均值 up_west）
    valid = (np.isfinite(df["daily_up_west_mean"].values) &
             np.isfinite(df["phase_speed_m_s"].values))
    df = df[valid].reset_index(drop=True)

    print(f"有效事件数: {len(df)}")
    print(f"daily_up_west_mean 范围: [{df['daily_up_west_mean'].min():.1f}, {df['daily_up_west_mean'].max():.1f}]")
    print(f"phase_speed_m_s 范围: [{df['phase_speed_m_s'].min():.2f}, {df['phase_speed_m_s'].max():.2f}]")

    return df


# ======================
# 辅助绘图函数
# ======================
def _add_regression_line(ax, x, y, color='red', label_prefix=""):
    """添加回归线和统计信息。"""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return
    slope, intercept, r_val, p_val, _ = stats.linregress(x[ok], y[ok])
    x_line = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 100)
    ax.plot(x_line, slope * x_line + intercept, '-', color=color, lw=2)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
    label = f"{label_prefix}r={r_val:.3f}, p={p_val:.4f} {sig}, N={ok.sum()}"
    return r_val, p_val, label


def _partial_corr(x, y, z):
    """计算偏相关：控制 z 后 x 和 y 的相关。"""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return np.nan, np.nan, 0
    x, y, z = x[ok], y[ok], z[ok]
    # x 对 z 的残差
    _, _, r_xz, _, _ = stats.linregress(z, x)
    res_x = x - (stats.linregress(z, x).slope * z + stats.linregress(z, x).intercept)
    # y 对 z 的残差
    res_y = y - (stats.linregress(z, y).slope * z + stats.linregress(z, y).intercept)
    r, p = stats.pearsonr(res_x, res_y)
    return r, p, ok.sum()


# ======================
# H1: 离散聚类 + 杠杆效应
# ======================
def diagnose_h1(df):
    """H1: 检查离散聚类和极端值的杠杆效应。"""
    print("\n" + "="*60)
    print("H1: 离散聚类与杠杆效应诊断")
    print("="*60)

    up_west = df["daily_up_west_mean"].values
    speed = df["phase_speed_m_s"].values

    # 识别聚类
    extreme_mask = up_west <= -70  # -81.25 等极端偏西
    moderate_mask = (up_west > -70) & (up_west <= -45)
    normal_mask = up_west > -45

    n_extreme = extreme_mask.sum()
    n_moderate = moderate_mask.sum()
    n_normal = normal_mask.sum()
    print(f"  极端偏西 (<-70°): {n_extreme} 事件")
    print(f"  中间偏西 (-70~-45°): {n_moderate} 事件")
    print(f"  正常范围 (>-45°): {n_normal} 事件")

    # 各组相速度统计
    for name, mask in [("极端偏西", extreme_mask), ("中间", moderate_mask), ("正常", normal_mask)]:
        spd = speed[mask]
        if len(spd) > 0:
            print(f"  {name}组: speed mean={np.nanmean(spd):.2f}, "
                  f"std={np.nanstd(spd):.2f}, N={len(spd)}")

    # --- D1: 聚类标注散点图 ---
    fig, ax = plt.subplots(figsize=(9, 7))

    colors = {'极端偏西\n(≤-70°)': '#E74C3C', '中间\n(-70~-45°)': '#F39C12',
              '正常\n(>-45°)': '#3498DB'}
    masks = [extreme_mask, moderate_mask, normal_mask]
    labels = list(colors.keys())

    for mask, label, color in zip(masks, labels, colors.values()):
        if mask.sum() > 0:
            ax.scatter(up_west[mask], speed[mask], c=color, s=60, alpha=0.7,
                       edgecolors='k', linewidths=0.5, label=f"{label} (N={mask.sum()})",
                       zorder=5)

    info = _add_regression_line(ax, up_west, speed, color='black')
    if info:
        r_all, p_all, label_all = info
        ax.text(0.02, 0.98, f"全部: {label_all}", transform=ax.transAxes,
                fontsize=10, va='top', fontweight='bold',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    # 剔除极端组后的回归线
    if n_extreme > 0 and n_extreme < len(df):
        no_extreme_mask = ~extreme_mask
        info2 = _add_regression_line(ax, up_west[no_extreme_mask], speed[no_extreme_mask],
                                     color='green', label_prefix="剔除极端: ")
        if info2:
            r2, p2, label2 = info2
            ax.text(0.02, 0.88, f"剔除极端: {label2}", transform=ax.transAxes,
                    fontsize=10, va='top', color='green',
                    bbox=dict(boxstyle='round', fc='honeydew', alpha=0.9))

    ax.set_xlabel("Upper ω West Boundary (relative lon, °)", fontsize=12)
    ax.set_ylabel("Phase Speed (m/s)", fontsize=12)
    ax.set_title("D1: Up-West vs Phase Speed — Cluster Analysis", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_upwest_speed_cluster.png", dpi=150)
    plt.close()
    print(f"  Saved: D1")

    # --- D2: 剔除极端值后散点图 ---
    if n_extreme > 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        no_ext = ~extreme_mask
        ax.scatter(up_west[no_ext], speed[no_ext], c='#2980B9', s=50, alpha=0.7,
                   edgecolors='k', linewidths=0.5)
        info2 = _add_regression_line(ax, up_west[no_ext], speed[no_ext])
        if info2:
            r2, p2, label2 = info2
            ax.text(0.02, 0.98, label2, transform=ax.transAxes, fontsize=10,
                    va='top', fontweight='bold',
                    bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
        ax.set_xlabel("Upper ω West Boundary (relative lon, °)", fontsize=12)
        ax.set_ylabel("Phase Speed (m/s)", fontsize=12)
        ax.set_title(f"D2: Without Extreme West Events (N={no_ext.sum()})",
                     fontsize=14, fontweight='bold')
        ax.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "scatter_upwest_speed_no_outlier.png", dpi=150)
        plt.close()
        print(f"  Saved: D2")

    # --- D9: Spearman 秩相关 ---
    rho_full, p_rho_full = stats.spearmanr(up_west, speed,
                                            nan_policy='omit')
    print(f"\n  Spearman 秩相关 (全部): rho={rho_full:.3f}, p={p_rho_full:.4f}")

    if n_extreme > 0:
        rho_no, p_rho_no = stats.spearmanr(up_west[~extreme_mask],
                                            speed[~extreme_mask],
                                            nan_policy='omit')
        print(f"  Spearman 秩相关 (剔除极端): rho={rho_no:.3f}, p={p_rho_no:.4f}")

    # Pearson 全部 vs 剔除
    r_all_p, p_all_p = stats.pearsonr(up_west, speed)
    print(f"  Pearson (全部): r={r_all_p:.3f}, p={p_all_p:.4f}")
    if n_extreme > 0:
        r_no, p_no = stats.pearsonr(up_west[~extreme_mask], speed[~extreme_mask])
        print(f"  Pearson (剔除极端): r={r_no:.3f}, p={p_no:.4f}")

    return extreme_mask


# ======================
# H2: 振幅混淆
# ======================
def diagnose_h2(df):
    """H2: MJO振幅混淆效应。"""
    print("\n" + "="*60)
    print("H2: MJO振幅混淆效应")
    print("="*60)

    up_west = df["daily_up_west_mean"].values
    speed = df["phase_speed_m_s"].values
    amp = df["mean_amp"].values

    # amp 与 up_west / speed 的相关
    ok = np.isfinite(amp) & np.isfinite(up_west) & np.isfinite(speed)
    if ok.sum() > 5:
        r_amp_uw, p_amp_uw = stats.pearsonr(amp[ok], up_west[ok])
        r_amp_sp, p_amp_sp = stats.pearsonr(amp[ok], speed[ok])
        print(f"  amp vs up_west: r={r_amp_uw:.3f}, p={p_amp_uw:.4f}")
        print(f"  amp vs speed:   r={r_amp_sp:.3f}, p={p_amp_sp:.4f}")

        # 偏相关
        r_partial, p_partial, n = _partial_corr(up_west, speed, amp)
        print(f"  偏相关 (控制amp): r={r_partial:.3f}, p={p_partial:.4f}, N={n}")

    # --- D3: 按振幅着色散点图 ---
    fig, ax = plt.subplots(figsize=(9, 7))
    ok = np.isfinite(amp)
    sc = ax.scatter(up_west[ok], speed[ok], c=amp[ok], s=60, alpha=0.8,
                    cmap='YlOrRd', edgecolors='k', linewidths=0.5, zorder=5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mean MJO Amplitude", fontsize=11)

    _add_regression_line(ax, up_west, speed, color='black')

    ax.set_xlabel("Upper ω West Boundary (°)", fontsize=12)
    ax.set_ylabel("Phase Speed (m/s)", fontsize=12)
    ax.set_title("D3: Up-West vs Phase Speed (colored by Amplitude)",
                 fontsize=14, fontweight='bold')

    r_partial, p_partial, n = _partial_corr(up_west, speed, amp)
    sig = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else "n.s."
    ax.text(0.02, 0.98,
            f"Partial r (ctrl amp) = {r_partial:.3f}, p={p_partial:.4f} {sig}",
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_upwest_speed_by_amp.png", dpi=150)
    plt.close()
    print(f"  Saved: D3")


# ======================
# H3: 持续时间混淆
# ======================
def diagnose_h3(df):
    """H3: 事件持续时间混淆效应。"""
    print("\n" + "="*60)
    print("H3: 事件持续时间混淆效应")
    print("="*60)

    up_west = df["daily_up_west_mean"].values
    speed = df["phase_speed_m_s"].values
    dur = df["duration_days"].values.astype(float)

    ok = np.isfinite(dur) & np.isfinite(up_west) & np.isfinite(speed)
    if ok.sum() > 5:
        r_dur_uw, p_dur_uw = stats.pearsonr(dur[ok], up_west[ok])
        r_dur_sp, p_dur_sp = stats.pearsonr(dur[ok], speed[ok])
        print(f"  duration vs up_west: r={r_dur_uw:.3f}, p={p_dur_uw:.4f}")
        print(f"  duration vs speed:   r={r_dur_sp:.3f}, p={p_dur_sp:.4f}")

        r_partial, p_partial, n = _partial_corr(up_west, speed, dur)
        print(f"  偏相关 (控制duration): r={r_partial:.3f}, p={p_partial:.4f}, N={n}")

    # --- D4: 按持续时间着色 ---
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(up_west[ok], speed[ok], c=dur[ok], s=60, alpha=0.8,
                    cmap='viridis', edgecolors='k', linewidths=0.5, zorder=5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Duration (days)", fontsize=11)

    _add_regression_line(ax, up_west, speed, color='black')

    r_partial, p_partial, n = _partial_corr(up_west, speed, dur)
    sig = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else "n.s."
    ax.text(0.02, 0.98,
            f"Partial r (ctrl duration) = {r_partial:.3f}, p={p_partial:.4f} {sig}",
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.set_xlabel("Upper ω West Boundary (°)", fontsize=12)
    ax.set_ylabel("Phase Speed (m/s)", fontsize=12)
    ax.set_title("D4: Up-West vs Phase Speed (colored by Duration)",
                 fontsize=14, fontweight='bold')
    ax.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_upwest_speed_by_duration.png", dpi=150)
    plt.close()
    print(f"  Saved: D4")


# ======================
# H4: 经度位置混淆
# ======================
def diagnose_h4(df):
    """H4: 不同经度位置的系统差异。"""
    print("\n" + "="*60)
    print("H4: 经度位置混淆效应")
    print("="*60)

    up_west = df["daily_up_west_mean"].values
    speed = df["phase_speed_m_s"].values
    center = df["mean_center_lon"].values

    ok = np.isfinite(center) & np.isfinite(up_west) & np.isfinite(speed)
    if ok.sum() > 5:
        r_cen_uw, p_cen_uw = stats.pearsonr(center[ok], up_west[ok])
        r_cen_sp, p_cen_sp = stats.pearsonr(center[ok], speed[ok])
        print(f"  center_lon vs up_west: r={r_cen_uw:.3f}, p={p_cen_uw:.4f}")
        print(f"  center_lon vs speed:   r={r_cen_sp:.3f}, p={p_cen_sp:.4f}")

        r_partial, p_partial, n = _partial_corr(up_west, speed, center)
        print(f"  偏相关 (控制center_lon): r={r_partial:.3f}, p={p_partial:.4f}, N={n}")

    # --- D5: 按中心经度着色 ---
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(up_west[ok], speed[ok], c=center[ok], s=60, alpha=0.8,
                    cmap='coolwarm', edgecolors='k', linewidths=0.5, zorder=5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Mean Center Longitude (°E)", fontsize=11)

    _add_regression_line(ax, up_west, speed, color='black')

    r_partial, p_partial, n = _partial_corr(up_west, speed, center)
    sig = "***" if p_partial < 0.001 else "**" if p_partial < 0.01 else "*" if p_partial < 0.05 else "n.s."
    ax.text(0.02, 0.98,
            f"Partial r (ctrl center_lon) = {r_partial:.3f}, p={p_partial:.4f} {sig}",
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))

    ax.set_xlabel("Upper ω West Boundary (°)", fontsize=12)
    ax.set_ylabel("Phase Speed (m/s)", fontsize=12)
    ax.set_title("D5: Up-West vs Phase Speed (colored by Center Lon)",
                 fontsize=14, fontweight='bold')
    ax.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_upwest_speed_by_centerlon.png", dpi=150)
    plt.close()
    print(f"  Saved: D5")


# ======================
# H6: 边界跳跃 → 上升区宽度替代
# ======================
def diagnose_h6(df, extreme_mask):
    """H6: 边界检测跳跃 + 上升区宽度分析。"""
    print("\n" + "="*60)
    print("H6: 边界检测跳跃 + 上升区宽度")
    print("="*60)

    speed = df["phase_speed_m_s"].values
    width = df["ascent_width"].values
    up_west = df["daily_up_west_mean"].values
    field_up_west = df["field_up_west"].values  # 保留用于对比

    # 对比平均场 up_west vs 逐日均值 up_west
    ok_both = np.isfinite(field_up_west) & np.isfinite(up_west)
    if ok_both.sum() > 5:
        r_compare, p_compare = stats.pearsonr(field_up_west[ok_both], up_west[ok_both])
        print(f"  平均场up_west vs 逐日均值up_west: r={r_compare:.3f}, p={p_compare:.4f}")

    # 逐日均值 up_west vs speed (现在是主变量)
    ok_daily = np.isfinite(up_west) & np.isfinite(speed)
    if ok_daily.sum() > 5:
        r_daily, p_daily = stats.pearsonr(up_west[ok_daily], speed[ok_daily])
        print(f"  逐日均值up_west vs speed: r={r_daily:.3f}, p={p_daily:.4f}")

    # 上升区宽度 vs speed
    ok_w = np.isfinite(width) & np.isfinite(speed)
    if ok_w.sum() > 5:
        r_width, p_width = stats.pearsonr(width[ok_w], speed[ok_w])
        print(f"  ascent_width vs speed: r={r_width:.3f}, p={p_width:.4f}")

    # --- D7: up_west 值分布对比 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：daily_up_west_mean 分布直方图
    ax = axes[0]
    ax.hist(up_west, bins=40, color='#3498DB', edgecolor='black', alpha=0.7)
    ax.axvline(-70, color='red', ls='--', lw=2, label='Extreme threshold (-70°)')
    ax.set_xlabel("Daily-Mean Up-West (°)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("D7a: Distribution of Daily-Mean Up-West", fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 添加统计信息
    stats_text = (f"Mean: {np.mean(up_west):.1f}°\n"
                  f"Median: {np.median(up_west):.1f}°\n"
                  f"Std: {np.std(up_west):.1f}°\n"
                  f"[{np.min(up_west):.1f}, {np.max(up_west):.1f}]")
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    # 右图：逐日均值 vs 平均场 up_west 对比
    ax = axes[1]
    ok_both = np.isfinite(field_up_west) & np.isfinite(up_west)
    ax.scatter(field_up_west[ok_both], up_west[ok_both], c='#2C3E50', s=40, alpha=0.7)
    lim = [min(field_up_west[ok_both].min(), up_west[ok_both].min()) - 5,
           max(field_up_west[ok_both].max(), up_west[ok_both].max()) + 5]
    ax.plot(lim, lim, 'r--', lw=1.5, label='1:1 line')
    ax.set_xlabel("Mean-Field Up-West (°)", fontsize=12)
    ax.set_ylabel("Daily-Mean Up-West (°)", fontsize=12)
    ax.set_title("D7b: Mean-Field vs Daily-Mean Up-West", fontsize=13, fontweight='bold')
    if ok_both.sum() > 5:
        r_c, p_c = stats.pearsonr(field_up_west[ok_both], up_west[ok_both])
        ax.text(0.02, 0.98, f"r={r_c:.3f}, p={p_c:.4f}", transform=ax.transAxes,
                fontsize=10, va='top', bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    ax.legend(fontsize=9)
    ax.grid(ls='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "profile_examples_81deg.png", dpi=150)
    plt.close()
    print(f"  Saved: D7")

    # --- D8: 上升区宽度 vs 相速度 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ok_w = np.isfinite(width) & np.isfinite(speed)
    ax.scatter(width[ok_w], speed[ok_w], c='#E74C3C', s=50, alpha=0.7,
               edgecolors='k', linewidths=0.5)
    info = _add_regression_line(ax, width[ok_w], speed[ok_w], color='black')
    if info:
        r_w, p_w, label_w = info
        ax.text(0.02, 0.98, label_w, transform=ax.transAxes, fontsize=10,
                va='top', fontweight='bold',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.9))
    ax.set_xlabel("Ascent Width (east - west, °)", fontsize=12)
    ax.set_ylabel("Phase Speed (m/s)", fontsize=12)
    ax.set_title("D8: Ascent Width vs Phase Speed", fontsize=14, fontweight='bold')
    ax.grid(ls='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_ascent_width_speed.png", dpi=150)
    plt.close()
    print(f"  Saved: D8")


# ======================
# D6: 偏相关汇总
# ======================
def diagnose_partial_summary(df):
    """D6: 偏相关系数汇总表格图。"""
    print("\n" + "="*60)
    print("D6: 偏相关系数汇总")
    print("="*60)

    up_west = df["daily_up_west_mean"].values
    speed = df["phase_speed_m_s"].values
    amp = df["mean_amp"].values
    dur = df["duration_days"].values.astype(float)
    center = df["mean_center_lon"].values
    field_uw = df["field_up_west"].values  # 平均场值，用于对比

    # 0-order Pearson
    r_raw, p_raw = stats.pearsonr(up_west, speed)
    rho_raw, p_rho_raw = stats.spearmanr(up_west, speed, nan_policy='omit')

    # 偏相关
    results = [
        ("原始 Pearson", r_raw, p_raw, len(up_west)),
        ("原始 Spearman", rho_raw, p_rho_raw, len(up_west)),
    ]

    controls = [
        ("控制 Amplitude", amp),
        ("控制 Duration", dur),
        ("控制 Center Lon", center),
    ]

    for name, ctrl in controls:
        r, p, n = _partial_corr(up_west, speed, ctrl)
        results.append((name, r, p, n))

    # 同时控制多个变量
    ok_all = (np.isfinite(amp) & np.isfinite(dur) & np.isfinite(center)
              & np.isfinite(up_west) & np.isfinite(speed))
    if ok_all.sum() > 15:
        from numpy.linalg import lstsq
        X = np.column_stack([amp[ok_all], dur[ok_all], center[ok_all],
                             np.ones(ok_all.sum())])
        # up_west 残差
        coef_uw, _, _, _ = lstsq(X, up_west[ok_all], rcond=None)
        res_uw = up_west[ok_all] - X @ coef_uw
        # speed 残差
        coef_sp, _, _, _ = lstsq(X, speed[ok_all], rcond=None)
        res_sp = speed[ok_all] - X @ coef_sp
        r_multi, p_multi = stats.pearsonr(res_uw, res_sp)
        results.append(("控制 Amp+Dur+CenLon", r_multi, p_multi, ok_all.sum()))

    # 同时展示平均场 up_west 的结果作为对比
    ok_field = np.isfinite(field_uw) & np.isfinite(speed)
    if ok_field.sum() > 5:
        r_field, p_field = stats.pearsonr(field_uw[ok_field], speed[ok_field])
        results.append(("平均场 up_west (对比)", r_field, p_field, ok_field.sum()))

    # 剔除极端值后
    no_ext = up_west > -70
    ok_ne = no_ext & np.isfinite(speed)
    if ok_ne.sum() > 5:
        r_ne, p_ne = stats.pearsonr(up_west[ok_ne], speed[ok_ne])
        results.append(("剔除极端(>-70°)", r_ne, p_ne, ok_ne.sum()))

    # 打印汇总
    print(f"\n  {'方法':<25s} {'r':>8s} {'p':>10s} {'sig':>5s} {'N':>5s}")
    print("  " + "-"*55)
    for name, r, p, n in results:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {name:<25s} {r:>8.3f} {p:>10.4f} {sig:>5s} {n:>5d}")

    # --- D6: 汇总表格图 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')

    cell_text = []
    for name, r, p, n in results:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        cell_text.append([name, f"{r:.3f}", f"{p:.4f}", sig, str(n)])

    col_labels = ["方法", "r / rho", "p-value", "Sig.", "N"]
    colors_cell = []
    for name, r, p, n in results:
        if p < 0.05 and r > 0:
            colors_cell.append(['#FADBD8'] * 5)  # 红色背景=正相关显著
        elif p < 0.05 and r < 0:
            colors_cell.append(['#D5F5E3'] * 5)  # 绿色背景=负相关显著
        else:
            colors_cell.append(['white'] * 5)

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellColours=colors_cell,
                     colColours=['#D6EAF8'] * 5,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    ax.set_title("D6: Partial Correlation Summary\n"
                 "(Red = significant positive, Green = significant negative)",
                 fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "partial_correlation_summary.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: D6")

    # --- D9: Robust 相关汇总 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    robust_text = []
    robust_text.append(["Pearson (全部)", f"{r_raw:.3f}", f"{p_raw:.4f}"])
    robust_text.append(["Spearman (全部)", f"{rho_raw:.3f}", f"{p_rho_raw:.4f}"])
    no_ext = up_west > -70
    r_ne, p_ne = stats.pearsonr(up_west[no_ext], speed[no_ext])
    rho_ne, p_rho_ne = stats.spearmanr(up_west[no_ext], speed[no_ext])
    robust_text.append(["Pearson (剔除极端)", f"{r_ne:.3f}", f"{p_ne:.4f}"])
    robust_text.append(["Spearman (剔除极端)", f"{rho_ne:.3f}", f"{p_rho_ne:.4f}"])

    table2 = ax.table(cellText=robust_text,
                      colLabels=["方法", "Correlation", "p-value"],
                      colColours=['#F5CBA7'] * 3,
                      loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1.0, 1.6)

    ax.set_title("D9: Robust Correlation Summary", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "robust_correlation_summary.png", dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: D9")


# ======================
# MAIN
# ======================
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("高层西边界与相速度正相关问题诊断")
    print("=" * 60)

    df = load_data()

    # H1: 聚类 + 杠杆
    extreme_mask = diagnose_h1(df)

    # H2: 振幅混淆
    diagnose_h2(df)

    # H3: 持续时间混淆
    diagnose_h3(df)

    # H4: 经度位置混淆
    diagnose_h4(df)

    # H6: 边界跳跃 + 宽度替代
    diagnose_h6(df, extreme_mask)

    # D6: 偏相关汇总
    diagnose_partial_summary(df)

    print("\n" + "=" * 60)
    print(f"所有诊断图表已保存到: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
