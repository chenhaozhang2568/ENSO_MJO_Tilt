# -*- coding: utf-8 -*-
"""
plot_front_indicators.py: 7种前端指标可视化

================================================================================
为每种指标生成：
    1. event_mean_distribution.png  - 逐事件均值直方图
    2. daily_distribution.png       - 逐日分布直方图
    3. indicator_vs_speed.png       - 与相速度散点回归图
    4. profiles/ (10张)             - 随机10天逐日场剖面图

额外：summary_correlation_table.png - 7种指标相关系数汇总条形图

输出目录：
    E:/Projects/ENSO_MJO_Tilt/outputs/figures/front_indicators/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.interpolate import Akima1DInterpolator
from pathlib import Path

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
INDICATORS_NC = r"E:\Datas\Derived\front_indicators_daily.nc"
EVENT_CSV     = r"E:\Datas\Derived\front_indicators_event_mean.csv"
EVENTS_CSV    = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
STEP3_NC      = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
W_NORM_NC     = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
Q_NORM_NC     = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
U_NORM_NC     = r"E:\Datas\Derived\era5_mjo_recon_u_norm_1979-2022.nc"
T_NORM_NC     = r"E:\Datas\Derived\era5_mjo_recon_t_norm_1979-2022.nc"

FIG_ROOT = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\front_indicators")

# ======================
# SETTINGS
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"
SMOOTH_WINDOW = 10
CSA_TARGET_DLON = 0.25

LOW_LAYER  = (1000.0, 850.0)
UP_LAYER   = (400.0, 200.0)

N_PROFILE_SAMPLES = 10

# 每种指标的画图范围配置
PLOT_RANGES = {
    "F1_q_front":          (0, 180),
    "F2_omega_sub_front":  (-90, 90),
    "F3_u_conv_front":     (-90, 180),
    "F4_q_grad_max":       (-90, 90),
    "F5_T_front":          (-90, 90),
    "F6_omega_low_east":   (-90, 90),
    "F7_u_shear_change":   (-90, 90),
}

# 指标定义
INDICATORS = [
    {"key": "F1_q_front",          "label": "q Front (pos->neg)", "folder": "F1_q_front",
     "field": "q", "layer": "low", "color": "#27AE60",
     "desc": "Low-level q (1000-850hPa mean)\nZero crossing: positive -> negative (5 deg tol)"},
    {"key": "F2_omega_sub_front",  "label": "omega Subsidence Front (neg->pos)", "folder": "F2_omega_subsidence_front",
     "field": "w", "layer": "low", "color": "#E74C3C",
     "desc": "Low-level omega (1000-850hPa mean)\nZero crossing: negative -> positive (5 deg tol)"},
    {"key": "F3_u_conv_front",     "label": "u Convergence Front (pos->neg)", "folder": "F3_u_convergence_front",
     "field": "u", "layer": "low", "color": "#2980B9",
     "desc": "Low-level u (1000-850hPa mean)\nZero crossing: positive -> negative"},
    {"key": "F4_q_grad_max",       "label": "q Gradient Min Position", "folder": "F4_q_gradient_max",
     "field": "q", "layer": "low", "color": "#8E44AD",
     "desc": "Low-level q (1000-850hPa mean)\ndq/dx minimum (steepest negative gradient)"},
    {"key": "F5_T_front",          "label": "T Front (pos->neg)", "folder": "F5_T_front",
     "field": "t", "layer": "low", "color": "#E67E22",
     "desc": "Low-level T (1000-850hPa mean)\nZero crossing: positive -> negative"},
    {"key": "F6_omega_low_east",   "label": "omega Low East Boundary", "folder": "F6_omega_low_east_boundary",
     "field": "w", "layer": "low", "color": "#C0392B",
     "desc": "Low-level omega (1000-850hPa mean)\nAscent region east boundary (5 deg tol)"},
    {"key": "F7_u_shear_change",   "label": "u Shear Change", "folder": "F7_u_vertical_shear",
     "field": "u", "layer": "shear", "color": "#1ABC9C",
     "desc": "u vertical shear (u_upper - u_lower)\nFirst zero crossing"},
]


# ======================
# 辅助函数
# ======================
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


def _interp_1d(src_lon, profile, target_lon):
    valid = np.isfinite(profile)
    if valid.sum() < 4:
        return np.full(len(target_lon), np.nan)
    return Akima1DInterpolator(src_lon[valid], profile[valid])(target_lon)


def _prepare_profile(raw_2d, lon_360, center, layer_mask, target_rel):
    """准备单日 1D 剖面：截取rel范围, 层平均, 插值+平滑"""
    rel_lon = lon_360 - center
    rel_lon = np.where(rel_lon > 180, rel_lon - 360, rel_lon)
    rel_lon = np.where(rel_lon < -180, rel_lon + 360, rel_lon)

    tr_min = target_rel.min() - 5
    tr_max = target_rel.max() + 5
    mask = (rel_lon >= tr_min) & (rel_lon <= tr_max)
    if mask.sum() < 7:
        return np.full(len(target_rel), np.nan)

    rel_sub = rel_lon[mask]
    data_sub = raw_2d[:, mask]
    sort_idx = np.argsort(rel_sub)
    rel_sub = rel_sub[sort_idx]
    data_sub = data_sub[:, sort_idx]

    layer_data = data_sub[layer_mask, :]
    if layer_data.shape[0] == 0:
        return np.full(len(target_rel), np.nan)
    profile = np.nanmean(layer_data, axis=0)
    interped = _interp_1d(rel_sub, profile, target_rel)
    return _smooth_1d(interped, SMOOTH_WINDOW)


def _load_field_raw(nc_path, var_name):
    """加载重构场原始数据。"""
    ds = xr.open_dataset(nc_path, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    da = ds[var_name]
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    lon_vals = da["lon"].values
    if lon_vals.min() < 0:
        new_lon = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
        da = da.assign_coords(lon=new_lon).sortby("lon")
    return da


# ======================
# 图: 分布直方图 (通用)
# ======================
def _plot_histogram(vals, ind_info, out_dir, kind="event_mean"):
    """
    绘制直方图。kind='event_mean' 或 'daily'
    """
    if len(vals) < 3:
        print(f"  {ind_info['key']}: too few values for {kind} distribution ({len(vals)})")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(vals, bins=30, color=ind_info["color"], edgecolor='black', alpha=0.7)

    mean_val = np.mean(vals)
    median_val = np.median(vals)
    ax.axvline(mean_val, color='navy', linestyle='--', linewidth=2,
               label=f'Mean: {mean_val:.1f}')
    ax.axvline(median_val, color='darkred', linestyle=':', linewidth=2,
               label=f'Median: {median_val:.1f}')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5,
               label='Convective Center')

    kind_label = "Event-Mean" if kind == "event_mean" else "Daily"
    ax.set_xlabel(f"{ind_info['label']} (Relative Longitude, deg)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"{kind_label} {ind_info['label']} Distribution (N={len(vals)})",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    stats_text = (
        f"Mean: {mean_val:.2f}\n"
        f"Median: {median_val:.2f}\n"
        f"Std: {np.std(vals):.2f}\n"
        f"Min: {np.min(vals):.2f}\n"
        f"Max: {np.max(vals):.2f}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=10)

    plt.tight_layout()
    fname = f"{kind}_distribution.png"
    out = out_dir / fname
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ======================
# 图: 与相速度散点回归图
# ======================
def plot_vs_speed(df, ind_info, out_dir):
    x = df["phase_speed_m_s"].values.astype(float)
    y = df[ind_info["key"]].values.astype(float)
    ok = np.isfinite(x) & np.isfinite(y)

    if ok.sum() < 5:
        print(f"  {ind_info['key']}: too few valid pairs ({ok.sum()})")
        return

    x_ok, y_ok = x[ok], y[ok]
    r, p = stats.pearsonr(x_ok, y_ok)

    if p < 0.01:
        sig_str = "exceeding the 99% confidence level."
    elif p < 0.05:
        sig_str = "exceeding the 95% confidence level."
    else:
        sig_str = "exceeding the not sig. confidence level."

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(x_ok, y_ok, c='black', s=30, zorder=5)

    slope, intercept, _, _, _ = stats.linregress(x_ok, y_ok)
    x_line = np.array([x_ok.min(), x_ok.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, zorder=4)

    ax.set_xlabel("Speed (m/s)", fontsize=14)
    ax.set_ylabel(f"{ind_info['label']} (deg)", fontsize=14)
    ax.text(0.95, 0.95, f"Cor={r:.2f}", transform=ax.transAxes,
            fontsize=14, fontweight='bold', ha='right', va='top')

    caption = (
        f"Scatter diagram of {ind_info['label']} (daily mean) (y) vs.\n"
        f"phase speed (x) for {ok.sum()} MJO events.\n"
        f"Red line: least squares fit.\n"
        f"Cor = {r:.2f} (p = {p:.4f}),\n"
        f"{sig_str}"
    )
    fig.text(0.5, -0.02, caption, ha='center', va='top', fontsize=10, style='italic')

    plt.tight_layout()
    out = out_dir / "indicator_vs_speed.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ======================
# 图: 逐日剖面图
# ======================
def plot_daily_profiles(ind_info, sample_days, field_data, out_dir):
    profile_dir = out_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    key = ind_info["key"]
    plot_min, plot_max = PLOT_RANGES[key]
    target_rel = np.arange(plot_min, plot_max + CSA_TARGET_DLON, CSA_TARGET_DLON)

    for day_idx, day_date in sample_days:
        c = field_data["center_lon"][day_idx]
        if not np.isfinite(c):
            continue

        indicator_val = float(field_data["indicators_ds"][key].values[day_idx])

        # 找事件编号
        events = field_data["events"]
        event_id = "?"
        day_ts = pd.Timestamp(day_date)
        for _, ev_row in events.iterrows():
            if pd.Timestamp(ev_row["start_date"]) <= day_ts <= pd.Timestamp(ev_row["end_date"]):
                event_id = int(ev_row["event_id"])
                break

        fig, ax = plt.subplots(figsize=(14, 5))

        if ind_info["layer"] == "shear":
            # F7: 画高层u、低层u、差值三条线
            u_raw = field_data["u_raw"]
            lon_u = field_data["lon_u"]
            u_low_profile = _prepare_profile(u_raw[day_idx], lon_u, c,
                                              field_data["low_mask_u"], target_rel)
            u_up_profile = _prepare_profile(u_raw[day_idx], lon_u, c,
                                             field_data["up_mask_u"], target_rel)
            shear = u_up_profile - u_low_profile

            ax.plot(target_rel, u_up_profile, 'r-', linewidth=1.5, alpha=0.7,
                    label='u upper (400-200 hPa mean)')
            ax.plot(target_rel, u_low_profile, 'b-', linewidth=1.5, alpha=0.7,
                    label='u lower (1000-850 hPa mean)')
            ax.plot(target_rel, shear, 'k-', linewidth=2.5,
                    label='Shear (upper - lower)')
            ax.axhline(0, color='gray', linewidth=1, alpha=0.5)
            ax.set_ylabel("u (normalized)", fontsize=12)

        else:
            # 普通单变量
            field_key = ind_info["field"]
            raw = field_data[f"{field_key}_raw"]
            lon = field_data[f"lon_{field_key}"]
            layer_mask = field_data[f"low_mask_{field_key}"]

            profile = _prepare_profile(raw[day_idx], lon, c, layer_mask, target_rel)
            ax.plot(target_rel, profile, '-', color=ind_info["color"], linewidth=2.5,
                    label=ind_info["desc"].split("\n")[0])
            ax.fill_between(target_rel, 0, profile,
                            where=profile > 0, alpha=0.15, color=ind_info["color"])
            ax.fill_between(target_rel, 0, profile,
                            where=profile < 0, alpha=0.10, color='gray')
            ax.axhline(0, color='gray', linewidth=1, alpha=0.5)

            var_label = {"q": "q", "w": "omega", "u": "u", "t": "T"}[field_key]
            ax.set_ylabel(f"{var_label} (normalized)", fontsize=12)

            # F4: 额外画梯度
            if key == "F4_q_grad_max":
                dq = np.diff(profile)
                dx = np.diff(target_rel)
                grad = dq / dx
                rel_mid = 0.5 * (target_rel[:-1] + target_rel[1:])
                ax2_twin = ax.twinx()
                ax2_twin.plot(rel_mid, grad, '--', color='purple', linewidth=1.2,
                              alpha=0.6, label='dq/dx')
                ax2_twin.set_ylabel("dq/dx", fontsize=10, color='purple')
                ax2_twin.tick_params(axis='y', labelcolor='purple')

        # 标注位置
        if plot_min <= 0 <= plot_max:
            ax.axvline(0, color='limegreen', linewidth=2.5, alpha=0.8, label='Convective Center')
        if np.isfinite(indicator_val):
            ax.axvline(indicator_val, color='red', linewidth=2.5, linestyle='--',
                       label=f'{ind_info["label"]}: {indicator_val:.1f} deg')

        ax.set_xlim(plot_min, plot_max)
        ax.set_xlabel("Relative Longitude (deg)", fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(alpha=0.3)

        date_str = pd.Timestamp(day_date).strftime("%Y-%m-%d")
        ind_val_str = f"{indicator_val:.1f}" if np.isfinite(indicator_val) else "NaN"
        ax.set_title(f"Event #{event_id} | {date_str} | Center: {c:.1f} E | "
                     f"{ind_info['label']} = {ind_val_str} deg",
                     fontsize=12, fontweight='bold')

        plt.tight_layout()
        out_path = profile_dir / f"daily_{date_str}_profile.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(sample_days)} profiles to {profile_dir}")


# ======================
# 汇总: 相关系数条形图
# ======================
def plot_summary_correlation(df, out_dir):
    speed = df["phase_speed_m_s"].values.astype(float)

    names = []
    corrs = []
    pvals = []

    for ind in INDICATORS:
        key = ind["key"]
        vals = df[key].values.astype(float)
        ok = np.isfinite(vals) & np.isfinite(speed)
        if ok.sum() > 5:
            r, p = stats.pearsonr(vals[ok], speed[ok])
            names.append(ind["label"])
            corrs.append(r)
            pvals.append(p)
        else:
            names.append(ind["label"])
            corrs.append(0)
            pvals.append(1)

    corrs = np.array(corrs)
    pvals = np.array(pvals)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    colors = [ind["color"] for ind in INDICATORS]

    ax.bar(x, corrs, color=colors, edgecolor='black', alpha=0.8, width=0.6)

    for i, (r, p) in enumerate(zip(corrs, pvals)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        y_off = 0.02 if r >= 0 else -0.04
        ax.text(i, r + y_off, f"{r:.2f}{sig}", ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel("Pearson Correlation (r)", fontsize=13)
    ax.set_title("Correlation of 7 Front Indicators with MJO Phase Speed",
                 fontsize=14, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)

    ax.text(0.02, 0.98, "* p<0.05  ** p<0.01  *** p<0.001",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    out = out_dir / "summary_correlation_table.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ======================
# MAIN
# ======================
def main():
    print("=" * 60)
    print("plot_front_indicators.py: 7 Front Indicators Visualization")
    print("=" * 60)

    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    # --- 加载数据 ---
    print("Loading indicator data...")
    ds_ind = xr.open_dataset(INDICATORS_NC)
    df_event = pd.read_csv(EVENT_CSV)
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])

    time_ind = pd.to_datetime(ds_ind["time"].values)
    print(f"  Events: {len(df_event)}, Daily records: {len(time_ind)}")

    # --- 加载原始场数据（用于剖面图）---
    print("Loading raw fields for profiles...")
    q_da = _load_field_raw(Q_NORM_NC, "q_mjo_recon_norm")
    w_da = _load_field_raw(W_NORM_NC, "w_mjo_recon_norm")
    u_da = _load_field_raw(U_NORM_NC, "u_mjo_recon_norm")
    t_da = _load_field_raw(T_NORM_NC, "t_mjo_recon_norm")

    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    center_lon = ds3["center_lon_track"].values.astype(float)
    time_step3 = pd.to_datetime(ds3["time"].values)

    # 时间索引映射
    time_step3_ns = time_step3.values.astype('datetime64[ns]')
    field_center = np.full(len(time_ind), np.nan)
    for i, t in enumerate(time_ind):
        t_ns = np.datetime64(t, 'ns')
        idx = np.searchsorted(time_step3_ns, t_ns)
        if idx < len(center_lon):
            field_center[i] = center_lon[idx]

    levels_q = q_da["level"].values.astype(float)
    levels_w = w_da["level"].values.astype(float)
    levels_u = u_da["level"].values.astype(float)
    levels_t = t_da["level"].values.astype(float)

    field_data = {
        "q_raw": q_da.values, "lon_q": q_da["lon"].values.astype(float),
        "w_raw": w_da.values, "lon_w": w_da["lon"].values.astype(float),
        "u_raw": u_da.values, "lon_u": u_da["lon"].values.astype(float),
        "t_raw": t_da.values, "lon_t": t_da["lon"].values.astype(float),
        "low_mask_q": (levels_q >= min(LOW_LAYER)) & (levels_q <= max(LOW_LAYER)),
        "low_mask_w": (levels_w >= min(LOW_LAYER)) & (levels_w <= max(LOW_LAYER)),
        "low_mask_u": (levels_u >= min(LOW_LAYER)) & (levels_u <= max(LOW_LAYER)),
        "low_mask_t": (levels_t >= min(LOW_LAYER)) & (levels_t <= max(LOW_LAYER)),
        "up_mask_u":  (levels_u >= min(UP_LAYER)) & (levels_u <= max(UP_LAYER)),
        "center_lon": field_center,
        "time": time_ind,
        "indicators_ds": ds_ind,
        "events": events,
    }

    # --- 选取随机10天 ---
    event_day_mask = np.zeros(len(time_ind), dtype=bool)
    for _, row in events.iterrows():
        ts = np.datetime64(row["start_date"])
        te = np.datetime64(row["end_date"])
        m = (time_ind >= ts) & (time_ind <= te)
        event_day_mask |= m

    any_valid = np.zeros(len(time_ind), dtype=bool)
    for ind in INDICATORS:
        vals = ds_ind[ind["key"]].values.astype(float)
        any_valid |= np.isfinite(vals)
    valid_days = event_day_mask & any_valid & np.isfinite(field_center)
    valid_indices = np.where(valid_days)[0]

    rng = np.random.default_rng()  # 不固定seed
    n_sample = min(N_PROFILE_SAMPLES, len(valid_indices))
    sample_indices = rng.choice(valid_indices, size=n_sample, replace=False)
    sample_indices.sort()
    sample_days = [(int(idx), time_ind[idx]) for idx in sample_indices]
    print(f"  Random sample days: {[pd.Timestamp(d).strftime('%Y-%m-%d') for _, d in sample_days]}")

    # --- 获取逐日有效值（用于 daily distribution）---
    daily_valid_mask = event_day_mask  # 所有事件日

    # --- 逐指标生成图片 ---
    for ind in INDICATORS:
        print(f"\n--- {ind['key']} ---")
        ind_dir = FIG_ROOT / ind["folder"]
        ind_dir.mkdir(parents=True, exist_ok=True)

        # 图1: 逐事件均值直方图
        event_vals = df_event[ind["key"]].dropna().values
        _plot_histogram(event_vals, ind, ind_dir, kind="event_mean")

        # 图2: 逐日分布直方图
        daily_vals = ds_ind[ind["key"]].values[daily_valid_mask].astype(float)
        daily_vals = daily_vals[np.isfinite(daily_vals)]
        _plot_histogram(daily_vals, ind, ind_dir, kind="daily")

        # 图3: vs 相速度
        plot_vs_speed(df_event, ind, ind_dir)

        # 图4: 10天剖面
        plot_daily_profiles(ind, sample_days, field_data, ind_dir)

    # --- 汇总相关系数图 ---
    print("\n--- Summary correlation table ---")
    plot_summary_correlation(df_event, FIG_ROOT)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {FIG_ROOT}")
    print("DONE")


if __name__ == "__main__":
    main()
