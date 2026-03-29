# -*- coding: utf-8 -*-
"""
generate_both_daily.py
生成 both_daily/ 文件夹：上下层均用逐日计算后取事件平均。

与 both_meanfield/ 对应：
  - both_meanfield: 上下层均从事件平均场上计算
  - both_daily:     每日独立计算 up_west 和 centroid，再对事件取平均

输出结构（与 both_meanfield 完全一致）：
  both_daily/
    event_profile/        (115 张，标注逐日 up_west + q_max 散点 + 均值连线)
    centroid_profile/     (115 张，标注均值 up_west + centroid 连线)
    6 张散点图
    event_daily_values.csv
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import Akima1DInterpolator, interp1d
from scipy import stats
from pathlib import Path

# === 路径 ===
BASE_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_diagnose")
OUT_DIR = BASE_DIR / "both_daily"

W_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
Q_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = r"E:\Datas\Derived\phase_speed_q_events.csv"

# === 参数 ===
SMOOTH_WINDOW = 10
CSA_TARGET_DLON = 0.25
REL_LON_RANGE = (-90, 90)
Q_MAX_SEARCH_RANGE = (-90, 90)
UP_LAYER = (400.0, 200.0)
Q_LOW_LAYER = (1000.0, 850.0)
LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}


# ==============================
# 辅助函数
# ==============================
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


def _ascent_boundary_zero(rel_lon, w_profile):
    m = np.isfinite(w_profile) & np.isfinite(rel_lon)
    if m.sum() < 7:
        return np.nan
    rr = rel_lon[m].astype(float)
    ww = w_profile[m].astype(float)
    pivot_idx = int(np.argmin(np.abs(rr)))
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return np.nan
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= 0:
            return float(rr[i])
    return np.nan


def _find_q_max(rel_lon, q_profile, search_min=-90, search_max=90):
    m = np.isfinite(q_profile) & np.isfinite(rel_lon)
    m = m & (rel_lon >= search_min) & (rel_lon <= search_max)
    if m.sum() < 7:
        return np.nan
    rr = rel_lon[m].astype(float)
    qq = q_profile[m].astype(float)
    return float(rr[int(np.argmax(qq))])


def _find_q_centroid(rel_lon, q_profile):
    m = np.isfinite(q_profile) & np.isfinite(rel_lon) & (q_profile > 0)
    if m.sum() < 3:
        return np.nan
    rr = rel_lon[m].astype(float)
    qq = q_profile[m].astype(float)
    q_sum = np.sum(qq)
    if q_sum < 1e-20:
        return np.nan
    return float(np.sum(qq * rr) / q_sum)


def _prepare_field_data(ds_w, ds_q, ds3):
    w_raw = ds_w['w_mjo_recon_norm'].values
    q_raw = ds_q['q_mjo_recon_norm'].values
    levels_w = ds_w['pressure_level'].values if 'pressure_level' in ds_w else ds_w['level'].values
    levels_q = ds_q['pressure_level'].values if 'pressure_level' in ds_q else ds_q['level'].values
    lon_w = ds_w['lon'].values
    lon_q = ds_q['lon'].values
    time_w = pd.to_datetime(ds_w['time'].values)
    center_lon = ds3['center_lon_track'].values.astype(float)
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
        'time': time_w, 'center_lon': center_lon,
    }


def _process_single_day(data, idx, center):
    """处理单日场，插值到统一网格"""
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    target_h = np.linspace(0.0, 12.0, 120)

    rel_lon_w = data['lon_w_360'] - center
    mask_w = (rel_lon_w >= REL_LON_RANGE[0]) & (rel_lon_w <= REL_LON_RANGE[1])
    rel_lons_w = rel_lon_w[mask_w]
    w_day = data['w_raw'][idx, :, :][:, data['w_sort']][:, mask_w]
    heights_w = np.array([LEVEL_TO_HEIGHT[int(p)] for p in data['levels_w']])

    w_interp = np.full((len(data['levels_w']), len(target_rel)), np.nan)
    for k in range(len(data['levels_w'])):
        interped = _cubic_spline_interp_1d(rel_lons_w, w_day[k, :], target_rel)
        w_interp[k, :] = _smooth_1d(interped, SMOOTH_WINDOW)

    w_h = np.full((len(target_h), len(target_rel)), np.nan)
    for j in range(len(target_rel)):
        col = w_interp[:, j]
        v = np.isfinite(col)
        if v.sum() >= 2:
            f = interp1d(heights_w[v], col[v], kind='linear', bounds_error=False, fill_value=np.nan)
            w_h[:, j] = f(target_h)

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
        v = np.isfinite(col)
        if v.sum() >= 2:
            f = interp1d(heights_q[v], col[v], kind='linear', bounds_error=False, fill_value=np.nan)
            q_h[:, j] = f(target_h)

    return w_h, q_h, target_rel, target_h


def _compute_daily_up_west(data, event_indices, centers):
    """逐日计算上层 omega=0 西边界"""
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    results = []
    for idx, c in zip(event_indices, centers):
        if not np.isfinite(c):
            results.append(np.nan)
            continue
        rel_lon_w = data['lon_w_360'] - c
        mask_w = (rel_lon_w >= REL_LON_RANGE[0]) & (rel_lon_w <= REL_LON_RANGE[1])
        w_day = data['w_raw'][idx, :, :][:, data['w_sort']][:, mask_w]
        up_mask = (data['levels_w'] >= UP_LAYER[1]) & (data['levels_w'] <= UP_LAYER[0])
        w_up_mean = np.nanmean(w_day[up_mask, :], axis=0)
        w_up_interp = _cubic_spline_interp_1d(rel_lon_w[mask_w], w_up_mean, target_rel)
        w_up_smooth = _smooth_1d(w_up_interp, SMOOTH_WINDOW)
        results.append(_ascent_boundary_zero(target_rel, w_up_smooth))
    return np.array(results)


def _compute_daily_centroid(data, event_indices, centers):
    """逐日计算低层 q centroid"""
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    results = []
    for idx, c in zip(event_indices, centers):
        if not np.isfinite(c):
            results.append(np.nan)
            continue
        rel_lon_q = data['lon_q_360'] - c
        mask_q = (rel_lon_q >= REL_LON_RANGE[0]) & (rel_lon_q <= REL_LON_RANGE[1])
        q_day = data['q_raw'][idx, :, :][:, data['q_sort']][:, mask_q]
        low_mask = (data['levels_q'] >= Q_LOW_LAYER[1]) & (data['levels_q'] <= Q_LOW_LAYER[0])
        q_low_mean = np.nanmean(q_day[low_mask, :], axis=0)
        q_interp = _cubic_spline_interp_1d(rel_lon_q[mask_q], q_low_mean, target_rel)
        q_smooth = _smooth_1d(q_interp, SMOOTH_WINDOW)
        results.append(_find_q_centroid(target_rel, q_smooth))
    return np.array(results)


def _compute_daily_q_max(data, event_indices, centers):
    """逐日计算低层 q 最大值经度"""
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    results = []
    for idx, c in zip(event_indices, centers):
        if not np.isfinite(c):
            results.append(np.nan)
            continue
        rel_lon_q = data['lon_q_360'] - c
        mask_q = (rel_lon_q >= REL_LON_RANGE[0]) & (rel_lon_q <= REL_LON_RANGE[1])
        q_day = data['q_raw'][idx, :, :][:, data['q_sort']][:, mask_q]
        low_mask = (data['levels_q'] >= Q_LOW_LAYER[1]) & (data['levels_q'] <= Q_LOW_LAYER[0])
        q_low_mean = np.nanmean(q_day[low_mask, :], axis=0)
        q_interp = _cubic_spline_interp_1d(rel_lon_q[mask_q], q_low_mean, target_rel)
        q_smooth = _smooth_1d(q_interp, SMOOTH_WINDOW)
        results.append(_find_q_max(target_rel, q_smooth,
                                   Q_MAX_SEARCH_RANGE[0], Q_MAX_SEARCH_RANGE[1]))
    return np.array(results)


# ==============================
# 绘图函数（复用自 reorganize_and_regenerate.py）
# ==============================
def plot_profile(eid, w_mean, q_mean, target_rel, target_h,
                 mean_up_west, mean_lower, lower_label, lower_color,
                 event_row, out_path, scatter_uw=None, scatter_qm=None):
    up_h_min, up_h_max = LEVEL_TO_HEIGHT[400], LEVEL_TO_HEIGHT[200]
    low_h_min, low_h_max = LEVEL_TO_HEIGHT[1000], LEVEL_TO_HEIGHT[850]
    up_h_mid = (up_h_min + up_h_max) / 2.0
    low_h_mid = (low_h_min + low_h_max) / 2.0

    fig, ax = plt.subplots(figsize=(14, 7))
    X, Y = np.meshgrid(target_rel, target_h)

    w_display = np.where((target_h >= up_h_min)[:, None], w_mean, np.nan)
    vmax_w = np.nanmax(np.abs(w_display)) * 0.8
    if vmax_w < 1e-6 or not np.isfinite(vmax_w): vmax_w = 0.01
    cf_w = ax.contourf(X, Y, w_display, levels=np.linspace(-vmax_w, vmax_w, 21),
                       cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vmax_w, vcenter=0, vmax=vmax_w),
                       extend='both', alpha=0.7)

    q_display = np.where((target_h <= low_h_max)[:, None], q_mean, np.nan)
    vmax_q = np.nanmax(np.abs(q_display)) * 0.8
    if vmax_q < 1e-10 or not np.isfinite(vmax_q): vmax_q = 1e-5
    cf_q = ax.contourf(X, Y, q_display, levels=np.linspace(-vmax_q, vmax_q, 21),
                       cmap='BrBG', norm=TwoSlopeNorm(vmin=-vmax_q, vcenter=0, vmax=vmax_q),
                       extend='both', alpha=0.9)

    w_contour = np.where((target_h >= up_h_min)[:, None], w_mean, np.nan)
    ax.contour(X, Y, w_contour, levels=[0], colors='black', linewidths=2.0)
    for h in [low_h_max, up_h_min]:
        ax.axhline(h, color='gray', lw=1.2, ls='-', alpha=0.6)

    ax.text(REL_LON_RANGE[1]-2, (low_h_min+low_h_max)/2, 'q (1000–850 hPa)',
            fontsize=9, color='darkgreen', fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.text(REL_LON_RANGE[1]-2, (up_h_min+up_h_max)/2, 'ω (400–200 hPa)',
            fontsize=9, color='darkred', fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    if scatter_uw is not None:
        v = scatter_uw[np.isfinite(scatter_uw)]
        if len(v) > 0:
            jitter = np.random.default_rng(42).uniform(-0.3, 0.3, len(v))
            ax.scatter(v, up_h_mid + jitter, c='red', s=30, alpha=0.5, zorder=8,
                       edgecolors='darkred', linewidths=0.5, label=f'Daily ω west (N={len(v)})')
    if scatter_qm is not None:
        v = scatter_qm[np.isfinite(scatter_qm)]
        if len(v) > 0:
            jitter = np.random.default_rng(43).uniform(-0.2, 0.2, len(v))
            ax.scatter(v, low_h_mid + jitter, c='limegreen', s=30, alpha=0.5, zorder=8,
                       edgecolors='darkgreen', linewidths=0.5, label=f'Daily q max (N={len(v)})')

    if np.isfinite(mean_up_west) and np.isfinite(mean_lower):
        tilt_val = mean_lower - mean_up_west
        ax.plot([mean_up_west, mean_lower], [up_h_mid, low_h_mid],
                'o-', color='gold', markersize=14, markeredgecolor='black',
                markeredgewidth=2, lw=3.5, zorder=10,
                label=f'Tilt ({lower_label}) = {tilt_val:.1f}°')
        ax.annotate(f'ω west: {mean_up_west:.1f}°', (mean_up_west, up_h_mid),
                    textcoords='offset points', xytext=(15, 10), fontsize=10,
                    color='darkgoldenrod', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=1.5))
        ax.annotate(f'{lower_label}: {mean_lower:.1f}°', (mean_lower, low_h_mid),
                    textcoords='offset points', xytext=(15, -20), fontsize=10,
                    color=lower_color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#EBF5FB', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color=lower_color, lw=1.5))
        mid_x = (mean_up_west + mean_lower) / 2
        mid_y = (up_h_mid + low_h_mid) / 2
        ax.text(mid_x + 5, mid_y, f'Δlon = {tilt_val:.1f}°', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', edgecolor='gold', alpha=0.9))

    ax.axvline(0, color='limegreen', lw=2.5, alpha=0.8, label='Convective Center')
    ax.set_ylim(0, 12); ax.set_xlim(REL_LON_RANGE)
    ax.set_ylabel('Height (km)', fontsize=12); ax.set_xlabel('Relative Longitude (°)', fontsize=12)

    ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim())
    pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in pticks])
    ax2.set_yticklabels([str(p) for p in pticks])
    ax2.set_ylabel('Pressure (hPa)', fontsize=12)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cax_w = inset_axes(ax, width='2%', height='45%', loc='upper right',
                       bbox_to_anchor=(0.14, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    fig.colorbar(cf_w, cax=cax_w).set_label('omega (norm)', fontsize=8)
    cax_q = inset_axes(ax, width='2%', height='45%', loc='lower right',
                       bbox_to_anchor=(0.14, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    fig.colorbar(cf_q, cax=cax_q).set_label('q (norm)', fontsize=8)

    eid_v = int(event_row['event_id'])
    ax.set_title(f"Event #{eid_v}: {event_row['start_date']} ~ {event_row['end_date']} "
                 f"({int(event_row['duration_days'])}d) — Daily Mean",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()


def plot_scatter(x, y, xlabel, ylabel, caption_x, caption_y, out_path):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c="black", s=25, zorder=3)
    x_line = np.linspace(x.min()-0.5, x.max()+0.5, 100)
    ax.plot(x_line, slope*x_line+intercept, "r-", linewidth=2, zorder=2)
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.text(0.95, 0.98, f"Cor={r_val:.2f}", transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="top", ha="right")
    ax.tick_params(axis="both", which="major", labelsize=12, direction="in",
                   top=True, right=True, length=6)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=3)
    ax.minorticks_on()
    for spine in ax.spines.values(): spine.set_linewidth(1.5)
    sig = "99%" if p_val < 0.01 else ("95%" if p_val < 0.05 else "not sig.")
    caption = (f"Scatter diagram of {caption_y} (y) vs.\n"
               f"{caption_x} (x) for {len(x)} MJO events.\n"
               f"Red line: least squares fit.\n"
               f"Cor = {r_val:.2f} (p = {p_val:.4f}),\n"
               f"exceeding the {sig} confidence level.")
    fig.text(0.5, -0.10, caption, ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    Cor={r_val:.2f}, p={p_val:.4f}, N={len(x)} → {out_path.name}")


# ==============================
# 主函数
# ==============================
def main():
    print("=" * 60)
    print("生成 both_daily/ 文件夹（上下层均逐日计算后取事件平均）")
    print("=" * 60)

    ep_dir = OUT_DIR / "event_profile"
    cp_dir = OUT_DIR / "centroid_profile"
    ep_dir.mkdir(parents=True, exist_ok=True)
    cp_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("  Loading data...")
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ds_w = xr.open_dataset(W_NORM_NC)
    ds_q = xr.open_dataset(Q_NORM_NC)
    ds3 = xr.open_dataset(STEP3_NC)
    data = _prepare_field_data(ds_w, ds_q, ds3)
    time_arr = data['time']
    df_ps = pd.read_csv(PHASE_SPEED_CSV)

    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    target_h = np.linspace(0.0, 12.0, 120)

    all_mean_uw = []      # 逐日 up_west 取事件平均
    all_mean_centroid = [] # 逐日 centroid 取事件平均
    all_mean_qmax = []     # 逐日 q_max 取事件平均

    for ev_idx, row in events.iterrows():
        eid = int(row['event_id'])
        ts = pd.Timestamp(row['start_date'])
        te = pd.Timestamp(row['end_date'])
        event_mask = (time_arr >= ts) & (time_arr <= te)
        event_indices = np.where(event_mask)[0]

        if len(event_indices) < 2:
            all_mean_uw.append(np.nan)
            all_mean_centroid.append(np.nan)
            all_mean_qmax.append(np.nan)
            continue

        centers = data['center_lon'][event_indices]
        if np.isfinite(centers).sum() < 2:
            all_mean_uw.append(np.nan)
            all_mean_centroid.append(np.nan)
            all_mean_qmax.append(np.nan)
            continue

        # 计算事件平均场（仅用于底图显示）
        w_h_list, q_h_list = [], []
        for i, idx in enumerate(event_indices):
            c = centers[i]
            if not np.isfinite(c):
                continue
            result = _process_single_day(data, idx, c)
            if result is not None:
                w_h_list.append(result[0])
                q_h_list.append(result[1])

        if len(w_h_list) < 2:
            all_mean_uw.append(np.nan)
            all_mean_centroid.append(np.nan)
            all_mean_qmax.append(np.nan)
            continue

        w_mean = np.nanmean(np.stack(w_h_list), axis=0)
        q_mean = np.nanmean(np.stack(q_h_list), axis=0)

        # --- 逐日计算三个量 ---
        daily_uw = _compute_daily_up_west(data, event_indices, centers)
        daily_centroid = _compute_daily_centroid(data, event_indices, centers)
        daily_qmax = _compute_daily_q_max(data, event_indices, centers)

        # 事件平均值
        mean_uw = float(np.nanmean(daily_uw))
        mean_centroid = float(np.nanmean(daily_centroid))
        mean_qmax = float(np.nanmean(daily_qmax))
        all_mean_uw.append(mean_uw)
        all_mean_centroid.append(mean_centroid)
        all_mean_qmax.append(mean_qmax)

        # event_profile: daily_uw + daily_qmax 散点，均值连线
        plot_profile(eid, w_mean, q_mean, target_rel, target_h,
                     mean_uw, mean_qmax, 'q max (daily)', 'darkgreen',
                     row, ep_dir / f"event_{eid:03d}_profile.png",
                     scatter_uw=daily_uw, scatter_qm=daily_qmax)

        # centroid_profile: 均值 up_west + centroid 连线
        plot_profile(eid, w_mean, q_mean, target_rel, target_h,
                     mean_uw, mean_centroid, 'q centroid (daily)', '#2471A3',
                     row, cp_dir / f"event_{eid:03d}_centroid_profile.png")

        print(f"  Event #{eid} (uw={mean_uw:.1f}°, centroid={mean_centroid:.1f}°, qmax={mean_qmax:.1f}°)")

    # === 6 张散点图 ===
    print("\n  Generating 6 scatter plots...")
    all_mean_uw = np.array(all_mean_uw)
    all_mean_centroid = np.array(all_mean_centroid)
    all_mean_qmax = np.array(all_mean_qmax)

    # 保存 CSV
    df_vals = pd.DataFrame({
        'event_id': events['event_id'].values,
        'daily_up_west': all_mean_uw,
        'daily_centroid': all_mean_centroid,
        'daily_q_max': all_mean_qmax,
    })
    df_vals.to_csv(OUT_DIR / "event_daily_values.csv", index=False)

    df = df_ps.merge(df_vals, on="event_id")
    speed = df["phase_speed_m_s"].values
    uw = df["daily_up_west"].values
    qm = df["daily_q_max"].values
    ct = df["daily_centroid"].values

    # 1. up_west vs phase_speed
    plot_scatter(speed, uw, "Speed (m/s)", "Upper ω West Boundary (°)",
                 "phase speed", "upper ω west (daily mean)",
                 OUT_DIR / "up_west_vs_phase_speed.png")
    # 2. q_max vs phase_speed
    plot_scatter(speed, qm, "Speed (m/s)", "Lower q Max Position (°)",
                 "phase speed", "lower q max (daily mean)",
                 OUT_DIR / "q_max_vs_phase_speed.png")
    # 3. up_west vs q_max
    plot_scatter(uw, qm, "Upper ω West Boundary (°)", "Lower q Max Position (°)",
                 "upper ω west (daily mean)", "lower q max (daily mean)",
                 OUT_DIR / "up_west_vs_q_max.png")
    # 4. centroid vs speed
    plot_scatter(speed, ct, "Speed (m/s)", "q Centroid Position (°)",
                 "phase speed", "q centroid (daily mean)",
                 OUT_DIR / "centroid_vs_speed.png")
    # 5. omega_west vs speed (same as 1, consistent naming)
    plot_scatter(speed, uw, "Speed (m/s)", "ω West Boundary (°)",
                 "phase speed", "ω west (daily mean)",
                 OUT_DIR / "omega_west_vs_speed.png")
    # 6. omega_west vs centroid
    plot_scatter(uw, ct, "ω West Boundary (°)", "q Centroid Position (°)",
                 "upper ω west (daily mean)", "q centroid (daily mean)",
                 OUT_DIR / "omega_west_vs_centroid.png")

    print(f"\n{'='*60}")
    print(f"Done! Results in: {OUT_DIR}")
    print(f"  event_profile/: {len(list(ep_dir.glob('*.png')))} files")
    print(f"  centroid_profile/: {len(list(cp_dir.glob('*.png')))} files")
    print(f"  + 6 scatter plots + CSV")


if __name__ == "__main__":
    main()
