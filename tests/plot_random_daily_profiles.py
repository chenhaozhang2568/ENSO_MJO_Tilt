# -*- coding: utf-8 -*-
"""
plot_random_daily_profiles.py
每次运行随机挑选 5 天，绘制当日 omega/q 场剖面图 + 当日 up_west 和 centroid 标注。

与 both_daily/centroid_profile 的区别：
  - 底图：当日场（非事件平均场）
  - 标注点：当日计算的 up_west 和 centroid（非事件均值）

输出：
  outputs/figures/tilt_q_diagnose/random_daily_profile/
    daily_YYYY-MM-DD_event_NNN.png  (×5)
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import Akima1DInterpolator, interp1d
from pathlib import Path

# === 路径 ===
BASE_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_diagnose")
OUT_DIR = BASE_DIR / "random_daily_profile"

W_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
Q_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"

# === 参数 ===
N_SAMPLES = 5
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


def _process_single_day_full(data, idx, center):
    """处理单日场，返回插值到统一网格的 w_h, q_h, target_rel, target_h"""
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    target_h = np.linspace(0.0, 12.0, 120)

    # omega
    rel_lon_w = data['lon_w_360'] - center
    rel_lon_w = np.where(rel_lon_w > 180, rel_lon_w - 360, rel_lon_w)
    rel_lon_w = np.where(rel_lon_w < -180, rel_lon_w + 360, rel_lon_w)
    sort_idx_w = np.argsort(rel_lon_w)
    rel_lon_w = rel_lon_w[sort_idx_w]
    mask_w = (rel_lon_w >= REL_LON_RANGE[0]) & (rel_lon_w <= REL_LON_RANGE[1])
    rel_lons_w = rel_lon_w[mask_w]
    w_day = data['w_raw'][idx, :, :][:, data['w_sort']][:, sort_idx_w][:, mask_w]
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

    # q
    rel_lon_q = data['lon_q_360'] - center
    rel_lon_q = np.where(rel_lon_q > 180, rel_lon_q - 360, rel_lon_q)
    rel_lon_q = np.where(rel_lon_q < -180, rel_lon_q + 360, rel_lon_q)
    sort_idx_q = np.argsort(rel_lon_q)
    rel_lon_q = rel_lon_q[sort_idx_q]
    mask_q = (rel_lon_q >= REL_LON_RANGE[0]) & (rel_lon_q <= REL_LON_RANGE[1])
    rel_lons_q = rel_lon_q[mask_q]
    q_day = data['q_raw'][idx, :, :][:, data['q_sort']][:, sort_idx_q][:, mask_q]
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


def _compute_single_day_up_west(data, idx, center):
    """计算单日上层 omega=0 西边界"""
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    rel_lon_w = data['lon_w_360'] - center
    rel_lon_w = np.where(rel_lon_w > 180, rel_lon_w - 360, rel_lon_w)
    rel_lon_w = np.where(rel_lon_w < -180, rel_lon_w + 360, rel_lon_w)
    sort_idx_w = np.argsort(rel_lon_w)
    rel_lon_w = rel_lon_w[sort_idx_w]
    mask_w = (rel_lon_w >= REL_LON_RANGE[0]) & (rel_lon_w <= REL_LON_RANGE[1])
    w_day = data['w_raw'][idx, :, :][:, data['w_sort']][:, sort_idx_w][:, mask_w]
    up_mask = (data['levels_w'] >= UP_LAYER[1]) & (data['levels_w'] <= UP_LAYER[0])
    w_up_mean = np.nanmean(w_day[up_mask, :], axis=0)
    w_up_interp = _cubic_spline_interp_1d(rel_lon_w[mask_w], w_up_mean, target_rel)
    w_up_smooth = _smooth_1d(w_up_interp, SMOOTH_WINDOW)
    return _ascent_boundary_zero(target_rel, w_up_smooth)


def _compute_single_day_centroid(data, idx, center):
    """计算单日低层 q centroid"""
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    rel_lon_q = data['lon_q_360'] - center
    rel_lon_q = np.where(rel_lon_q > 180, rel_lon_q - 360, rel_lon_q)
    rel_lon_q = np.where(rel_lon_q < -180, rel_lon_q + 360, rel_lon_q)
    sort_idx_q = np.argsort(rel_lon_q)
    rel_lon_q = rel_lon_q[sort_idx_q]
    mask_q = (rel_lon_q >= REL_LON_RANGE[0]) & (rel_lon_q <= REL_LON_RANGE[1])
    q_day = data['q_raw'][idx, :, :][:, data['q_sort']][:, sort_idx_q][:, mask_q]
    low_mask = (data['levels_q'] >= Q_LOW_LAYER[1]) & (data['levels_q'] <= Q_LOW_LAYER[0])
    q_low_mean = np.nanmean(q_day[low_mask, :], axis=0)
    q_interp = _cubic_spline_interp_1d(rel_lon_q[mask_q], q_low_mean, target_rel)
    q_smooth = _smooth_1d(q_interp, SMOOTH_WINDOW)
    return _find_q_centroid(target_rel, q_smooth)


def plot_daily_profile(w_h, q_h, target_rel, target_h,
                       up_west, centroid, date_str, eid, out_path):
    """绘制单日场剖面图 + 当日 up_west 和 centroid 标注"""
    up_h_min, up_h_max = LEVEL_TO_HEIGHT[400], LEVEL_TO_HEIGHT[200]
    low_h_min, low_h_max = LEVEL_TO_HEIGHT[1000], LEVEL_TO_HEIGHT[850]
    up_h_mid = (up_h_min + up_h_max) / 2.0
    low_h_mid = (low_h_min + low_h_max) / 2.0

    fig, ax = plt.subplots(figsize=(14, 7))
    X, Y = np.meshgrid(target_rel, target_h)

    # omega 填色（400 hPa 以上）
    w_display = np.where((target_h >= up_h_min)[:, None], w_h, np.nan)
    vmax_w = np.nanmax(np.abs(w_display)) * 0.8
    if vmax_w < 1e-6 or not np.isfinite(vmax_w):
        vmax_w = 0.01
    cf_w = ax.contourf(X, Y, w_display, levels=np.linspace(-vmax_w, vmax_w, 21),
                       cmap='RdBu_r', norm=TwoSlopeNorm(vmin=-vmax_w, vcenter=0, vmax=vmax_w),
                       extend='both', alpha=0.7)

    # q 填色（850 hPa 以下）
    q_display = np.where((target_h <= low_h_max)[:, None], q_h, np.nan)
    vmax_q = np.nanmax(np.abs(q_display)) * 0.8
    if vmax_q < 1e-10 or not np.isfinite(vmax_q):
        vmax_q = 1e-5
    cf_q = ax.contourf(X, Y, q_display, levels=np.linspace(-vmax_q, vmax_q, 21),
                       cmap='BrBG', norm=TwoSlopeNorm(vmin=-vmax_q, vcenter=0, vmax=vmax_q),
                       extend='both', alpha=0.9)

    # omega=0 等值线
    w_contour = np.where((target_h >= up_h_min)[:, None], w_h, np.nan)
    ax.contour(X, Y, w_contour, levels=[0], colors='black', linewidths=2.0)

    # 水平分隔线
    for h in [low_h_max, up_h_min]:
        ax.axhline(h, color='gray', lw=1.2, ls='-', alpha=0.6)

    # 层标签
    ax.text(REL_LON_RANGE[1]-2, (low_h_min+low_h_max)/2, 'q (1000–850 hPa)',
            fontsize=9, color='darkgreen', fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.text(REL_LON_RANGE[1]-2, (up_h_min+up_h_max)/2, 'ω (400–200 hPa)',
            fontsize=9, color='darkred', fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # 标注点 + 连线
    if np.isfinite(up_west) and np.isfinite(centroid):
        tilt_val = centroid - up_west
        ax.plot([up_west, centroid], [up_h_mid, low_h_mid],
                'o-', color='gold', markersize=14, markeredgecolor='black',
                markeredgewidth=2, lw=3.5, zorder=10,
                label=f'Tilt (centroid) = {tilt_val:.1f}°')

        ax.annotate(f'ω west: {up_west:.1f}°', (up_west, up_h_mid),
                    textcoords='offset points', xytext=(15, 10), fontsize=10,
                    color='darkgoldenrod', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=1.5))

        ax.annotate(f'q centroid: {centroid:.1f}°', (centroid, low_h_mid),
                    textcoords='offset points', xytext=(15, -20), fontsize=10,
                    color='#2471A3', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#EBF5FB', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#2471A3', lw=1.5))

        mid_x = (up_west + centroid) / 2
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
                       bbox_to_anchor=(0.14, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    fig.colorbar(cf_w, cax=cax_w).set_label('omega (norm)', fontsize=8)
    cax_q = inset_axes(ax, width='2%', height='45%', loc='lower right',
                       bbox_to_anchor=(0.14, 0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    fig.colorbar(cf_q, cax=cax_q).set_label('q (norm)', fontsize=8)

    ax.set_title(f"Event #{eid} — {date_str} (Daily Field)",
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 60)
    print("随机采样 5 天绘制逐日场剖面图")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("  Loading data...")
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ds_w = xr.open_dataset(W_NORM_NC)
    ds_q = xr.open_dataset(Q_NORM_NC)
    ds3 = xr.open_dataset(STEP3_NC)
    data = _prepare_field_data(ds_w, ds_q, ds3)
    time_arr = data['time']

    # 收集所有有效的 (event_id, time_index, center_lon) 候选
    candidates = []
    for _, row in events.iterrows():
        eid = int(row['event_id'])
        ts = pd.Timestamp(row['start_date'])
        te = pd.Timestamp(row['end_date'])
        event_mask = (time_arr >= ts) & (time_arr <= te)
        indices = np.where(event_mask)[0]
        for idx in indices:
            c = data['center_lon'][idx]
            if np.isfinite(c):
                candidates.append((eid, idx, c))

    print(f"  总候选天数: {len(candidates)}")

    # 随机选 5 天
    rng = np.random.default_rng()
    chosen = rng.choice(len(candidates), size=min(N_SAMPLES, len(candidates)), replace=False)

    for i, ci in enumerate(chosen):
        eid, idx, center = candidates[ci]
        date_str = str(time_arr[idx].date())
        print(f"\n  [{i+1}/{N_SAMPLES}] Event #{eid}, {date_str}, center={center:.1f}°E")

        # 处理当日场
        w_h, q_h, target_rel, target_h = _process_single_day_full(data, idx, center)

        # 当日的 up_west 和 centroid
        up_west = _compute_single_day_up_west(data, idx, center)
        centroid = _compute_single_day_centroid(data, idx, center)

        tilt = centroid - up_west if (np.isfinite(up_west) and np.isfinite(centroid)) else np.nan
        print(f"    up_west={up_west:.1f}°, centroid={centroid:.1f}°, tilt={tilt:.1f}°")

        out_path = OUT_DIR / f"daily_{date_str}_event_{eid:03d}.png"
        plot_daily_profile(w_h, q_h, target_rel, target_h,
                           up_west, centroid, date_str, eid, out_path)
        print(f"    Saved: {out_path.name}")

    print(f"\n{'='*60}")
    print(f"Done! {N_SAMPLES} daily profiles saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
