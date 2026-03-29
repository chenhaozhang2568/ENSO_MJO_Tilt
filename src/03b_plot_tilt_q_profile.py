# -*- coding: utf-8 -*-
"""
绘制指定日期的 MJO 垂直截面结构图（新 tilt_q 定义验证）。

上半部分：omega 高层场（400-200 hPa）填色 + omega=0 等值线
下半部分：q 低层场（1000-850 hPa）填色

标注：
  - 上层点：omega 高层西边界 (up_west_rel)
  - 下层点：q 低层最大值位置 (q_max_rel)
  - 连线 + 标注经度 + tilt_q 值
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d, Akima1DInterpolator
from pathlib import Path

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# === Paths ===
W_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
Q_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
TILT_Q_NC  = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"
OUT_DIR    = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_profile"

# === 参数 ===
SMOOTH_WINDOW = 10
CSA_TARGET_DLON = 0.25
REL_LON_RANGE = (-90, 90)
N_RANDOM = 5    # 随机画几天

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}


def _smooth_1d(profile, window):
    """沿经度方向做滑动平均平滑"""
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


def plot_one_day(idx, ds_w, ds_q, ds3, ds_tilt_q, out_dir):
    """绘制单日垂直剖面图"""
    time = pd.to_datetime(ds_w['time'].values)
    w_raw = ds_w['w_mjo_recon_norm'].values          # (time, level, lon)
    q_raw = ds_q['q_mjo_recon_norm'].values
    levels_w = ds_w['pressure_level'].values
    levels_q = ds_q['pressure_level'].values
    lon_w = ds_w['lon'].values
    lon_q = ds_q['lon'].values
    center_lon = ds3['center_lon_track'].values

    date = time[idx]
    c = center_lon[idx]

    # 读取 tilt_q 预计算结果
    tilt_q_time = pd.to_datetime(ds_tilt_q['time'].values)
    tilt_q_idx = np.where(tilt_q_time == date)[0]
    if len(tilt_q_idx) == 0:
        print(f"  [SKIP] {date.strftime('%Y-%m-%d')}: not in tilt_q file")
        return
    tilt_q_idx = tilt_q_idx[0]

    q_max_rel = float(ds_tilt_q['q_max_rel'].values[tilt_q_idx])
    up_west_rel = float(ds_tilt_q['up_west_rel'].values[tilt_q_idx])
    tilt_val = float(ds_tilt_q['tilt_q'].values[tilt_q_idx])

    if not np.isfinite(tilt_val):
        print(f"  [SKIP] {date.strftime('%Y-%m-%d')}: tilt_q is NaN")
        return

    print(f"  Plotting: {date.strftime('%Y-%m-%d')}, center={c:.1f}°, "
          f"q_max_rel={q_max_rel:.1f}°, up_west={up_west_rel:.1f}°, tilt_q={tilt_val:.1f}°")

    # === omega 处理 ===
    # 转为 0-360
    lon_w_360 = np.where(lon_w < 0, lon_w + 360, lon_w)
    w_sort = np.argsort(lon_w_360)
    lon_w_360 = lon_w_360[w_sort]

    rel_lon_w = lon_w_360 - c
    # 截取 -90 到 90
    mask_w = (rel_lon_w >= REL_LON_RANGE[0]) & (rel_lon_w <= REL_LON_RANGE[1])
    rel_lons_w = rel_lon_w[mask_w]

    w_day = w_raw[idx, :, :][:, w_sort][:, mask_w]  # (level, lon_subset)
    heights_w = np.array([LEVEL_TO_HEIGHT[int(p)] for p in levels_w])

    # 插值到 0.25° 再平滑
    target_rel = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + CSA_TARGET_DLON, CSA_TARGET_DLON)
    w_interp_smooth = np.full((len(levels_w), len(target_rel)), np.nan)
    for k in range(len(levels_w)):
        interped = _cubic_spline_interp_1d(rel_lons_w, w_day[k, :], target_rel)
        w_interp_smooth[k, :] = _smooth_1d(interped, SMOOTH_WINDOW)

    # 插值到均匀高度网格
    target_h = np.linspace(0.0, 12.0, 120)
    w_h_interp = np.full((len(target_h), len(target_rel)), np.nan)
    for j in range(len(target_rel)):
        col = w_interp_smooth[:, j]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            f = interp1d(heights_w[valid], col[valid], kind='linear',
                         bounds_error=False, fill_value=np.nan)
            w_h_interp[:, j] = f(target_h)

    # === q 处理 ===
    lon_q_360 = np.where(lon_q < 0, lon_q + 360, lon_q)
    q_sort = np.argsort(lon_q_360)
    lon_q_360 = lon_q_360[q_sort]

    rel_lon_q = lon_q_360 - c
    mask_q = (rel_lon_q >= REL_LON_RANGE[0]) & (rel_lon_q <= REL_LON_RANGE[1])
    rel_lons_q = rel_lon_q[mask_q]

    q_day = q_raw[idx, :, :][:, q_sort][:, mask_q]
    heights_q = np.array([LEVEL_TO_HEIGHT[int(p)] for p in levels_q])

    q_interp_smooth = np.full((len(levels_q), len(target_rel)), np.nan)
    for k in range(len(levels_q)):
        interped = _cubic_spline_interp_1d(rel_lons_q, q_day[k, :], target_rel)
        q_interp_smooth[k, :] = _smooth_1d(interped, SMOOTH_WINDOW)

    q_h_interp = np.full((len(target_h), len(target_rel)), np.nan)
    for j in range(len(target_rel)):
        col = q_interp_smooth[:, j]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            f = interp1d(heights_q[valid], col[valid], kind='linear',
                         bounds_error=False, fill_value=np.nan)
            q_h_interp[:, j] = f(target_h)

    # === 定义显示区域 ===
    # 上层 omega：400-200 hPa → 高度 7.2-12.0 km
    up_h_min = LEVEL_TO_HEIGHT[400]   # 7.2
    up_h_max = LEVEL_TO_HEIGHT[200]   # 12.0
    # 下层 q：1000-850 hPa → 高度 0.1-1.5 km
    low_h_min = LEVEL_TO_HEIGHT[1000] # 0.1
    low_h_max = LEVEL_TO_HEIGHT[850]  # 1.5

    # tilt 两端点的高度：各层中点
    up_h_mid = (up_h_min + up_h_max) / 2.0
    low_h_mid = (low_h_min + low_h_max) / 2.0

    # === Plot ===
    fig, ax = plt.subplots(figsize=(14, 7))

    X, Y = np.meshgrid(target_rel, target_h)

    # omega 填色（仅 400 hPa 以上，即 height >= 7.2 km）
    w_display = np.where((target_h >= up_h_min)[:, None], w_h_interp, np.nan)
    vmax_w = np.nanmax(np.abs(w_display)) * 0.8
    if vmax_w < 1e-6:
        vmax_w = 0.01
    norm_w = TwoSlopeNorm(vmin=-vmax_w, vcenter=0, vmax=vmax_w)
    cf_w = ax.contourf(X, Y, w_display,
                       levels=np.linspace(-vmax_w, vmax_w, 21),
                       cmap='RdBu_r', norm=norm_w, extend='both', alpha=0.7)

    # q 填色（仅 850 hPa 以下，即 height <= 1.5 km）
    q_display = np.where((target_h <= low_h_max)[:, None], q_h_interp, np.nan)
    vmax_q = np.nanmax(np.abs(q_display)) * 0.8
    if vmax_q < 1e-10:
        vmax_q = 1e-5
    norm_q = TwoSlopeNorm(vmin=-vmax_q, vcenter=0, vmax=vmax_q)
    cf_q = ax.contourf(X, Y, q_display,
                       levels=np.linspace(-vmax_q, vmax_q, 21),
                       cmap='BrBG', norm=norm_q, extend='both', alpha=0.9)

    # omega=0 等值线（仅 400 hPa 以上）
    w_contour = np.where((target_h >= up_h_min)[:, None], w_h_interp, np.nan)
    ax.contour(X, Y, w_contour, levels=[0], colors='black', linewidths=2.0)

    # 水平分隔线：标示上下层范围
    for h in [low_h_max, up_h_min]:
        ax.axhline(h, color='gray', lw=1.2, ls='-', alpha=0.6)

    # 层标签（放在右侧避免遮挡）
    ax.text(REL_LON_RANGE[1] - 2, (low_h_min + low_h_max) / 2,
            'q (1000–850 hPa)', fontsize=9, color='darkgreen',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    ax.text(REL_LON_RANGE[1] - 2, (up_h_min + up_h_max) / 2,
            'ω (400–200 hPa)', fontsize=9, color='darkred',
            fontweight='bold', va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

    # === Tilt 标注 ===
    if np.isfinite(q_max_rel) and np.isfinite(up_west_rel):
        # 连线
        ax.plot([up_west_rel, q_max_rel], [up_h_mid, low_h_mid],
                'o-', color='gold', markersize=14, markeredgecolor='black',
                markeredgewidth=2, lw=3.5, zorder=10,
                label=f'Tilt_q = {tilt_val:.1f}°')

        # 上层点标注
        ax.annotate(f'Up ω west: {up_west_rel:.1f}°',
                    (up_west_rel, up_h_mid),
                    textcoords='offset points', xytext=(15, 10),
                    fontsize=10, color='darkgoldenrod', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='darkgoldenrod', lw=1.5))

        # 下层点标注
        ax.annotate(f'Low q max: {q_max_rel:.1f}°',
                    (q_max_rel, low_h_mid),
                    textcoords='offset points', xytext=(15, -20),
                    fontsize=10, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', fc='honeydew', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

        # tilt 值标注在连线中点
        mid_x = (up_west_rel + q_max_rel) / 2
        mid_y = (up_h_mid + low_h_mid) / 2
        ax.text(mid_x + 5, mid_y, f'Δlon = {tilt_val:.1f}°',
                fontsize=12, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow',
                          edgecolor='gold', alpha=0.9))

    # 对流中心线
    ax.axvline(0, color='limegreen', lw=2.5, alpha=0.8, label='Convective Center')

    # 坐标轴
    ax.set_ylim(0, 12)
    ax.set_xlim(REL_LON_RANGE)
    ax.set_ylabel('Height (km)', fontsize=12)
    ax.set_xlabel('Relative Longitude (°)', fontsize=12)

    # 右轴：气压
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in pticks])
    ax2.set_yticklabels([str(p) for p in pticks])
    ax2.set_ylabel('Pressure (hPa)', fontsize=12)

    # colorbars（使用 inset_axes 精确定位）
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # omega colorbar：右上方（高度 45%，顶部对齐）
    cax_w = inset_axes(ax, width='2%', height='45%', loc='upper right',
                       bbox_to_anchor=(0.14, 0.0, 1, 1), bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar_w = fig.colorbar(cf_w, cax=cax_w, orientation='vertical')
    cbar_w.set_label('omega (norm)', fontsize=8)
    # q colorbar：右下方（高度 25%，底部对齐）
    cax_q = inset_axes(ax, width='2%', height='45%', loc='lower right',
                       bbox_to_anchor=(0.14, 0.0, 1, 1), bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar_q = fig.colorbar(cf_q, cax=cax_q, orientation='vertical')
    cbar_q.set_label('q (norm)', fontsize=8)

    title = (f"MJO Vertical Structure - {date.strftime('%Y-%m-%d')} "
             f"(smoothed, window={SMOOTH_WINDOW})\n"
             f"Center: {c:.1f}E  |  "
             f"Tilt_q = q_max({q_max_rel:.1f}) - w_west({up_west_rel:.1f}) = {tilt_val:.1f}")
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    out = Path(out_dir) / f"tilt_q_profile_{date.strftime('%Y%m%d')}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    ds_w = xr.open_dataset(W_NORM_NC)
    ds_q = xr.open_dataset(Q_NORM_NC)
    ds3 = xr.open_dataset(STEP3_NC)
    ds_tilt_q = xr.open_dataset(TILT_Q_NC)

    time = pd.to_datetime(ds_w['time'].values)
    center_lon = ds3['center_lon_track'].values
    amp = ds3['amp'].values

    tilt_q_time = pd.to_datetime(ds_tilt_q['time'].values)
    tilt_q_vals = ds_tilt_q['tilt_q'].values

    # 筛选有效日期
    valid_days = []
    for t in range(len(time)):
        if time[t].month not in {11, 12, 1, 2, 3, 4}:
            continue
        if not np.isfinite(center_lon[t]) or amp[t] < 0.5:
            continue
        # 匹配 tilt_q 时间
        tq_idx = np.where(tilt_q_time == time[t])[0]
        if len(tq_idx) == 0:
            continue
        if not np.isfinite(tilt_q_vals[tq_idx[0]]):
            continue
        valid_days.append(t)

    print(f"  Valid days for plotting: {len(valid_days)}")

    rng = np.random.default_rng()
    chosen = rng.choice(valid_days, size=min(N_RANDOM, len(valid_days)), replace=False)

    for idx in chosen:
        plot_one_day(idx, ds_w, ds_q, ds3, ds_tilt_q, out_dir)

    print(f"\nDone. Figures saved to: {out_dir}")


if __name__ == '__main__':
    main()
