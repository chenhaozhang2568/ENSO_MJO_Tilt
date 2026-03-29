# -*- coding: utf-8 -*-
"""
绘制指定日期的 omega 垂直截面结构图。
横轴：相对经度（对流中心为0）
纵轴：高度（km，均匀分布），右轴显示气压层
包含：omega 填色（经20点滑动平均平滑）+ omega=0 等值线（粗黑线）
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# === Paths ===
W_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
TILT_NC    = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
OUT_DIR    = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\omega_structure"

# === 平滑参数 (与 03 代码一致) ===
SMOOTH_WINDOW = 10  # 滑动平均窗口

# === 指定日期（None 则随机抽取） ===
TARGET_DATE = "1985-02-22"  # 设为 "YYYY-MM-DD" 指定日期，设为 None 随机抽取

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}

def _smooth_1d(profile, window):
    """沿经度方向做滑动平均平滑"""
    if window <= 1:
        return profile
    kernel = np.ones(window) / window
    # 对 NaN 做处理：先用 0 替代 NaN，再除以有效点数
    valid = np.isfinite(profile).astype(float)
    filled = np.where(np.isfinite(profile), profile, 0.0)
    smoothed = np.convolve(filled, kernel, mode='same')
    count = np.convolve(valid, kernel, mode='same')
    count[count < 1e-10] = np.nan
    return smoothed / count

def main():
    from pathlib import Path
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    ds_w = xr.open_dataset(W_NORM_NC)
    ds3 = xr.open_dataset(STEP3_NC)
    ds_tilt = xr.open_dataset(TILT_NC)

    w = ds_w['w_mjo_recon_norm'].values          # (time, level, lon)
    time = pd.to_datetime(ds_w['time'].values)
    levels = ds_w['pressure_level'].values        # [1000,925,...,200]
    lon = ds_w['lon'].values
    center_lon = ds3['center_lon_track'].values
    tilt = ds_tilt['tilt'].values
    amp = ds3['amp'].values

    heights = np.array([LEVEL_TO_HEIGHT[int(p)] for p in levels])

    if TARGET_DATE is not None:
        # 指定日期模式
        target = pd.Timestamp(TARGET_DATE)
        idx = np.where(time == target)[0]
        if len(idx) == 0:
            raise ValueError(f"Date {TARGET_DATE} not found in dataset")
        idx = idx[0]
    else:
        # 随机抽取模式
        valid_days = []
        for t in range(len(time)):
            if time[t].month not in {11, 12, 1, 2, 3, 4}:
                continue
            if not np.isfinite(tilt[t]) or not np.isfinite(center_lon[t]):
                continue
            if amp[t] < 0.5:
                continue
            valid_days.append(t)
        rng = np.random.default_rng()
        idx = rng.choice(valid_days)

    date = time[idx]
    c = center_lon[idx]
    tilt_val = tilt[idx]

    print(f"Selected: {date.strftime('%Y-%m-%d')}, center={c:.1f}°, tilt={tilt_val:.1f}°")

    # 相对经度
    rel_lon = (lon - c + 180) % 360 - 180
    sort_order = np.argsort(rel_lon)
    rel_lon_sorted = rel_lon[sort_order]

    # 截取 -90 到 +90
    lon_mask = (rel_lon_sorted >= -90) & (rel_lon_sorted <= 90)
    rel_lons = rel_lon_sorted[lon_mask]

    # omega 数据 (level, rel_lon)
    w_day = w[idx, :, :][:, sort_order][:, lon_mask]

    # === 对每层做滑动平均平滑（与 03 代码 SMOOTH_WINDOW=20 一致） ===
    w_smoothed = np.full_like(w_day, np.nan)
    for k in range(len(levels)):
        w_smoothed[k, :] = _smooth_1d(w_day[k, :], SMOOTH_WINDOW)

    # 插值到均匀高度网格
    target_h = np.linspace(0.0, 13.0, 130)
    w_interp = np.full((len(target_h), len(rel_lons)), np.nan)
    for j in range(len(rel_lons)):
        col = w_smoothed[:, j]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            f = interp1d(heights[valid], col[valid], kind='linear',
                         bounds_error=False, fill_value=np.nan)
            w_interp[:, j] = f(target_h)

    # === Plot ===
    fig, ax = plt.subplots(figsize=(14, 7))

    X, Y = np.meshgrid(rel_lons, target_h)

    # 填色
    vmax = np.nanmax(np.abs(w_interp)) * 0.8
    if vmax < 1e-6:
        vmax = 0.01
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cf = ax.contourf(X, Y, w_interp,
                     levels=np.linspace(-vmax, vmax, 21),
                     cmap='RdBu_r', norm=norm, extend='both')

    # omega=0 等值线（粗黑线）
    ax.contour(X, Y, w_interp, levels=[0], colors='black', linewidths=2.5)

    # === 读取预计算的低层/高层西边界（来自更新后的 03 代码输出） ===
    LOW_LEVELS = [int(p) for p in levels if 600 <= p <= 1000]
    UP_LEVELS  = [int(p) for p in levels if 200 <= p <= 400]
    low_h_mid = np.mean([LEVEL_TO_HEIGHT[p] for p in LOW_LEVELS])
    up_h_mid  = np.mean([LEVEL_TO_HEIGHT[p] for p in UP_LEVELS])

    low_west_rel = ds_tilt['low_west_rel'].values[idx]
    up_west_rel  = ds_tilt['up_west_rel'].values[idx]

    print(f"  Low west boundary: {low_west_rel:.1f}°, Upper west boundary: {up_west_rel:.1f}°")
    print(f"  Tilt (low-up): {low_west_rel - up_west_rel:.1f}°")

    # 画西边界和 tilt 线
    if np.isfinite(low_west_rel) and np.isfinite(up_west_rel):
        ax.plot([up_west_rel, low_west_rel], [up_h_mid, low_h_mid],
                'o-', color='gold', markersize=12, markeredgecolor='black',
                markeredgewidth=1.5, lw=3, zorder=10,
                label=f'West Boundary (tilt={low_west_rel-up_west_rel:.1f}°)')
        ax.annotate(f'Upper: {up_west_rel:.1f}°', (up_west_rel, up_h_mid),
                    textcoords='offset points', xytext=(10, 10),
                    fontsize=9, color='darkgoldenrod', fontweight='bold')
        ax.annotate(f'Lower: {low_west_rel:.1f}°', (low_west_rel, low_h_mid),
                    textcoords='offset points', xytext=(10, -15),
                    fontsize=9, color='darkgoldenrod', fontweight='bold')

    # 对流中心线
    ax.axvline(0, color='limegreen', lw=2.5, alpha=0.8, label='Convective Center')

    # 左轴：高度
    ax.set_ylim(0, 13)
    ax.set_ylabel('Height (km)', fontsize=12)
    ax.set_xlabel('Relative Longitude (°)', fontsize=12)

    # 右轴：气压层
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in pticks])
    ax2.set_yticklabels([str(p) for p in pticks])
    ax2.set_ylabel('Pressure (hPa)', fontsize=12)

    # colorbar
    cbar = fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.12, shrink=0.8)
    cbar.set_label('ω (Pa/s, normalized)', fontsize=10)

    title = (f"MJO Omega Structure — {date.strftime('%Y-%m-%d')} (smoothed, window={SMOOTH_WINDOW})\n"
             f"Center: {c:.1f}°E, Tilt: {tilt_val:.1f}°")
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='upper right')

    out = Path(OUT_DIR) / f"omega_structure_{date.strftime('%Y%m%d')}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

if __name__ == '__main__':
    main()
