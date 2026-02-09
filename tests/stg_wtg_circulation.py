# -*- coding: utf-8 -*-
"""
stg_wtg_circulation.py: STG/WTG 垂直环流合成图 (v2)

修改:
1. 垂直轴按高度均匀分布（不是气压）
2. 使用归一化后的重构数据
3. 调小垂直速度比例
4. 平滑处理
5. 增强显著性打点可见性
6. 横轴扩展到 -90 到 180 度
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.ndimage import gaussian_filter
from pathlib import Path

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
U_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_u_norm_1979-2022.nc"
W_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
TILT_NC = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\circulation")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 气压层对应高度 (km) - 标准大气
LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}

WINTER_MONTHS = {11, 12, 1, 2, 3, 4}


def load_data():
    """加载 u, w, 中心经度, tilt 数据（使用归一化重构数据）"""
    print("[1] Loading NORMALIZED reconstructed data...")
    
    ds_u = xr.open_dataset(U_RECON_NC, engine="netcdf4")
    ds_w = xr.open_dataset(W_RECON_NC, engine="netcdf4")
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4")
    ds_tilt = xr.open_dataset(TILT_NC, engine="netcdf4")
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    
    # 使用归一化后的重构数据
    u = ds_u["u_mjo_recon_norm"].values  # (time, level, lon)
    w = ds_w["w_mjo_recon_norm"].values
    
    time = pd.to_datetime(ds_u["time"].values)
    levels = ds_u["pressure_level"].values
    lon = ds_u["lon"].values
    
    center_lon = ds3["center_lon_track"].values
    amp = ds3["amp"].values
    tilt = ds_tilt["tilt"].values
    
    print(f"  ✓ Using NORMALIZED data: u_mjo_recon_norm, w_mjo_recon_norm")
    print(f"  Shape: u={u.shape}, w={w.shape}")
    print(f"  Levels: {levels}")
    print(f"  Events: {len(events)}")
    
    return {
        'u': u, 'w': w, 'time': time, 'levels': levels, 'lon': lon,
        'center_lon': center_lon, 'amp': amp, 'tilt': tilt, 'events': events
    }


def classify_stg_wtg(data):
    """按 tilt 标准差分类事件为 STG 和 WTG"""
    events = data['events']
    time = data['time']
    tilt = data['tilt']
    
    event_tilts = []
    for _, ev in events.iterrows():
        start = pd.Timestamp(ev['start_date'])
        end = pd.Timestamp(ev['end_date'])
        mask = (time >= start) & (time <= end)
        tv = tilt[mask]
        valid = np.isfinite(tv)
        if valid.sum() > 0:
            event_tilts.append({
                'event_id': ev['event_id'],
                'start': start, 'end': end,
                'mean_tilt': np.nanmean(tv[valid])
            })
    
    df = pd.DataFrame(event_tilts)
    
    mean_tilt = df['mean_tilt'].mean()
    std_tilt = df['mean_tilt'].std()
    
    stg_events = df[df['mean_tilt'] > mean_tilt + 0.5 * std_tilt]
    wtg_events = df[df['mean_tilt'] < mean_tilt - 0.5 * std_tilt]
    
    print(f"\n[2] STG/WTG classification:")
    print(f"  Mean tilt: {mean_tilt:.2f}°, Std: {std_tilt:.2f}°")
    print(f"  STG threshold: > {mean_tilt + 0.5 * std_tilt:.2f}° → {len(stg_events)} events")
    print(f"  WTG threshold: < {mean_tilt - 0.5 * std_tilt:.2f}° → {len(wtg_events)} events")
    
    return stg_events, wtg_events


def create_composite(data, event_list, lon_range=(-90, 180)):
    """
    创建相对经度合成
    横轴范围: lon_range (默认 -90 到 180)
    """
    time = data['time']
    lon = data['lon']
    levels = data['levels']
    u = data['u']
    w = data['w']
    center_lon = data['center_lon']
    amp = data['amp']
    
    dlon = lon[1] - lon[0]
    rel_lon_min, rel_lon_max = lon_range
    n_rel_lon = int((rel_lon_max - rel_lon_min) / dlon) + 1
    rel_lons = np.linspace(rel_lon_min, rel_lon_max, n_rel_lon)
    
    # 收集所有样本
    u_samples = []
    w_samples = []
    
    for _, ev in event_list.iterrows():
        start = ev['start']
        end = ev['end']
        mask = (time >= start) & (time <= end)
        day_indices = np.where(mask)[0]
        
        for idx in day_indices:
            if time[idx].month not in WINTER_MONTHS:
                continue
            c = center_lon[idx]
            a = amp[idx]
            if not np.isfinite(c) or not np.isfinite(a) or a < 0.5:
                continue
            
            # 相对经度
            rel = (lon - c + 180) % 360 - 180
            
            u_day = np.full((len(levels), n_rel_lon), np.nan)
            w_day = np.full((len(levels), n_rel_lon), np.nan)
            
            for j, rl in enumerate(rel_lons):
                k = np.argmin(np.abs(rel - rl))
                if np.abs(rel[k] - rl) < dlon:
                    u_day[:, j] = u[idx, :, k]
                    w_day[:, j] = w[idx, :, k]
            
            u_samples.append(u_day)
            w_samples.append(w_day)
    
    u_stack = np.array(u_samples)
    w_stack = np.array(w_samples)
    
    u_mean = np.nanmean(u_stack, axis=0)
    w_mean = np.nanmean(w_stack, axis=0)
    
    # t 检验
    n = u_stack.shape[0]
    w_std = np.nanstd(w_stack, axis=0, ddof=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        t_w = w_mean / (w_std / np.sqrt(n))
    
    p_w = 2 * (1 - stats.t.cdf(np.abs(t_w), df=n-1))
    sig_mask = p_w < 0.05
    
    return {
        'u': u_mean, 'w': w_mean,
        'sig_mask': sig_mask,
        'rel_lons': rel_lons, 'levels': levels,
        'n_samples': n
    }


def interpolate_to_height(data, levels, target_heights):
    """
    将气压层数据插值到等高度坐标
    """
    heights = np.array([LEVEL_TO_HEIGHT.get(int(p), 5.0) for p in levels])
    
    from scipy.interpolate import interp1d
    
    n_lon = data.shape[1]
    result = np.full((len(target_heights), n_lon), np.nan)
    
    for j in range(n_lon):
        valid = np.isfinite(data[:, j])
        if valid.sum() >= 2:
            f = interp1d(heights[valid], data[valid, j], kind='linear', 
                        bounds_error=False, fill_value=np.nan)
            result[:, j] = f(target_heights)
    
    return result


def plot_circulation(ax, comp, title, show_ylabel=True):
    """
    绘制单个垂直环流图
    - 垂直轴按高度均匀分布
    - 平滑处理
    - 增强显著性点
    """
    
    rel_lons = comp['rel_lons']
    levels = comp['levels']
    u_orig = comp['u']
    w_orig = comp['w']
    sig_mask_orig = comp['sig_mask']
    
    # 定义等间距高度坐标 (0.5 到 12 km)
    target_heights = np.linspace(0.5, 12, 24)  # 24 层
    
    # 将数据插值到等高度坐标
    u = interpolate_to_height(u_orig, levels, target_heights)
    w = interpolate_to_height(w_orig, levels, target_heights)
    sig_raw = interpolate_to_height(sig_mask_orig.astype(float), levels, target_heights)
    sig_mask = sig_raw > 0.5  # 重新二值化
    
    # 平滑处理 (高斯滤波)
    u_smooth = gaussian_filter(np.nan_to_num(u, nan=0), sigma=1.0)
    w_smooth = gaussian_filter(np.nan_to_num(w, nan=0), sigma=1.0)
    
    # 恢复 NaN
    nan_mask = np.isnan(u) | np.isnan(w)
    u_smooth[nan_mask] = np.nan
    w_smooth[nan_mask] = np.nan
    
    # 创建网格 (高度坐标)
    X, Y = np.meshgrid(rel_lons, target_heights)
    
    # 标准化垂直速度
    # 注意：使用固定的 omega 量级来标准化，而不是合成后的 std
    # 典型 MJO omega 异常约 0.01 Pa/s
    w_ref = 0.01  # Pa/s，参考值
    w_norm = w_smooth / w_ref
    
    # 填色图 (颜色间隔 0.2)
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=0.5)
    cf = ax.contourf(X, Y, w_norm, levels=np.arange(-1.0, 0.6, 0.2), 
                     cmap='RdBu_r', norm=norm, extend='both')
    
    # 显著性打点 (增大点)
    for i in range(len(target_heights)):
        for j in range(0, len(rel_lons), 4):
            if sig_mask[i, j]:
                ax.plot(rel_lons[j], target_heights[i], 'k.', markersize=2.5, alpha=0.8)
    
    # 矢量图
    # 调小垂直速度比例
    u_plot = u_smooth.copy()
    w_plot = -w_smooth * 800  # 减小放大倍数
    
    # 抽样显示
    skip_x = 6
    skip_y = 2
    
    ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x], 
              u_plot[::skip_y, ::skip_x], w_plot[::skip_y, ::skip_x],
              color='black', scale=40, width=0.004, headwidth=4, headlength=4,
              pivot='middle')
    
    # 对流中心绿线
    ax.axvline(0, color='limegreen', linewidth=3.5, alpha=0.95)
    
    # 设置轴 (高度坐标)
    ax.set_ylim(0.5, 12)
    ax.set_xlim(-90, 180)
    ax.set_xticks(np.arange(-90, 181, 30))
    ax.set_xticklabels([f'{int(x)}°' for x in np.arange(-90, 181, 30)], fontsize=9)
    
    if show_ylabel:
        ax.set_ylabel('Height (km)', fontsize=11)
    
    ax.set_xlabel('Relative Longitude', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 右侧 y 轴：气压 (hPa)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    # 在高度坐标系中标注对应的气压
    height_to_pressure = {v: k for k, v in LEVEL_TO_HEIGHT.items()}
    pressure_ticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    height_ticks = [LEVEL_TO_HEIGHT[p] for p in pressure_ticks]
    ax2.set_yticks(height_ticks)
    ax2.set_yticklabels([str(p) for p in pressure_ticks])
    ax2.set_ylabel('Pressure (hPa)', fontsize=11)
    
    return cf


def main():
    print("="*70)
    print("STG/WTG 垂直环流合成图 (v2)")
    print("="*70)
    
    data = load_data()
    stg_events, wtg_events = classify_stg_wtg(data)
    
    print("\n[3] Creating STG composite...")
    stg_comp = create_composite(data, stg_events, lon_range=(-90, 180))
    print(f"  STG samples: {stg_comp['n_samples']}")
    
    print("\n[4] Creating WTG composite...")
    wtg_comp = create_composite(data, wtg_events, lon_range=(-90, 180))
    print(f"  WTG samples: {wtg_comp['n_samples']}")
    
    print("\n[5] Plotting...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    cf1 = plot_circulation(axes[0], stg_comp, '(a) Strong Tilt', show_ylabel=True)
    cf2 = plot_circulation(axes[1], wtg_comp, '(b) Weak Tilt', show_ylabel=False)
    
    # 添加共享 colorbar
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    cbar = fig.colorbar(cf2, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Standardized Anomalous Vertical Velocity', fontsize=10)
    cbar.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.1, 0.2, 0.3, 0.4])
    
    plt.subplots_adjust(bottom=0.12, wspace=0.25)
    
    fig_path = FIG_DIR / "stg_wtg_vertical_circulation_v2.png"
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
