# -*- coding: utf-8 -*-
"""
enso_circulation.py: ENSO 分类垂直环流合成图

按 El Nino / La Nina / Neutral 分为三张图
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
from scipy.interpolate import interp1d
from pathlib import Path

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
U_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_u_norm_1979-2022.nc"
W_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ONI_FILE = r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\circulation")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 气压层对应高度 (km)
LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}

WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
ONI_THRESHOLD = 0.5


def load_oni():
    """加载 ONI 指数"""
    oni = pd.read_csv(ONI_FILE, sep=r'\s+', header=0, engine='python')
    month_map = {'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 'AMJ': 5, 'MJJ': 6,
                'JJA': 7, 'JAS': 8, 'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12}
    records = []
    for _, row in oni.iterrows():
        seas = row['SEAS']
        year = int(row['YR'])
        anom = row['ANOM']
        if seas in month_map:
            records.append({'year': year, 'month': month_map[seas], 'oni': anom})
    oni_df = pd.DataFrame(records)
    oni_df['date'] = pd.to_datetime(oni_df[['year', 'month']].assign(day=1))
    return oni_df.set_index('date')['oni']


def classify_enso(date, oni_series):
    """分类 ENSO 相"""
    target = pd.Timestamp(year=date.year, month=date.month, day=1)
    if target in oni_series.index:
        oni_val = oni_series.loc[target]
    else:
        idx = oni_series.index.get_indexer([target], method='nearest')[0]
        if idx >= 0 and idx < len(oni_series):
            oni_val = oni_series.iloc[idx]
        else:
            return None
    if oni_val >= ONI_THRESHOLD:
        return 'El Nino'
    elif oni_val <= -ONI_THRESHOLD:
        return 'La Nina'
    return 'Neutral'


def load_data():
    """加载数据"""
    print("[1] Loading data...")
    
    ds_u = xr.open_dataset(U_RECON_NC, engine="netcdf4")
    ds_w = xr.open_dataset(W_RECON_NC, engine="netcdf4")
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4")
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    oni_series = load_oni()
    
    u = ds_u["u_mjo_recon_norm"].values
    w = ds_w["w_mjo_recon_norm"].values
    
    time = pd.to_datetime(ds_u["time"].values)
    levels = ds_u["pressure_level"].values
    lon = ds_u["lon"].values
    
    center_lon = ds3["center_lon_track"].values
    amp = ds3["amp"].values
    
    print(f"  Shape: u={u.shape}, w={w.shape}")
    print(f"  Events: {len(events)}")
    
    return {
        'u': u, 'w': w, 'time': time, 'levels': levels, 'lon': lon,
        'center_lon': center_lon, 'amp': amp, 'events': events, 'oni': oni_series
    }


def classify_events_by_enso(data):
    """按 ENSO 相分类事件"""
    events = data['events']
    oni_series = data['oni']
    
    enso_events = {'El Nino': [], 'La Nina': [], 'Neutral': []}
    
    for _, ev in events.iterrows():
        start = pd.Timestamp(ev['start_date'])
        end = pd.Timestamp(ev['end_date'])
        center_date = start + (end - start) / 2
        
        enso = classify_enso(center_date, oni_series)
        if enso:
            enso_events[enso].append({
                'event_id': ev['event_id'],
                'start': start, 'end': end
            })
    
    print(f"\n[2] ENSO classification:")
    for phase, evs in enso_events.items():
        print(f"  {phase}: {len(evs)} events")
    
    return enso_events


def create_composite(data, event_list, lon_range=(-90, 180)):
    """创建合成"""
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
    
    u_samples = []
    w_samples = []
    
    for ev in event_list:
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
    
    if len(u_samples) == 0:
        return None
    
    u_stack = np.array(u_samples)
    w_stack = np.array(w_samples)
    
    u_mean = np.nanmean(u_stack, axis=0)
    w_mean = np.nanmean(w_stack, axis=0)
    
    n = u_stack.shape[0]
    w_std = np.nanstd(w_stack, axis=0, ddof=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        t_w = w_mean / (w_std / np.sqrt(n))
    
    p_w = 2 * (1 - stats.t.cdf(np.abs(t_w), df=n-1))
    sig_mask = p_w < 0.05
    
    return {
        'u': u_mean, 'w': w_mean,
        'u_stack': u_stack, 'w_stack': w_stack,  # 保存原始样本用于差值显著性检验
        'sig_mask': sig_mask,
        'rel_lons': rel_lons, 'levels': levels,
        'n_samples': n
    }


def interpolate_to_height(data, levels, target_heights):
    """插值到等高度坐标"""
    heights = np.array([LEVEL_TO_HEIGHT.get(int(p), 5.0) for p in levels])
    
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
    """绘制垂直环流图"""
    
    rel_lons = comp['rel_lons']
    levels = comp['levels']
    u_orig = comp['u']
    w_orig = comp['w']
    sig_mask_orig = comp['sig_mask']
    
    target_heights = np.linspace(0.5, 12, 24)
    
    u = interpolate_to_height(u_orig, levels, target_heights)
    w = interpolate_to_height(w_orig, levels, target_heights)
    sig_raw = interpolate_to_height(sig_mask_orig.astype(float), levels, target_heights)
    sig_mask = sig_raw > 0.5
    
    u_smooth = gaussian_filter(np.nan_to_num(u, nan=0), sigma=1.0)
    w_smooth = gaussian_filter(np.nan_to_num(w, nan=0), sigma=1.0)
    
    nan_mask = np.isnan(u) | np.isnan(w)
    u_smooth[nan_mask] = np.nan
    w_smooth[nan_mask] = np.nan
    
    X, Y = np.meshgrid(rel_lons, target_heights)
    
    w_ref = 0.01
    w_norm = w_smooth / w_ref
    
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=0.5)
    cf = ax.contourf(X, Y, w_norm, levels=np.arange(-1.0, 0.6, 0.2), 
                     cmap='RdBu_r', norm=norm, extend='both')
    
    for i in range(len(target_heights)):
        for j in range(0, len(rel_lons), 4):
            if sig_mask[i, j]:
                ax.plot(rel_lons[j], target_heights[i], 'k.', markersize=2.5, alpha=0.8)
    
    u_plot = u_smooth.copy()
    w_plot = -w_smooth * 800
    
    skip_x = 6
    skip_y = 2
    
    ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x], 
              u_plot[::skip_y, ::skip_x], w_plot[::skip_y, ::skip_x],
              color='black', scale=40, width=0.004, headwidth=4, headlength=4,
              pivot='middle')
    
    ax.axvline(0, color='limegreen', linewidth=3.5, alpha=0.95)
    
    ax.set_ylim(0.5, 12)
    ax.set_xlim(-90, 180)
    ax.set_xticks(np.arange(-90, 181, 30))
    ax.set_xticklabels([f'{int(x)}°' for x in np.arange(-90, 181, 30)], fontsize=9)
    
    if show_ylabel:
        ax.set_ylabel('Height (km)', fontsize=11)
    
    ax.set_xlabel('Relative Longitude', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    pressure_ticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    height_ticks = [LEVEL_TO_HEIGHT[p] for p in pressure_ticks]
    ax2.set_yticks(height_ticks)
    ax2.set_yticklabels([str(p) for p in pressure_ticks])
    ax2.set_ylabel('Pressure (hPa)', fontsize=11)
    
    return cf


def main():
    print("="*70)
    print("ENSO 分类垂直环流合成图")
    print("="*70)
    
    data = load_data()
    enso_events = classify_events_by_enso(data)
    
    # 计算三个 ENSO 相的合成
    composites = {}
    enso_order = ['El Nino', 'La Nina', 'Neutral']
    
    for i, phase in enumerate(enso_order):
        print(f"\n[{i+3}] Creating {phase} composite...")
        event_list = enso_events[phase]
        comp = create_composite(data, event_list, lon_range=(-90, 180))
        if comp is not None:
            print(f"  Samples: {comp['n_samples']}")
            composites[phase] = comp
    
    # ============ 第一组图：三个 ENSO 相 (1x3) ============
    print("\n[6] Creating ENSO phases figure (1x3)...")
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 6))
    
    labels1 = ['(a) El Nino', '(b) La Nina', '(c) Neutral']
    
    for i, (phase, label) in enumerate(zip(enso_order, labels1)):
        if phase in composites:
            cf = plot_circulation(axes1[i], composites[phase], label, show_ylabel=(i==0))
    
    cbar_ax1 = fig1.add_axes([0.25, 0.02, 0.5, 0.02])
    cbar1 = fig1.colorbar(cf, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('Standardized Anomalous Vertical Velocity', fontsize=10)
    cbar1.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4])
    
    plt.subplots_adjust(bottom=0.12, wspace=0.25)
    
    fig_path1 = FIG_DIR / "enso_circulation_phases.png"
    fig1.savefig(fig_path1, dpi=200, bbox_inches='tight')
    print(f"  Saved: {fig_path1}")
    plt.close(fig1)
    
    # ============ 第二组图：三个差值 (1x3) ============
    print("\n[7] Creating difference figure (1x3)...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    
    diff_pairs = [
        ('El Nino', 'La Nina', '(a) El Nino - La Nina'),
        ('El Nino', 'Neutral', '(b) El Nino - Neutral'),
        ('La Nina', 'Neutral', '(c) La Nina - Neutral'),
    ]
    
    for i, (p1, p2, label) in enumerate(diff_pairs):
        if p1 in composites and p2 in composites:
            # 计算差值
            u_diff = composites[p1]['u'] - composites[p2]['u']
            w_diff = composites[p1]['w'] - composites[p2]['w']
            
            # Two-sample t-test 检验差值显著性
            w_stack1 = composites[p1]['w_stack']
            w_stack2 = composites[p2]['w_stack']
            n1, n2 = w_stack1.shape[0], w_stack2.shape[0]
            
            # 计算显著性
            w_mean1 = np.nanmean(w_stack1, axis=0)
            w_mean2 = np.nanmean(w_stack2, axis=0)
            w_var1 = np.nanvar(w_stack1, axis=0, ddof=1)
            w_var2 = np.nanvar(w_stack2, axis=0, ddof=1)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                se = np.sqrt(w_var1/n1 + w_var2/n2)
                t_stat = (w_mean1 - w_mean2) / se
                df = (w_var1/n1 + w_var2/n2)**2 / (
                    (w_var1/n1)**2/(n1-1) + (w_var2/n2)**2/(n2-1)
                )  # Welch's approximation
            
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=df))
            sig_mask_diff = p_val < 0.05
            
            diff_comp = {
                'u': u_diff, 'w': w_diff,
                'sig_mask': sig_mask_diff,
                'rel_lons': composites[p1]['rel_lons'],
                'levels': composites[p1]['levels'],
                'n_samples': n1 + n2
            }
            cf_diff = plot_circulation_diff(axes2[i], diff_comp, label, show_ylabel=(i==0))
    
    cbar_ax2 = fig2.add_axes([0.25, 0.02, 0.5, 0.02])
    cbar2 = fig2.colorbar(cf_diff, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Difference in Vertical Velocity', fontsize=10)
    
    plt.subplots_adjust(bottom=0.12, wspace=0.25)
    
    fig_path2 = FIG_DIR / "enso_circulation_diff.png"
    fig2.savefig(fig_path2, dpi=200, bbox_inches='tight')
    print(f"  Saved: {fig_path2}")
    plt.close(fig2)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


def plot_circulation_diff(ax, comp, title, show_ylabel=True):
    """绘制差值图（使用对称的红蓝色标）"""
    
    rel_lons = comp['rel_lons']
    levels = comp['levels']
    u_orig = comp['u']
    w_orig = comp['w']
    sig_mask_orig = comp['sig_mask']
    
    target_heights = np.linspace(0.5, 12, 24)
    
    u = interpolate_to_height(u_orig, levels, target_heights)
    w = interpolate_to_height(w_orig, levels, target_heights)
    sig_raw = interpolate_to_height(sig_mask_orig.astype(float), levels, target_heights)
    sig_mask = sig_raw > 0.5
    
    u_smooth = gaussian_filter(np.nan_to_num(u, nan=0), sigma=1.0)
    w_smooth = gaussian_filter(np.nan_to_num(w, nan=0), sigma=1.0)
    
    nan_mask = np.isnan(u) | np.isnan(w)
    u_smooth[nan_mask] = np.nan
    w_smooth[nan_mask] = np.nan
    
    X, Y = np.meshgrid(rel_lons, target_heights)
    
    w_ref = 0.01
    w_norm = w_smooth / w_ref
    
    # 对称色标 (差值较小，扩大显示范围)
    cf = ax.contourf(X, Y, w_norm, levels=np.linspace(-0.15, 0.15, 13), 
                     cmap='RdBu_r', extend='both')
    
    # 显著性区域用斜线阴影标记（不与矢量重叠）
    ax.contourf(X, Y, sig_mask.astype(float), levels=[0.5, 1.5],
                colors='none', hatches=['...'], alpha=0)
    
    # 矢量 (放大差值风矢量)
    u_plot = u_smooth.copy()
    w_plot = -w_smooth * 600  # 减小 w/u 比例
    
    skip_x = 6
    skip_y = 2
    
    ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x], 
              u_plot[::skip_y, ::skip_x], w_plot[::skip_y, ::skip_x],
              color='black', scale=10, width=0.004, headwidth=4, headlength=4,
              pivot='middle')  # scale 进一步减小使箭头更大
    
    ax.axvline(0, color='limegreen', linewidth=3.5, alpha=0.95)
    
    ax.set_ylim(0.5, 12)
    ax.set_xlim(-90, 180)
    ax.set_xticks(np.arange(-90, 181, 30))
    ax.set_xticklabels([f'{int(x)}' for x in np.arange(-90, 181, 30)], fontsize=9)
    
    if show_ylabel:
        ax.set_ylabel('Height (km)', fontsize=11)
    ax.set_xlabel('Relative Longitude', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    pressure_ticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    height_ticks = [LEVEL_TO_HEIGHT[p] for p in pressure_ticks]
    ax2.set_yticks(height_ticks)
    ax2.set_yticklabels([str(p) for p in pressure_ticks])
    ax2.set_ylabel('Pressure (hPa)', fontsize=11)
    
    return cf
    
    rel_lons = comp['rel_lons']
    levels = comp['levels']
    u_orig = comp['u']
    w_orig = comp['w']
    
    target_heights = np.linspace(0.5, 12, 24)
    
    u = interpolate_to_height(u_orig, levels, target_heights)
    w = interpolate_to_height(w_orig, levels, target_heights)
    
    u_smooth = gaussian_filter(np.nan_to_num(u, nan=0), sigma=1.0)
    w_smooth = gaussian_filter(np.nan_to_num(w, nan=0), sigma=1.0)
    
    nan_mask = np.isnan(u) | np.isnan(w)
    u_smooth[nan_mask] = np.nan
    w_smooth[nan_mask] = np.nan
    
    X, Y = np.meshgrid(rel_lons, target_heights)
    
    # 差值用对称的色标
    w_ref = 0.01
    w_norm = w_smooth / w_ref
    
    # 对称色标
    vmax = max(abs(np.nanmin(w_norm)), abs(np.nanmax(w_norm)))
    vmax = min(vmax, 0.5)  # 限制范围
    
    cf = ax.contourf(X, Y, w_norm, levels=np.linspace(-vmax, vmax, 11), 
                     cmap='RdBu_r', extend='both')
    
    # 矢量
    u_plot = u_smooth.copy()
    w_plot = -w_smooth * 800
    
    skip_x = 6
    skip_y = 2
    
    ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x], 
              u_plot[::skip_y, ::skip_x], w_plot[::skip_y, ::skip_x],
              color='black', scale=40, width=0.004, headwidth=4, headlength=4,
              pivot='middle')
    
    ax.axvline(0, color='limegreen', linewidth=3.5, alpha=0.95)
    
    ax.set_ylim(0.5, 12)
    ax.set_xlim(-90, 180)
    ax.set_xticks(np.arange(-90, 181, 30))
    ax.set_xticklabels([f'{int(x)}' for x in np.arange(-90, 181, 30)], fontsize=9)
    
    ax.set_ylabel('Height (km)', fontsize=11)
    ax.set_xlabel('Relative Longitude', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    pressure_ticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    height_ticks = [LEVEL_TO_HEIGHT[p] for p in pressure_ticks]
    ax2.set_yticks(height_ticks)
    ax2.set_yticklabels([str(p) for p in pressure_ticks])
    ax2.set_ylabel('Pressure (hPa)', fontsize=11)
    
    return cf


if __name__ == "__main__":
    main()
