# -*- coding: utf-8 -*-
"""
enso_physical_mechanism.py — ENSO 分组的物理机制合成分析

功能：
    验证不同 ENSO 相位下 MJO 垂直倾斜的物理机制差异，
    分析风切变、水汽、MSE 和环流型的 ENSO 分组合成。
输入：
    era5_mjo_recon_{t,q,u,v,w}_norm_1979-2022.nc, mjo_mvEOF_step3_1979-2022.nc,
    mjo_events_step3_1979-2022.csv, oni.ascii.txt
输出：
    figures/enso_mechanism/enso_physical_mechanism_composite.png
用法：
    python tests/enso_physical_mechanism.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from scipy import stats

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ========================
# PATHS
# ========================
T_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_t_norm_1979-2022.nc"
Q_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
U_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_u_norm_1979-2022.nc"
V_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_v_norm_1979-2022.nc"
W_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\enso_mechanism")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 物理常数
Cp = 1004.0  # J/(kg·K)
Lv = 2.5e6   # J/kg
g = 9.8      # m/s²

# 气压层对应高度 (近似)
LEVEL_TO_HEIGHT = {
    1000: 100, 925: 750, 850: 1500, 700: 3000,
    600: 4200, 500: 5500, 400: 7200, 300: 9200, 200: 12000
}

ENSO_ORDER = ["El Nino", "La Nina", "Neutral"]
ENSO_COLORS = {"El Nino": "#E74C3C", "La Nina": "#3498DB", "Neutral": "#95A5A6"}
AMP_THRESHOLD = 0.5





def load_data():
    """加载所有需要的数据"""
    print("[1] Loading data...")
    
    # 各变量
    ds_t = xr.open_dataset(T_RECON_NC, engine="netcdf4")
    ds_q = xr.open_dataset(Q_RECON_NC, engine="netcdf4")
    ds_u = xr.open_dataset(U_RECON_NC, engine="netcdf4")
    ds_v = xr.open_dataset(V_RECON_NC, engine="netcdf4")
    ds_w = xr.open_dataset(W_RECON_NC, engine="netcdf4")
    
    # Step3 (中心经度、振幅)
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4")
    
    # 事件 + ENSO 分类
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    enso_stats = pd.read_csv(ENSO_STATS_CSV)
    enso_map = dict(zip(enso_stats['event_id'], enso_stats['enso_phase']))
    
    # 提取数据
    time = pd.to_datetime(ds_t["time"].values)
    lon = ds_t["lon"].values
    levels = ds_t["pressure_level"].values if "pressure_level" in ds_t.dims else ds_t["level"].values
    
    T = ds_t["t_mjo_recon_norm"].values  # (time, level, lon)
    q = ds_q["q_mjo_recon_norm"].values
    u = ds_u["u_mjo_recon_norm"].values
    v = ds_v["v_mjo_recon_norm"].values
    w = ds_w["w_mjo_recon_norm"].values
    
    center_lon = ds3["center_lon_track"].values
    amplitude = ds3["amp"].values
    
    print(f"  Loaded: T, q, u, v, w - shape: {T.shape}")
    print(f"  Events: {len(events)}")
    
    return {
        'time': time, 'lon': lon, 'levels': levels,
        'T': T, 'q': q, 'u': u, 'v': v, 'w': w,
        'center_lon': center_lon, 'amplitude': amplitude,
        'events': events, 'enso_map': enso_map
    }


def create_enso_composites(data, variable, lon_half_width=60):
    """
    按 ENSO 分组创建相对经度合成
    
    Args:
        data: 数据字典
        variable: 'T', 'q', 'u', 'v', 'w', 'shear', 'mse'
    
    Returns:
        dict: {enso_phase: composite array (level, rel_lon)}
    """
    events = data['events']
    enso_map = data['enso_map']
    time = data['time']
    lon = data['lon']
    levels = data['levels']
    center_lon = data['center_lon']
    amplitude = data['amplitude']
    dlon = lon[1] - lon[0]
    
    n_rel_lon = int(2 * lon_half_width / dlon) + 1
    rel_lons = np.linspace(-lon_half_width, lon_half_width, n_rel_lon)
    
    # 按 ENSO 分组收集日数据
    enso_days = {phase: [] for phase in ENSO_ORDER}
    
    for _, ev in events.iterrows():
        eid = ev['event_id']
        phase = enso_map.get(eid)
        if phase is None or phase not in ENSO_ORDER:
            continue
        
        start = pd.Timestamp(ev['start_date'])
        end = pd.Timestamp(ev['end_date'])
        mask = (time >= start) & (time <= end)
        day_indices = np.where(mask)[0]
        
        for idx in day_indices:
            c = center_lon[idx]
            amp = amplitude[idx]
            if not np.isfinite(c) or not np.isfinite(amp) or amp < AMP_THRESHOLD:
                continue
            enso_days[phase].append((idx, c, amp))
    
    # 准备变量数据
    if variable == 'shear':
        # 风切变: u200 - u850
        u = data['u']
        idx_200 = np.argmin(np.abs(levels - 200))
        idx_850 = np.argmin(np.abs(levels - 850))
        var_data = u[:, idx_200, :] - u[:, idx_850, :]  # (time, lon)
        is_2d = True
    elif variable == 'mse':
        # MSE = Cp*T + Lv*q (忽略 gz 项，只看异常)
        T = data['T']
        q = data['q']
        mse = Cp * T + Lv * q  # (time, level, lon)
        var_data = mse
        is_2d = False
    elif variable == 'q_low':
        # 低层比湿 (850-700 hPa 平均)
        q = data['q']
        idx_850 = np.argmin(np.abs(levels - 850))
        idx_700 = np.argmin(np.abs(levels - 700))
        var_data = np.mean(q[:, min(idx_850,idx_700):max(idx_850,idx_700)+1, :], axis=1)
        is_2d = True
    elif variable == 'div_low':
        # 低层散度 (du/dx，简化)
        u = data['u']
        idx_850 = np.argmin(np.abs(levels - 850))
        u_850 = u[:, idx_850, :]
        var_data = u_850  # 直接用 u 场代替散度
        is_2d = True
    else:
        var_data = data[variable]
        is_2d = (var_data.ndim == 2)
    
    # 创建合成
    composites = {}
    
    for phase in ENSO_ORDER:
        days = enso_days[phase]
        if len(days) == 0:
            continue
        
        if is_2d:
            stack = np.full((len(days), n_rel_lon), np.nan)
        else:
            stack = np.full((len(days), len(levels), n_rel_lon), np.nan)
        
        for i, (idx, c, amp) in enumerate(days):
            # 相对经度
            rel = (lon - c + 180) % 360 - 180
            
            for j, rl in enumerate(rel_lons):
                # 找最近的经度
                k = np.argmin(np.abs(rel - rl))
                if np.abs(rel[k] - rl) < dlon:
                    if is_2d:
                        stack[i, j] = var_data[idx, k]
                    else:
                        stack[i, :, j] = var_data[idx, :, k]
        
        composites[phase] = np.nanmean(stack, axis=0)
    
    return composites, rel_lons, levels


def plot_all_analyses(data):
    """绑定所有分析的图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # ======== 风切变 ========
    ax1 = axes[0, 0]
    shear_comp, rel_lons, _ = create_enso_composites(data, 'shear')
    
    for phase in ENSO_ORDER:
        if phase in shear_comp:
            ax1.plot(rel_lons, shear_comp[phase], 
                    color=ENSO_COLORS[phase], linewidth=2, label=phase)
    
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('相对经度 (°)', fontsize=11)
    ax1.set_ylabel('风切变 u200-u850 (m/s per amp)', fontsize=11)
    ax1.set_title('(a) 垂直风切变', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-60, 60)
    
    # ======== 低层水汽 ========
    ax2 = axes[0, 1]
    q_comp, rel_lons, _ = create_enso_composites(data, 'q_low')
    
    for phase in ENSO_ORDER:
        if phase in q_comp:
            ax2.plot(rel_lons, q_comp[phase] * 1000,  # 转为 g/kg
                    color=ENSO_COLORS[phase], linewidth=2, label=phase)
    
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('相对经度 (°)', fontsize=11)
    ax2.set_ylabel('低层比湿 q (g/kg per amp)', fontsize=11)
    ax2.set_title('(b) 低层水汽 (850-700 hPa)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-60, 60)
    
    # ======== MSE (选取低层) ========
    ax3 = axes[1, 0]
    mse_comp, rel_lons, levels = create_enso_composites(data, 'mse')
    
    # 低层 MSE (850 hPa)
    idx_850 = np.argmin(np.abs(levels - 850))
    for phase in ENSO_ORDER:
        if phase in mse_comp:
            ax3.plot(rel_lons, mse_comp[phase][idx_850, :] / 1000,  # 转为 kJ/kg
                    color=ENSO_COLORS[phase], linewidth=2, label=phase)
    
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('相对经度 (°)', fontsize=11)
    ax3.set_ylabel('MSE (kJ/kg per amp)', fontsize=11)
    ax3.set_title('(c) 低层 MSE (850 hPa)', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-60, 60)
    
    # ======== 低层 u 场 (代表环流) ========
    ax4 = axes[1, 1]
    u_comp, rel_lons, _ = create_enso_composites(data, 'div_low')
    
    for phase in ENSO_ORDER:
        if phase in u_comp:
            ax4.plot(rel_lons, u_comp[phase], 
                    color=ENSO_COLORS[phase], linewidth=2, label=phase)
    
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('相对经度 (°)', fontsize=11)
    ax4.set_ylabel('u_850 (m/s per amp)', fontsize=11)
    ax4.set_title('(d) 低层纬向风 (850 hPa)', fontsize=13, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-60, 60)
    
    plt.tight_layout()
    fig_path = FIG_DIR / "enso_physical_mechanism_composite.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()
    
    return shear_comp, q_comp, mse_comp, u_comp, rel_lons


def analyze_all_regions(composites, rel_lons, name):
    """
    分析西侧、中间、东侧三个区域的 ENSO 差异
    西侧: -60 ~ -30°
    中间: -20 ~ +20°
    东侧: +30 ~ +60°
    """
    print(f"\n--- {name} 三区域分析 ---")
    
    west_mask = (rel_lons >= -60) & (rel_lons <= -30)
    center_mask = (rel_lons >= -20) & (rel_lons <= 20)
    east_mask = (rel_lons >= 30) & (rel_lons <= 60)
    
    print(f"{'位置':<10} {'El Nino':>12} {'La Nina':>12} {'Neutral':>12} {'最大者':>12}")
    print("-" * 60)
    
    for region, mask, region_name in [('西侧(后)', west_mask, 'west'), 
                                       ('中间', center_mask, 'center'),
                                       ('东侧(前)', east_mask, 'east')]:
        vals = {}
        for phase in ENSO_ORDER:
            if phase in composites:
                comp = composites[phase]
                if comp.ndim == 1:
                    vals[phase] = np.nanmean(comp[mask])
                else:
                    # 取 850 hPa (index 5)
                    vals[phase] = np.nanmean(comp[5, mask]) if comp.shape[0] > 5 else np.nanmean(comp[0, mask])
        
        max_phase = max(vals, key=lambda x: abs(vals[x])) if vals else 'N/A'
        print(f"{region:<10} {vals.get('El Nino', np.nan):>12.4f} {vals.get('La Nina', np.nan):>12.4f} {vals.get('Neutral', np.nan):>12.4f} {max_phase:>12}")


def main():
    print("="*70)
    print("ENSO 分组物理机制合成分析")
    print("="*70)
    
    data = load_data()
    
    print("\n[2] Creating composites...")
    shear_comp, q_comp, mse_comp, u_comp, rel_lons = plot_all_analyses(data)
    
    print("\n[3] Analyzing all regions...")
    analyze_all_regions(shear_comp, rel_lons, "风切变")
    analyze_all_regions(q_comp, rel_lons, "低层水汽")
    analyze_all_regions(mse_comp, rel_lons, "MSE")
    analyze_all_regions(u_comp, rel_lons, "低层u风")
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
