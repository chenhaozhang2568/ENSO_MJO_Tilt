# -*- coding: utf-8 -*-
"""
tilt_2d_analysis.py — 2D MJO Tilt 综合分析

功能：
    使用 3D omega 重建数据（保留纬度维度），分析 Tilt 在纬度-阈值空间的分布，
    生成 omega 合成场、边界叠加、2D 热力图、ENSO 主导区域及差异图（共 5 张）。
输入：
    era5_mjo_recon_w_norm_3d_1979-2022.nc, mjo_mvEOF_step3_1979-2022.nc,
    mjo_events_step3_1979-2022.csv, oni.ascii.txt
输出：
    figures/mjo_3d_structure/ 下的 5 张分析图
用法：
    python tests/tilt_2d_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy import stats

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
W_NORM_3D_NC = Path(r"E:\Datas\Derived\era5_mjo_recon_w_norm_3d_1979-2022.nc")
STEP3_NC     = Path(r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc")
EVENTS_CSV   = Path(r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv")
ENSO_STATS_CSV = Path(r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv")
OUT_DIR      = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\mjo_3d_structure")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# SETTINGS
# ======================
THRESHOLDS_HALF = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
WINTER_MONTHS   = {11, 12, 1, 2, 3, 4}
LOW_LAYER       = (1000.0, 600.0)
UP_LAYER        = (400.0, 200.0)
PIVOT_DELTA_DEG = 10.0
MIN_VALID_POINTS = 5
AMP_THRESHOLD   = 0.5
LON_HALF_WIDTH  = 60
P_VALUE_THRESHOLD = 0.10
SIGMA_SMOOTH    = 1.0
SMOOTH_WINDOW   = 1          # 与 03 一致：1=不平滑

ENSO_ORDER  = ["El Nino", "La Nina", "Neutral"]
ENSO_COLORS = {"El Nino": "#E74C3C", "La Nina": "#3498DB", "Neutral": "#95A5A6"}


# ======================
# HELPERS
# ======================



def calc_boundary(rel_lon, w, threshold_frac):
    """计算上升区西/东边界"""
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return np.nan, np.nan

    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)

    if SMOOTH_WINDOW > 1 and len(ww) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        ww = np.convolve(ww, kernel, mode='same')

    win = (rr >= -PIVOT_DELTA_DEG) & (rr <= PIVOT_DELTA_DEG)
    if win.any():
        j0 = int(np.nanargmin(ww[win]))
        pivot_idx = np.where(win)[0][j0]
    else:
        pivot_idx = int(np.nanargmin(ww))

    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return np.nan, np.nan

    thr = threshold_frac * wmin

    # West
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:
            west_idx = min(i + 1, pivot_idx)
            break
    if west_idx is None:
        return np.nan, np.nan

    # East
    east_idx = None
    for i in range(pivot_idx, len(ww)):
        if ww[i] >= thr:
            east_idx = max(i - 1, pivot_idx)
            break
    if east_idx is None:
        return np.nan, np.nan

    west = float(rr[west_idx])
    east = float(rr[east_idx])

    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return np.nan, np.nan

    return west, east


def load_3d_omega():
    """加载 3D omega 和 MJO 追踪数据"""
    print("[1] Loading data...")
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon = ds3['center_lon_track'].values
    mjo_amp = ds3['amp'].values
    time_mjo = pd.to_datetime(ds3.time.values)
    ds3.close()
    
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])
    
    ds = xr.open_dataset(W_NORM_3D_NC)
    data = ds['w_mjo_recon_norm_3d']
    if "pressure_level" in data.dims:
        data = data.rename({"pressure_level": "level"})
    
    time_data = pd.to_datetime(data.time.values)
    levels = data.level.values
    lats = data.lat.values
    lon = data.lon.values
    
    low_mask = (levels >= LOW_LAYER[1]) & (levels <= LOW_LAYER[0])
    up_mask = (levels >= UP_LAYER[1]) & (levels <= UP_LAYER[0])
    
    dlon = np.abs(lon[1] - lon[0])
    n_rel_bins = int(2 * LON_HALF_WIDTH / dlon) + 1
    rel_lons = np.linspace(-LON_HALF_WIDTH, LON_HALF_WIDTH, n_rel_bins)
    
    print(f"  Events: {len(events)}")
    print(f"  Low layer: {levels[low_mask]} hPa")
    print(f"  Upper layer: {levels[up_mask]} hPa")
    
    return (center_lon, mjo_amp, time_mjo, events, data, time_data,
            levels, lats, lon, low_mask, up_mask, rel_lons)


def compute_daily_samples(center_lon, mjo_amp, time_mjo, events, data,
                          time_data, lats, lon, low_mask, up_mask, rel_lons,
                          enso_map=None):
    """
    遍历所有事件所有天，采样到相对坐标系，计算各纬度的边界和 tilt。
    
    Returns:
        samples: dict with keys 'low_sample', 'up_sample' for omega composites
        tilt_data_west/east: {enso_phase: list of (n_half, n_lats) arrays}
        n_samples, enso_counts
    """
    lon_360 = np.mod(lon, 360)
    n_half = len(THRESHOLDS_HALF)
    n_lats = len(lats)
    amp_ok = np.isfinite(mjo_amp) & (mjo_amp > AMP_THRESHOLD)
    
    # Collectors
    all_low_composites = []
    all_up_composites = []
    tilt_west = {p: [] for p in ENSO_ORDER} if enso_map is not None else {'All': []}
    tilt_east = {p: [] for p in ENSO_ORDER} if enso_map is not None else {'All': []}
    enso_counts = {p: 0 for p in ENSO_ORDER} if enso_map is not None else {'All': 0}
    n_samples = 0
    
    print("\n[2] Processing MJO events...")
    for ev_idx, (_, ev) in enumerate(events.iterrows()):
        start = ev['start_date']
        end = ev['end_date']
        
        # ENSO classification (复用管线预分类 CSV)
        if enso_map is not None:
            enso = enso_map.get(ev['event_id'])
            if enso is None or enso not in ENSO_ORDER:
                continue
        else:
            enso = 'All'
        
        mask = (time_mjo >= start) & (time_mjo <= end)
        day_indices = np.where(mask)[0]
        
        for idx in day_indices:
            if time_mjo[idx].month not in WINTER_MONTHS:
                continue
            clon = center_lon[idx]
            if not np.isfinite(clon) or not amp_ok[idx]:
                continue
            
            t = time_mjo[idx]
            data_idx = np.where(time_data == t)[0]
            if len(data_idx) == 0:
                continue
            data_idx = data_idx[0]
            
            daily_data = data.isel(time=data_idx).values  # (level, lat, lon)
            if np.all(np.isnan(daily_data)):
                continue
            
            # Layer means
            w_low_full = np.nanmean(daily_data[low_mask, :, :], axis=0)
            w_up_full = np.nanmean(daily_data[up_mask, :, :], axis=0)
            
            # Sample to relative lon grid
            clon_360 = np.mod(clon, 360)
            low_sample = np.zeros((n_lats, len(rel_lons)))
            up_sample = np.zeros((n_lats, len(rel_lons)))
            
            for j, rlon in enumerate(rel_lons):
                target_lon = np.mod(clon_360 + rlon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - target_lon))
                low_sample[:, j] = w_low_full[:, lon_idx]
                up_sample[:, j] = w_up_full[:, lon_idx]
            
            all_low_composites.append(low_sample)
            all_up_composites.append(up_sample)
            
            # Compute tilt at each (threshold, lat)
            daily_tw = np.full((n_half, n_lats), np.nan)
            daily_te = np.full((n_half, n_lats), np.nan)
            
            for lat_idx in range(n_lats):
                for thr_idx, thr_val in enumerate(THRESHOLDS_HALF):
                    lw, le = calc_boundary(rel_lons, low_sample[lat_idx, :], thr_val)
                    uw, ue = calc_boundary(rel_lons, up_sample[lat_idx, :], thr_val)
                    
                    if np.isfinite(lw) and np.isfinite(uw):
                        daily_tw[thr_idx, lat_idx] = lw - uw
                    if np.isfinite(le) and np.isfinite(ue):
                        daily_te[thr_idx, lat_idx] = le - ue
            
            tilt_west[enso].append(daily_tw)
            tilt_east[enso].append(daily_te)
            enso_counts[enso] += 1
            n_samples += 1
        
        if (ev_idx + 1) % 20 == 0:
            print(f"  Processed {ev_idx + 1}/{len(events)} events, {n_samples} samples")
    
    print(f"\n  Total samples: {n_samples}")
    if enso_map is not None:
        for p in ENSO_ORDER:
            print(f"    {p}: {enso_counts[p]}")
    
    # Mean composites for omega field plots
    mean_low = np.nanmean(all_low_composites, axis=0)   # (lat, rel_lon)
    mean_up = np.nanmean(all_up_composites, axis=0)
    mean_low = gaussian_filter(mean_low, sigma=SIGMA_SMOOTH)
    mean_up = gaussian_filter(mean_up, sigma=SIGMA_SMOOTH)
    
    return (mean_low, mean_up, tilt_west, tilt_east, 
            n_samples, enso_counts)


# ======================
# PLOT FUNCTIONS
# ======================
def plot_omega_composite(mean_low, mean_up, lats, rel_lons, n_samples):
    """图1: 低层/高层 omega 复合场 + tilt 条形图 (3 panels)"""
    print("\n[Plot 1] Omega composite + tilt...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    vmax = np.nanpercentile(np.abs([mean_low, mean_up]), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    X, Y = np.meshgrid(rel_lons, lats)
    
    for ax, field, label in [(axes[0], mean_low, 'Low'), (axes[1], mean_up, 'Upper')]:
        layer = LOW_LAYER if label == 'Low' else UP_LAYER
        cf = ax.contourf(X, Y, field, levels=21, cmap='RdBu_r', norm=norm)
        ax.contour(X, Y, field, levels=[0], colors='k', linewidths=1.5)
        ax.set_xlabel('Relative Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title(f'{label} Layer ({layer[0]}-{layer[1]} hPa)\nω Composite', fontweight='bold')
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        fig.colorbar(cf, ax=ax, shrink=0.9)
    
    # Panel 3: tilt bar chart by latitude (using threshold = 0%)
    ax3 = axes[2]
    low_w = np.array([calc_boundary(rel_lons, mean_low[i, :], 0.0)[0] for i in range(len(lats))])
    up_w = np.array([calc_boundary(rel_lons, mean_up[i, :], 0.0)[0] for i in range(len(lats))])
    tilt = low_w - up_w
    colors = ['red' if t > 0 else 'blue' for t in tilt]
    ax3.barh(lats, tilt, height=np.abs(lats[1]-lats[0])*0.8, color=colors, alpha=0.7)
    ax3.axvline(0, color='k', linewidth=1.5)
    ax3.axvline(np.nanmean(tilt), color='green', linestyle='--', linewidth=2,
                label=f'Mean={np.nanmean(tilt):.1f}°')
    ax3.set_xlabel('Tilt (°)')
    ax3.set_ylabel('Latitude (°)')
    ax3.set_title(f'Tilt by Latitude (N={n_samples})\nRed=westward tilt', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    out = OUT_DIR / "01_omega_composite_tilt.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_boundary_overlay(mean_low, mean_up, lats, rel_lons, n_samples):
    """图2: ω 复合场 + 边界线叠加"""
    print("\n[Plot 2] West boundary overlay...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    vmax = np.nanpercentile(np.abs(mean_low), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    X, Y = np.meshgrid(rel_lons, lats)
    cf = ax.contourf(X, Y, mean_low, levels=21, cmap='RdBu_r', norm=norm, alpha=0.8)
    ax.contour(X, Y, mean_low, levels=[0], colors='k', linewidths=1)
    
    low_w = np.array([calc_boundary(rel_lons, mean_low[i, :], 0.0)[0] for i in range(len(lats))])
    up_w = np.array([calc_boundary(rel_lons, mean_up[i, :], 0.0)[0] for i in range(len(lats))])
    
    ax.plot(low_w, lats, 'b-o', linewidth=3, markersize=8, label='Low Layer West')
    ax.plot(up_w, lats, 'r-s', linewidth=3, markersize=8, label='Upper Layer West')
    
    ax.axvline(0, color='k', linestyle='--', alpha=0.7)
    ax.set_xlabel('Relative Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.set_title(f'MJO ω Composite + West Boundaries (N={n_samples})', fontweight='bold')
    ax.legend(loc='lower left')
    fig.colorbar(cf, ax=ax, shrink=0.8, label='ω')
    
    plt.tight_layout()
    out = OUT_DIR / "02_west_boundary_overlay.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_tilt_2d_heatmap(tilt_west, tilt_east, lats, n_samples):
    """图3: 2D tilt 热力图 (阈值×纬度)"""
    print("\n[Plot 3] 2D tilt heatmap...")
    n_half = len(THRESHOLDS_HALF)
    
    # Combine all ENSO phases
    all_west = []
    all_east = []
    for phase in tilt_west:
        all_west.extend(tilt_west[phase])
        all_east.extend(tilt_east[phase])
    
    mean_west = np.nanmean(all_west, axis=0)  # (n_half, n_lats)
    mean_east = np.nanmean(all_east, axis=0)
    
    # Full axis: west(0→100) + east(100→0)
    mean_2d = np.vstack([mean_west, mean_east[-2::-1, :]])
    n_full = 2 * n_half - 1
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    x = np.arange(n_full)
    x_labels = [f"{int(t*100)}" for t in THRESHOLDS_HALF] + \
               [f"{int(t*100)}" for t in THRESHOLDS_HALF[-2::-1]]
    X, Y = np.meshgrid(x, lats)
    
    vmax = np.nanpercentile(np.abs(mean_2d), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cf = ax.contourf(X, Y, mean_2d.T, levels=21, cmap='RdBu_r', norm=norm)
    ax.contour(X, Y, mean_2d.T, levels=[0], colors='k', linewidths=1.5)
    ax.axvline(n_half - 1, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xticks(x[::2])
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax.set_ylabel('Latitude (°)', fontsize=11)
    ax.set_title(f'2D Tilt Sensitivity Map (N={n_samples})', fontsize=13, fontweight='bold')
    fig.colorbar(cf, ax=ax, shrink=0.9, label='Tilt (°)')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    out = OUT_DIR / "03_tilt_2d_heatmap.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_enso_dominance(tilt_west, tilt_east, lats, n_samples, enso_counts):
    """图4: ENSO tilt 主导区域图 + 显著性"""
    print("\n[Plot 4] ENSO dominance map...")
    n_half = len(THRESHOLDS_HALF)
    n_full = 2 * n_half - 1
    n_lats = len(lats)
    
    # Compute mean tilt by ENSO
    mean_by_enso = {}
    all_by_enso = {}
    for phase in ENSO_ORDER:
        w_arr = np.array(tilt_west[phase])
        e_arr = np.array(tilt_east[phase])
        mean_w = np.nanmean(w_arr, axis=0)
        mean_e = np.nanmean(e_arr, axis=0)
        mean_by_enso[phase] = np.vstack([mean_w, mean_e[-2::-1, :]])
        
        full = np.zeros((w_arr.shape[0], n_full, n_lats))
        full[:, :n_half, :] = w_arr
        full[:, n_half:, :] = e_arr[:, -2::-1, :]
        all_by_enso[phase] = full
    
    # Max phase at each point
    max_phase = np.empty((n_full, n_lats), dtype=object)
    for i in range(n_full):
        for j in range(n_lats):
            vals = {p: mean_by_enso[p][i, j] for p in ENSO_ORDER}
            max_phase[i, j] = max(vals, key=lambda k: vals[k] if np.isfinite(vals[k]) else -np.inf)
    
    # Significance (EN vs LN)
    p_values = np.full((n_full, n_lats), np.nan)
    en = all_by_enso['El Nino']
    ln = all_by_enso['La Nina']
    for i in range(n_full):
        for j in range(n_lats):
            ev = en[:, i, j][np.isfinite(en[:, i, j])]
            lv = ln[:, i, j][np.isfinite(ln[:, i, j])]
            if len(ev) > 5 and len(lv) > 5:
                p_values[i, j] = stats.ttest_ind(ev, lv, equal_var=False).pvalue
    
    significant = p_values < P_VALUE_THRESHOLD
    
    # Plot dominance
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    x = np.arange(n_full)
    x_labels = [f"{int(t*100)}" for t in THRESHOLDS_HALF] + \
               [f"{int(t*100)}" for t in THRESHOLDS_HALF[-2::-1]]
    
    phase_num = {'El Nino': 0, 'La Nina': 1, 'Neutral': 2}
    cmap = ListedColormap([ENSO_COLORS[p] for p in ENSO_ORDER])
    color_map = np.array([[phase_num[max_phase[i, j]] for j in range(n_lats)] for i in range(n_full)])
    
    X, Y = np.meshgrid(x, lats)
    ax.pcolormesh(X, Y, color_map.T, cmap=cmap, vmin=-0.5, vmax=2.5, shading='auto')
    
    # Stippling
    sig_x = [x[i] for i in range(n_full) for j in range(n_lats) if significant[i, j]]
    sig_y = [lats[j] for i in range(n_full) for j in range(n_lats) if significant[i, j]]
    ax.scatter(sig_x, sig_y, c='black', s=5, marker='.', alpha=0.8)
    
    ax.axvline(n_half - 1, color='black', linestyle='--', linewidth=2, alpha=0.6)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax.set_ylabel('Latitude (°)', fontsize=11)
    ax.set_title(f'ENSO Tilt Dominance (N={n_samples}) | dots=p<{P_VALUE_THRESHOLD}',
                 fontsize=13, fontweight='bold')
    ax.legend(handles=[Patch(facecolor=ENSO_COLORS[p], label=f'{p} ({enso_counts[p]})')
                       for p in ENSO_ORDER], loc='upper right')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    out = OUT_DIR / "04_enso_dominance.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()
    
    # Plot 5: EN - LN difference
    print("\n[Plot 5] El Nino - La Nina difference...")
    fig2, ax2 = plt.subplots(figsize=(14, 6), dpi=150)
    diff = mean_by_enso['El Nino'] - mean_by_enso['La Nina']
    vmax = np.nanpercentile(np.abs(diff), 95)
    if vmax == 0:
        vmax = 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cf = ax2.contourf(X, Y, diff.T, levels=21, cmap='RdBu_r', norm=norm)
    ax2.contour(X, Y, diff.T, levels=[0], colors='k', linewidths=1.5)
    ax2.scatter(sig_x, sig_y, c='black', s=5, marker='.', alpha=0.8)
    ax2.axvline(n_half - 1, color='purple', linestyle='--', linewidth=2, alpha=0.6)
    
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), 2)], fontsize=8)
    ax2.set_xlabel('阈值 (%) ← 西侧 | 东侧 →', fontsize=11)
    ax2.set_ylabel('Latitude (°)', fontsize=11)
    ax2.set_title('El Nino - La Nina Tilt Difference | dots=significant', fontweight='bold')
    fig2.colorbar(cf, ax=ax2, shrink=0.9, label='Tilt diff (°)')
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    out2 = OUT_DIR / "05_enso_diff.png"
    plt.savefig(out2, bbox_inches='tight')
    print(f"  Saved: {out2}")
    plt.close()
    
    print(f"\n  El Nino dominant: {np.sum(max_phase == 'El Nino')}")
    print(f"  La Nina dominant: {np.sum(max_phase == 'La Nina')}")
    print(f"  Neutral dominant: {np.sum(max_phase == 'Neutral')}")
    print(f"  Significant (p<{P_VALUE_THRESHOLD}): {np.sum(significant)}/{n_full*n_lats}")


# ======================
# MAIN
# ======================
def main():
    print("=" * 70)
    print("2D MJO Tilt 综合分析")
    print("=" * 70)
    
    # Load all data
    (center_lon, mjo_amp, time_mjo, events, data, time_data,
     levels, lats, lon, low_mask, up_mask, rel_lons) = load_3d_omega()
    
    enso_stats = pd.read_csv(ENSO_STATS_CSV)
    enso_map = dict(zip(enso_stats['event_id'], enso_stats['enso_phase']))
    
    # One-pass computation
    (mean_low, mean_up, tilt_west, tilt_east, 
     n_samples, enso_counts) = compute_daily_samples(
        center_lon, mjo_amp, time_mjo, events, data,
        time_data, lats, lon, low_mask, up_mask, rel_lons,
        enso_map=enso_map
    )
    
    # 5 plots
    plot_omega_composite(mean_low, mean_up, lats, rel_lons, n_samples)
    plot_boundary_overlay(mean_low, mean_up, lats, rel_lons, n_samples)
    plot_tilt_2d_heatmap(tilt_west, tilt_east, lats, n_samples)
    plot_enso_dominance(tilt_west, tilt_east, lats, n_samples, enso_counts)
    
    print(f"\n{'='*70}")
    print(f"Done! 5 figures saved to: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
