# -*- coding: utf-8 -*-
"""
stg_wtg_analysis.py — STG/WTG 分组垂直环流与 omega 合成分析

功能：
    按事件平均 Tilt 的 ±0.7σ 阈值将 MJO 事件分为 STG（强倾斜）和 WTG（弱倾斜）两组，
    对比两组的垂直环流结构（高度坐标矢量图）和标准化 omega 合成（气压坐标），
    并检验两组相速度差异。
输入：
    era5_mjo_recon_{u,w}_norm_1979-2022.nc, mjo_mvEOF_step3_1979-2022.nc,
    tilt_daily_step4_layermean_1979-2022.nc, mjo_events_step3_1979-2022.csv,
    tilt_event_stats_with_enso_1979-2022.csv
输出：
    figures/circulation/stg_wtg_vertical_circulation_v2.png,
    figures/stg_wtg/stg_wtg_omega_composite.png, tilt_vs_phase_speed_scatter.png
用法：
    python tests/stg_wtg_analysis.py             # 全部
    python tests/stg_wtg_analysis.py circulation  # 环流图
    python tests/stg_wtg_analysis.py composite    # omega 合成 + 相速度
"""

from __future__ import annotations

import sys
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
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
U_RECON_NC     = r"E:\Datas\Derived\era5_mjo_recon_u_norm_1979-2022.nc"
W_RECON_NC     = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
STEP3_NC       = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
TILT_NC        = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV     = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures")
FIG_DIR_CIRC = FIG_DIR / "circulation"
FIG_DIR_STG  = FIG_DIR / "stg_wtg"

WINTER_MONTHS = {11, 12, 1, 2, 3, 4}

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}

TILT_THRESHOLD_STD = 0.7
AMP_THRESHOLD = 0.5


# ======================
# SHARED HELPERS
# ======================
def _rename_level(da):
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    return da


def _interpolate_to_height(data, levels, target_heights):
    """将气压层数据插值到等高度坐标"""
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


# ============================================================
# 1. CIRCULATION (from stg_wtg_circulation.py)
# ============================================================
def run_circulation():
    """STG/WTG 垂直环流合成图 (高度坐标 + 矢量图)"""
    FIG_DIR_CIRC.mkdir(parents=True, exist_ok=True)
    print("\n[Circulation] Loading data...")

    ds_u = xr.open_dataset(U_RECON_NC)
    ds_w = xr.open_dataset(W_RECON_NC)
    ds3 = xr.open_dataset(STEP3_NC)
    ds_tilt = xr.open_dataset(TILT_NC)
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])

    u = ds_u["u_mjo_recon_norm"].values
    w = ds_w["w_mjo_recon_norm"].values
    time = pd.to_datetime(ds_u["time"].values)
    levels = ds_u["pressure_level"].values
    lon = ds_u["lon"].values
    center_lon = ds3["center_lon_track"].values
    amp = ds3["amp"].values
    tilt = ds_tilt["tilt"].values

    print(f"  Shape: u={u.shape}, levels={levels}")

    # Classify STG/WTG
    event_tilts = []
    for _, ev in events.iterrows():
        start, end = pd.Timestamp(ev['start_date']), pd.Timestamp(ev['end_date'])
        mask = (time >= start) & (time <= end)
        tv = tilt[mask]
        valid = np.isfinite(tv)
        if valid.sum() > 0:
            event_tilts.append({'start': start, 'end': end,
                                'mean_tilt': np.nanmean(tv[valid])})
    df_ev = pd.DataFrame(event_tilts)
    mt, st = df_ev['mean_tilt'].mean(), df_ev['mean_tilt'].std()
    stg_events = df_ev[df_ev['mean_tilt'] > mt + TILT_THRESHOLD_STD * st]
    wtg_events = df_ev[df_ev['mean_tilt'] < mt - TILT_THRESHOLD_STD * st]
    print(f"  STG: {len(stg_events)}, WTG: {len(wtg_events)}")

    def _composite(event_list, lon_range=(-90, 180)):
        dlon = lon[1] - lon[0]
        n_rel = int((lon_range[1] - lon_range[0]) / dlon) + 1
        rel_lons = np.linspace(lon_range[0], lon_range[1], n_rel)
        u_samples, w_samples = [], []
        for _, ev in event_list.iterrows():
            mask = (time >= ev['start']) & (time <= ev['end'])
            for idx in np.where(mask)[0]:
                if time[idx].month not in WINTER_MONTHS:
                    continue
                c, a = center_lon[idx], amp[idx]
                if not np.isfinite(c) or not np.isfinite(a) or a < AMP_THRESHOLD:
                    continue
                rel = (lon - c + 180) % 360 - 180
                u_d = np.full((len(levels), n_rel), np.nan)
                w_d = np.full((len(levels), n_rel), np.nan)
                for j, rl in enumerate(rel_lons):
                    k = np.argmin(np.abs(rel - rl))
                    if np.abs(rel[k] - rl) < dlon:
                        u_d[:, j] = u[idx, :, k]
                        w_d[:, j] = w[idx, :, k]
                u_samples.append(u_d)
                w_samples.append(w_d)
        u_s, w_s = np.array(u_samples), np.array(w_samples)
        u_m, w_m = np.nanmean(u_s, axis=0), np.nanmean(w_s, axis=0)
        n = u_s.shape[0]
        w_std = np.nanstd(w_s, axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_w = w_m / (w_std / np.sqrt(n))
        p_w = 2 * (1 - stats.t.cdf(np.abs(t_w), df=n - 1))
        return {'u': u_m, 'w': w_m, 'sig_mask': p_w < 0.05,
                'rel_lons': rel_lons, 'levels': levels, 'n_samples': n}

    stg_comp = _composite(stg_events)
    wtg_comp = _composite(wtg_events)
    print(f"  STG samples: {stg_comp['n_samples']}, WTG: {wtg_comp['n_samples']}")

    def _plot_circ(ax, comp, title, show_ylabel=True):
        rel_lons = comp['rel_lons']
        target_h = np.linspace(0.5, 12, 24)
        u_h = _interpolate_to_height(comp['u'], levels, target_h)
        w_h = _interpolate_to_height(comp['w'], levels, target_h)
        sig_h = _interpolate_to_height(comp['sig_mask'].astype(float), levels, target_h) > 0.5
        u_sm = gaussian_filter(np.nan_to_num(u_h, nan=0), sigma=1.0)
        w_sm = gaussian_filter(np.nan_to_num(w_h, nan=0), sigma=1.0)
        nm = np.isnan(u_h) | np.isnan(w_h)
        u_sm[nm] = np.nan
        w_sm[nm] = np.nan
        X, Y = np.meshgrid(rel_lons, target_h)
        w_norm = w_sm / 0.01
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=0.5)
        cf = ax.contourf(X, Y, w_norm, levels=np.arange(-1.0, 0.6, 0.2),
                         cmap='RdBu_r', norm=norm, extend='both')
        for i in range(len(target_h)):
            for j in range(0, len(rel_lons), 4):
                if sig_h[i, j]:
                    ax.plot(rel_lons[j], target_h[i], 'k.', markersize=2.5, alpha=0.8)
        ax.quiver(X[::2, ::6], Y[::2, ::6], u_sm[::2, ::6], -w_sm[::2, ::6] * 800,
                  color='black', scale=40, width=0.004, headwidth=4, pivot='middle')
        ax.axvline(0, color='limegreen', lw=3.5, alpha=0.95)
        ax.set_ylim(0.5, 12)
        ax.set_xlim(-90, 180)
        ax.set_xticks(np.arange(-90, 181, 30))
        if show_ylabel:
            ax.set_ylabel('Height (km)', fontsize=11)
        ax.set_xlabel('Relative Longitude', fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
        ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in pticks])
        ax2.set_yticklabels([str(p) for p in pticks])
        ax2.set_ylabel('Pressure (hPa)', fontsize=11)
        return cf

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    cf1 = _plot_circ(axes[0], stg_comp, '(a) Strong Tilt')
    cf2 = _plot_circ(axes[1], wtg_comp, '(b) Weak Tilt', show_ylabel=False)
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    cbar = fig.colorbar(cf2, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Standardized Anomalous Vertical Velocity', fontsize=10)
    plt.subplots_adjust(bottom=0.12, wspace=0.25)
    out = FIG_DIR_CIRC / "stg_wtg_vertical_circulation_v2.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


# ============================================================
# 2. COMPOSITE (from stg_wtg_composite.py)
# ============================================================
def run_composite():
    """STG/WTG omega 合成 + 相速度分析"""
    FIG_DIR_STG.mkdir(parents=True, exist_ok=True)
    print("\n[Composite] Loading data...")

    ds_w = xr.open_dataset(W_RECON_NC)
    w_recon = _rename_level(ds_w['w_mjo_recon_norm'])
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon = ds3['center_lon_track'].values
    time_mjo = pd.to_datetime(ds3.time.values)
    mjo_amp = ds3['amp'].values
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])
    enso_stats = pd.read_csv(ENSO_STATS_CSV)

    # Phase speed
    phase_speeds = []
    for _, ev in events.iterrows():
        eid = ev['event_id']
        mask = (time_mjo >= ev['start_date']) & (time_mjo <= ev['end_date'])
        lons = center_lon[mask]
        days = np.arange(len(lons))
        valid = np.isfinite(lons)
        if valid.sum() < 5:
            phase_speeds.append({'event_id': eid, 'phase_speed_ms': np.nan})
            continue
        slope, *_ = stats.linregress(days[valid], lons[valid])
        phase_speeds.append({'event_id': eid, 'phase_speed_ms': slope * 111e3 / 86400})
    enso_stats = enso_stats.merge(pd.DataFrame(phase_speeds), on='event_id')

    # STG / WTG classification
    tilt_mean = enso_stats['mean_tilt'].mean()
    tilt_std = enso_stats['mean_tilt'].std()
    enso_stats['group'] = 'Normal'
    enso_stats.loc[enso_stats['mean_tilt'] > tilt_mean + TILT_THRESHOLD_STD * tilt_std, 'group'] = 'STG'
    enso_stats.loc[enso_stats['mean_tilt'] < tilt_mean - TILT_THRESHOLD_STD * tilt_std, 'group'] = 'WTG'
    n_stg = (enso_stats['group'] == 'STG').sum()
    n_wtg = (enso_stats['group'] == 'WTG').sum()
    print(f"  STG: {n_stg}, WTG: {n_wtg}, Normal: {len(enso_stats) - n_stg - n_wtg}")

    w_time = pd.to_datetime(w_recon.time.values)
    levels = w_recon.level.values
    lon = w_recon.lon.values
    lon_360 = np.mod(lon, 360)
    dlon = np.abs(lon[1] - lon[0])

    def _comp(group_name, half_width=60):
        group_eids = enso_stats[enso_stats['group'] == group_name]['event_id'].values
        n_rel = int(2 * half_width / dlon) + 1
        rel_lons = np.linspace(-half_width, half_width, n_rel)
        samples = []
        for eid in group_eids:
            ev = events[events['event_id'] == eid].iloc[0]
            mask = (time_mjo >= ev['start_date']) & (time_mjo <= ev['end_date'])
            for idx in np.where(mask)[0]:
                c = center_lon[idx]
                a = mjo_amp[idx]
                if not np.isfinite(c) or not np.isfinite(a) or a < AMP_THRESHOLD:
                    continue
                t = time_mjo[idx]
                w_idx = np.where(w_time == t)[0]
                if len(w_idx) == 0:
                    continue
                w_day = w_recon.isel(time=w_idx[0]).values / a
                c360 = np.mod(c, 360)
                sample = np.zeros((len(levels), n_rel))
                for j, rl in enumerate(rel_lons):
                    tlon = np.mod(c360 + rl, 360)
                    sample[:, j] = w_day[:, np.argmin(np.abs(lon_360 - tlon))]
                samples.append(sample)
        samples = np.array(samples)
        mean_c = np.nanmean(samples, axis=0)
        std_c = np.nanstd(samples, axis=0)
        std_c[std_c == 0] = np.nan
        n = np.sum(~np.isnan(samples), axis=0)
        from scipy.stats import t as t_dist
        t_stat = mean_c / (std_c / np.sqrt(n))
        sig = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=np.maximum(n - 1, 1))) < 0.05
        w_std_all = np.nanstd(samples)
        return mean_c / w_std_all, sig, rel_lons, levels, len(group_eids), len(samples)

    stg_std, stg_sig, rel_lons, levels, n_stg_ev, n_stg_s = _comp('STG')
    wtg_std, wtg_sig, _, _, n_wtg_ev, n_wtg_s = _comp('WTG')
    print(f"  STG: {n_stg_s} samples / {n_stg_ev} events")
    print(f"  WTG: {n_wtg_s} samples / {n_wtg_ev} events")

    # Plot composites
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150, sharey=True)
    vmax = max(np.nanpercentile(np.abs(stg_std), 95), np.nanpercentile(np.abs(wtg_std), 95))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    xx, yy = np.meshgrid(rel_lons, levels)

    for ax, data, sig, title in [
        (axes[0], stg_std, stg_sig, f"(a) STG (N={n_stg_ev})"),
        (axes[1], wtg_std, wtg_sig, f"(b) WTG (N={n_wtg_ev})"),
    ]:
        cf = ax.contourf(rel_lons, levels, data, levels=20, cmap='RdBu_r', norm=norm, extend='both')
        ax.contour(rel_lons, levels, data, levels=10, colors='k', linewidths=0.5, alpha=0.5)
        ax.scatter(xx[sig], yy[sig], s=1, c='k', alpha=0.3, marker='.')
        ax.axvline(0, color='green', lw=2)
        ax.set_xlabel("Relative Longitude (°)")
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(-60, 60)
        ax.grid(ls='--', alpha=0.3)
    axes[0].set_ylabel("Pressure (hPa)")
    cbar = fig.colorbar(cf, ax=axes, orientation='vertical', shrink=0.8) 
    cbar.set_label('Standardized ω Anomaly')
    out = FIG_DIR_STG / "stg_wtg_omega_composite.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    # Phase speed comparison
    stg_spd = enso_stats[enso_stats['group'] == 'STG']['phase_speed_ms'].dropna()
    wtg_spd = enso_stats[enso_stats['group'] == 'WTG']['phase_speed_ms'].dropna()
    t_val, p_val = stats.ttest_ind(stg_spd, wtg_spd, equal_var=False)
    print(f"\n  Phase speed: STG={stg_spd.mean():.2f} m/s, WTG={wtg_spd.mean():.2f} m/s")
    print(f"  t={t_val:+.3f}, p={p_val:.4f}")

    # Scatter
    valid = enso_stats.dropna(subset=['mean_tilt', 'phase_speed_ms'])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = {'STG': '#E74C3C', 'WTG': '#3498DB', 'Normal': '#95A5A6'}
    for g in ['STG', 'Normal', 'WTG']:
        sub = valid[valid['group'] == g]
        ax.scatter(sub['phase_speed_ms'], sub['mean_tilt'], c=colors[g],
                   label=f"{g} (N={len(sub)})", s=60, alpha=0.7, edgecolors='k', linewidths=0.5)
    x, y = valid['phase_speed_ms'].values, valid['mean_tilt'].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    ax.plot(np.linspace(x.min(), x.max(), 100),
            slope * np.linspace(x.min(), x.max(), 100) + intercept, 'r-', lw=2, label=f'r={r:.2f}')
    ax.set_xlabel("Phase Speed (m/s)")
    ax.set_ylabel("Tilt (°)")
    ax.set_title(f"Tilt vs Phase Speed (N={len(valid)}, r={r:.3f}, p={p:.4f})")
    ax.legend()
    ax.grid(ls='--', alpha=0.3)
    out = FIG_DIR_STG / "tilt_vs_phase_speed_scatter.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    enso_stats.to_csv(FIG_DIR_STG / "event_stg_wtg_classification.csv", index=False)


# ============================================================
# MAIN
# ============================================================
ANALYSES = {
    "circulation": run_circulation,
    "composite": run_composite,
}


def main():
    print("=" * 70)
    print("STG/WTG Analysis")
    print("=" * 70)
    if len(sys.argv) > 1:
        name = sys.argv[1].lower()
        if name in ANALYSES:
            ANALYSES[name]()
        else:
            print(f"Unknown: {name}. Available: {list(ANALYSES.keys())}")
            sys.exit(1)
    else:
        for func in ANALYSES.values():
            func()
    print("\nDone!")


if __name__ == "__main__":
    main()
