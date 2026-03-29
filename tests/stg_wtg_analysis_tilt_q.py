# -*- coding: utf-8 -*-
"""
stg_wtg_analysis_tilt_q.py — STG/WTG 分组垂直环流与 omega 合成分析（tilt_q 版）

功能：
    按事件平均 tilt_q (q_max - up_west) 的 ±0.7σ 阈值将 MJO 事件分为
    STG（强倾斜）和 WTG（弱倾斜）两组，对比两组的垂直环流结构（高度坐标矢量图）
    和标准化 omega 合成（气压坐标），并检验两组相速度差异。

与 stg_wtg_analysis.py 的区别：
    - 使用 tilt_q_daily_1979-2022.nc 中的 tilt_q（q最大值 - omega西边界）
    - 使用 tilt_q_phase_speed_by_enso.csv（已含 phase_speed_m_s）
    - 输出到 figures/stg_wtg_tilt_q/ 文件夹

输入：
    era5_mjo_recon_{u,w}_norm_1979-2022.nc, mjo_mvEOF_step3_1979-2022.nc,
    tilt_q_daily_1979-2022.nc, mjo_events_step3_1979-2022.csv,
    tilt_q_phase_speed_by_enso.csv
输出：
    figures/stg_wtg_tilt_q/stg_wtg_omega_composite.png
    figures/stg_wtg_tilt_q/stg_wtg_vertical_circulation.png
    figures/stg_wtg_tilt_q/tilt_q_vs_phase_speed_scatter.png
    figures/stg_wtg_tilt_q/event_stg_wtg_classification.csv
用法：
    python tests/stg_wtg_analysis_tilt_q.py             # 全部
    python tests/stg_wtg_analysis_tilt_q.py circulation  # 环流图
    python tests/stg_wtg_analysis_tilt_q.py composite    # omega 合成 + 相速度
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
TILT_Q_NC      = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"
EVENTS_CSV     = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
TILT_Q_CSV     = r"E:\Datas\Derived\tilt_q_phase_speed_by_enso.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\stg_wtg_tilt_q")

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


def _classify_stg_wtg(df, tilt_col='mean_tilt_q'):
    """按 ±0.7σ 阈值分组"""
    mt = df[tilt_col].mean()
    st = df[tilt_col].std()
    df = df.copy()
    df['group'] = 'Normal'
    df.loc[df[tilt_col] > mt + TILT_THRESHOLD_STD * st, 'group'] = 'STG'
    df.loc[df[tilt_col] < mt - TILT_THRESHOLD_STD * st, 'group'] = 'WTG'
    n_stg = (df['group'] == 'STG').sum()
    n_wtg = (df['group'] == 'WTG').sum()
    print(f"  Tilt_q: mean={mt:.1f}, std={st:.1f}")
    print(f"  STG threshold: > {mt + TILT_THRESHOLD_STD * st:.1f}")
    print(f"  WTG threshold: < {mt - TILT_THRESHOLD_STD * st:.1f}")
    print(f"  STG: {n_stg}, WTG: {n_wtg}, Normal: {len(df) - n_stg - n_wtg}")
    return df


# ============================================================
# 1. CIRCULATION — 垂直环流合成图
# ============================================================
def run_circulation():
    """STG/WTG 垂直环流合成图 (高度坐标 + 矢量图)"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[Circulation] Loading data...")

    ds_u = xr.open_dataset(U_RECON_NC)
    ds_w = xr.open_dataset(W_RECON_NC)
    ds3 = xr.open_dataset(STEP3_NC)
    ds_tilt_q = xr.open_dataset(TILT_Q_NC)
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])

    u = ds_u["u_mjo_recon_norm"].values
    w = ds_w["w_mjo_recon_norm"].values
    time = pd.to_datetime(ds_u["time"].values)
    levels = ds_u["pressure_level"].values
    lon = ds_u["lon"].values
    center_lon = ds3["center_lon_track"].values
    amp = ds3["amp"].values
    tilt_q = ds_tilt_q["tilt_q"].values
    tilt_q_time = pd.to_datetime(ds_tilt_q["time"].values)

    print(f"  Shape: u={u.shape}, levels={levels}")

    # 计算逐事件 tilt_q 平均
    event_tilts = []
    for _, ev in events.iterrows():
        start = pd.Timestamp(ev['start_date'])
        end = pd.Timestamp(ev['end_date'])
        mask = (tilt_q_time >= start) & (tilt_q_time <= end)
        tv = tilt_q[mask]
        valid = np.isfinite(tv)
        if valid.sum() > 0:
            event_tilts.append({'event_id': ev['event_id'],
                                'start': start, 'end': end,
                                'mean_tilt_q': float(np.nanmean(tv[valid]))})
    df_ev = pd.DataFrame(event_tilts)
    df_ev = _classify_stg_wtg(df_ev)

    stg_events = df_ev[df_ev['group'] == 'STG']
    wtg_events = df_ev[df_ev['group'] == 'WTG']

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
        w_norm = w_sm / 0.02
        norm = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
        cf = ax.contourf(X, Y, w_norm, levels=np.arange(-1.0, 1.01, 0.2),
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
    cf1 = _plot_circ(axes[0], stg_comp,
                     f'(a) Strong Tilt_q (N={stg_comp["n_samples"]})')
    cf2 = _plot_circ(axes[1], wtg_comp,
                     f'(b) Weak Tilt_q (N={wtg_comp["n_samples"]})',
                     show_ylabel=False)
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.03])
    cbar = fig.colorbar(cf2, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Standardized Anomalous Vertical Velocity', fontsize=11)
    plt.subplots_adjust(bottom=0.13, wspace=0.25)
    out = FIG_DIR / "stg_wtg_vertical_circulation.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


# ============================================================
# 2. COMPOSITE — omega 合成 + 相速度分析
# ============================================================
def run_composite():
    """STG/WTG omega 合成 + 相速度分析（含 u/w 风矢量）"""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[Composite] Loading data...")

    ds_u = xr.open_dataset(U_RECON_NC)
    ds_w = xr.open_dataset(W_RECON_NC)
    u_recon = _rename_level(ds_u['u_mjo_recon_norm'])
    w_recon = _rename_level(ds_w['w_mjo_recon_norm'])
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon = ds3['center_lon_track'].values
    time_mjo = pd.to_datetime(ds3.time.values)
    mjo_amp = ds3['amp'].values
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])

    # 直接读取 tilt_q CSV（已含 phase_speed_m_s 和 mean_tilt_q）
    tilt_q_stats = pd.read_csv(TILT_Q_CSV)
    print(f"  Events: {len(tilt_q_stats)}")

    # STG / WTG 分组
    tilt_q_stats = _classify_stg_wtg(tilt_q_stats)

    u_time = pd.to_datetime(u_recon.time.values)
    w_time = pd.to_datetime(w_recon.time.values)
    levels = w_recon.level.values
    lon = w_recon.lon.values
    lon_360 = np.mod(lon, 360)
    dlon = np.abs(lon[1] - lon[0])

    LON_WEST, LON_EAST = -90, 180

    def _comp(group_name):
        group_eids = tilt_q_stats[tilt_q_stats['group'] == group_name]['event_id'].values
        n_rel = int((LON_EAST - LON_WEST) / dlon) + 1
        rel_lons = np.linspace(LON_WEST, LON_EAST, n_rel)
        w_samples, u_samples = [], []
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
                u_idx = np.where(u_time == t)[0]
                if len(w_idx) == 0 or len(u_idx) == 0:
                    continue
                w_day = w_recon.isel(time=w_idx[0]).values / a
                u_day = u_recon.isel(time=u_idx[0]).values / a
                c360 = np.mod(c, 360)
                w_sample = np.zeros((len(levels), n_rel))
                u_sample = np.zeros((len(levels), n_rel))
                for j, rl in enumerate(rel_lons):
                    tlon = np.mod(c360 + rl, 360)
                    k = np.argmin(np.abs(lon_360 - tlon))
                    w_sample[:, j] = w_day[:, k]
                    u_sample[:, j] = u_day[:, k]
                w_samples.append(w_sample)
                u_samples.append(u_sample)
        w_samples = np.array(w_samples)
        u_samples = np.array(u_samples)
        w_mean = np.nanmean(w_samples, axis=0)
        u_mean = np.nanmean(u_samples, axis=0)
        w_std_c = np.nanstd(w_samples, axis=0, ddof=1)
        w_std_c[w_std_c == 0] = np.nan
        n = np.sum(~np.isnan(w_samples), axis=0)
        from scipy.stats import t as t_dist
        t_stat = w_mean / (w_std_c / np.sqrt(n))
        sig = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=np.maximum(n - 1, 1))) < 0.05
        w_std_all = np.nanstd(w_samples)
        return (w_mean / w_std_all, u_mean, w_mean, sig,
                rel_lons, levels, len(group_eids), len(w_samples))

    stg_norm, stg_u, stg_w, stg_sig, rel_lons, levels, n_stg_ev, n_stg_s = _comp('STG')
    wtg_norm, wtg_u, wtg_w, wtg_sig, _, _, n_wtg_ev, n_wtg_s = _comp('WTG')
    print(f"  STG: {n_stg_s} samples / {n_stg_ev} events")
    print(f"  WTG: {n_wtg_s} samples / {n_wtg_ev} events")

    # ---- omega 合成图 ----
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=150)
    clevs = np.arange(-1.0, 1.01, 0.2)
    norm_clr = TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
    height_ticks = [LEVEL_TO_HEIGHT[p] for p in pticks]
    target_h = np.linspace(0.5, 12, 24)

    datasets = [
        (axes[0], stg_norm, stg_u, stg_w, stg_sig,
         f'(a) Strong Tilt_q (N={n_stg_ev} events, {n_stg_s} days)', True),
        (axes[1], wtg_norm, wtg_u, wtg_w, wtg_sig,
         f'(b) Weak Tilt_q (N={n_wtg_ev} events, {n_wtg_s} days)', False),
    ]

    for ax, w_data, u_data, w_raw, sig, title, show_ylabel in datasets:
        w_h = _interpolate_to_height(w_data, levels, target_h)
        u_h = _interpolate_to_height(u_data, levels, target_h)
        w_raw_h = _interpolate_to_height(w_raw, levels, target_h)
        sig_h = _interpolate_to_height(sig.astype(float), levels, target_h) > 0.5

        w_sm = gaussian_filter(np.nan_to_num(w_h, nan=0), sigma=1.0)
        u_sm = gaussian_filter(np.nan_to_num(u_h, nan=0), sigma=1.0)
        w_raw_sm = gaussian_filter(np.nan_to_num(w_raw_h, nan=0), sigma=1.0)
        nm = np.isnan(w_h) | np.isnan(u_h)
        w_sm[nm] = np.nan
        u_sm[nm] = np.nan
        w_raw_sm[nm] = np.nan

        X, Y = np.meshgrid(rel_lons, target_h)

        cf = ax.contourf(X, Y, w_sm, levels=clevs, cmap='RdBu_r',
                          norm=norm_clr, extend='both')

        for i in range(len(target_h)):
            for j in range(0, len(rel_lons), 4):
                if sig_h[i, j]:
                    ax.plot(rel_lons[j], target_h[i], 'k.', markersize=1.5, alpha=0.8)

        w_arrow = -w_raw_sm * 50
        skip_x, skip_y = 6, 2
        ax.quiver(X[::skip_y, ::skip_x], Y[::skip_y, ::skip_x],
                  u_sm[::skip_y, ::skip_x], w_arrow[::skip_y, ::skip_x],
                  color='black', scale=30, width=0.003,
                  headwidth=4, headlength=4, pivot='middle')

        qk_x, qk_y = 0.88, 0.95
        q = ax.quiver([rel_lons[-1] * qk_x], [target_h[-1] * qk_y],
                       [5], [0], color='black', scale=30, width=0.003,
                       headwidth=4, headlength=4, pivot='middle')
        ax.quiverkey(q, qk_x, qk_y, 5, '5', labelpos='W',
                     coordinates='axes', fontproperties={'size': 8})
        ax.text(qk_x, qk_y + 0.05, 'Reference Vector', transform=ax.transAxes,
                fontsize=7, ha='center', va='bottom')

        ax.axvline(0, color='limegreen', lw=3.0, alpha=0.95)
        ax.set_ylim(0.1, 12)
        ax.set_xlim(LON_WEST, LON_EAST)
        ax.set_xticks(np.arange(LON_WEST, LON_EAST + 1, 30))
        ax.set_xticklabels([f'{int(x)}' for x in np.arange(LON_WEST, LON_EAST + 1, 30)],
                            fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')

        if show_ylabel:
            ax.set_ylabel('Height (km)', fontsize=11)

        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(height_ticks)
        ax2.set_yticklabels([str(p) for p in pticks])
        if not show_ylabel:
            ax2.set_ylabel('Pressure (hPa)', fontsize=11)

    cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.03])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal',
                         ticks=np.arange(-1.0, 1.01, 0.2))
    cbar.set_label('Normalized Anomalous Vertical Velocity', fontsize=11)
    plt.subplots_adjust(bottom=0.14, wspace=0.30)
    out = FIG_DIR / "stg_wtg_omega_composite.png"
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    # ---- 相速度对比 ----
    stg_spd = tilt_q_stats[tilt_q_stats['group'] == 'STG']['phase_speed_m_s'].dropna()
    wtg_spd = tilt_q_stats[tilt_q_stats['group'] == 'WTG']['phase_speed_m_s'].dropna()
    t_val, p_val = stats.ttest_ind(stg_spd, wtg_spd, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(stg_spd, wtg_spd, alternative='two-sided')
    print(f"\n  Phase speed:")
    print(f"    STG: mean={stg_spd.mean():.2f}, std={stg_spd.std():.2f}, N={len(stg_spd)}")
    print(f"    WTG: mean={wtg_spd.mean():.2f}, std={wtg_spd.std():.2f}, N={len(wtg_spd)}")
    print(f"    Welch t-test: t={t_val:+.3f}, p={p_val:.4f}")
    print(f"    Mann-Whitney: U={u_stat:.0f}, p={u_pval:.4f}")

    # ---- tilt_q vs phase_speed 散点图 ----
    valid = tilt_q_stats.dropna(subset=['mean_tilt_q', 'phase_speed_m_s'])
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = {'STG': '#E74C3C', 'WTG': '#3498DB', 'Normal': '#95A5A6'}
    for g in ['STG', 'Normal', 'WTG']:
        sub = valid[valid['group'] == g]
        ax.scatter(sub['phase_speed_m_s'], sub['mean_tilt_q'], c=colors[g],
                   label=f"{g} (N={len(sub)})", s=60, alpha=0.7,
                   edgecolors='k', linewidths=0.5)

    x, y = valid['phase_speed_m_s'].values, valid['mean_tilt_q'].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, 'r-', lw=2, label=f'r={r:.2f}')

    ax.set_xlabel("Phase Speed (m/s)", fontsize=12)
    ax.set_ylabel("Tilt_q (deg)", fontsize=12)
    sig_str = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.set_title(f"Tilt_q vs Phase Speed (N={len(valid)}, r={r:.3f}, p={p:.4f} {sig_str})",
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(ls='--', alpha=0.3)

    # 添加相速度对比文字
    text_str = (f"STG: speed={stg_spd.mean():.2f} m/s (N={len(stg_spd)})\n"
                f"WTG: speed={wtg_spd.mean():.2f} m/s (N={len(wtg_spd)})\n"
                f"Welch t-test: p={p_val:.4f}")
    ax.text(0.02, 0.02, text_str, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    out = FIG_DIR / "tilt_q_vs_phase_speed_scatter.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    # 保存分类 CSV
    tilt_q_stats.to_csv(FIG_DIR / "event_stg_wtg_classification.csv", index=False)
    print(f"  Saved: {FIG_DIR / 'event_stg_wtg_classification.csv'}")


# ============================================================
# MAIN
# ============================================================
ANALYSES = {
    "circulation": run_circulation,
    "composite": run_composite,
}


def main():
    print("=" * 70)
    print("STG/WTG Analysis (tilt_q version)")
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
