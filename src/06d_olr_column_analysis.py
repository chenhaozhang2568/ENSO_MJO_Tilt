# -*- coding: utf-8 -*-
"""
06d_olr_column_analysis.py
OLR重构场、柱积分水汽(q)、柱积分MSE 与 MJO相速度的 1D 分析
输出到 olr/ 和 column_integrated/ 文件夹
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS & CONSTANTS
# ======================
DERIVED_DIR = Path(r"E:\Datas\Derived")
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")

AMP_THRESHOLD = 0.5
SIG_ALPHA = 0.05
GROUP_SIGMA = 0.7
REL_LON_WEST, REL_LON_EAST = -180.0, 180.0

# Physical constants
CP = 1004.0    # J/(kg·K)
LV = 2.501e6   # J/kg
G  = 9.81      # m/s²


# ======================
# HELPERS
# ======================
def _rename_level(da):
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    return da





def _align_1d(data_day, lons, center_lon, rel_lons, dlon):
    """Align a 1D (lon,) field to center_lon → relative longitude."""
    lon_360 = np.mod(lons, 360)
    c360 = np.mod(center_lon, 360)
    sample = np.full(len(rel_lons), np.nan, dtype=np.float32)
    for j, rl in enumerate(rel_lons):
        tlon = np.mod(c360 + rl, 360)
        k = np.argmin(np.abs(lon_360 - tlon))
        if np.abs(lon_360[k] - tlon) < dlon:
            sample[j] = data_day[k]
    return sample


def _column_integrate(data_3d, levels_hPa):
    """Integrate (time, level, lon) → (time, lon) using trapezoidal rule."""
    sort_idx = np.argsort(levels_hPa)
    levels_Pa = levels_hPa[sort_idx] * 100.0
    data_sorted = data_3d[:, sort_idx, :]
    return np.abs(np.trapz(data_sorted, x=levels_Pa, axis=1)) / G


# ======================
# DATA LOADING → (time, lon) timeseries
# ======================
def load_olr_timeseries():
    """Load OLR recon normalized by amplitude. Returns (data, time, lons)."""
    ds = xr.open_dataset(STEP3_NC)
    olr = ds['olr_recon'].values.astype(np.float64)
    amp = ds['amp'].values.astype(np.float64)
    lons = ds['lon'].values
    time_all = pd.to_datetime(ds['time'].values)
    ds.close()
    amp[amp < AMP_THRESHOLD] = np.nan
    olr_norm = (olr / amp[:, None]).astype(np.float32)
    return olr_norm, time_all, lons


def load_column_q_timeseries():
    """Load q recon norm, column integrate."""
    nc = DERIVED_DIR / "era5_mjo_recon_q_norm_1979-2022.nc"
    ds = xr.open_dataset(nc)
    da = _rename_level(ds['q_mjo_recon_norm'])
    data = da.values
    levels = da['level'].values
    lons = da['lon'].values
    time_all = pd.to_datetime(da['time'].values)
    ds.close()
    col = _column_integrate(data, levels.astype(float))
    return col.astype(np.float32), time_all, lons


def load_column_mse_timeseries():
    """Load q and T recon norm, compute Cp*T + Lv*q, column integrate."""
    ds_q = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_q_norm_1979-2022.nc")
    ds_t = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_t_norm_1979-2022.nc")
    da_q = _rename_level(ds_q['q_mjo_recon_norm'])
    da_t = _rename_level(ds_t['t_mjo_recon_norm'])
    mse_3d = CP * da_t.values + LV * da_q.values
    levels = da_q['level'].values
    lons = da_q['lon'].values
    time_all = pd.to_datetime(da_q['time'].values)
    ds_q.close(); ds_t.close()
    col = _column_integrate(mse_3d, levels.astype(float))
    return col.astype(np.float32), time_all, lons


# ======================
# ANALYSIS: background and perturbation
# ======================
def compute_bg_1d(ts_data, ts_time, events, lons):
    """Background: event-mean 1D field in absolute coordinates."""
    n_ev = len(events)
    nX = len(lons)
    event_means = np.full((n_ev, nX), np.nan, dtype=np.float32)
    day_sum = np.zeros(nX, dtype=np.float64)
    day_cnt = np.zeros(nX, dtype=np.float64)
    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (ts_time >= pd.Timestamp(ev["start_date"])) & \
               (ts_time <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        block = ts_data[mask]
        event_means[i] = np.nanmean(block, axis=0)
        finite = np.isfinite(block)
        day_sum += np.where(finite, block, 0).sum(axis=0)
        day_cnt += finite.sum(axis=0)
    day_cnt[day_cnt == 0] = np.nan
    daywise_mean = (day_sum / day_cnt).astype(np.float32)
    return event_means, daywise_mean


def compute_mjo_1d(ts_data, ts_time, events, center_lon_all, time_step3,
                   amp_all, lons):
    """Perturbation: aligned 1D field in relative coordinates."""
    dlon = np.abs(lons[1] - lons[0])
    n_rel = int((REL_LON_EAST - REL_LON_WEST) / dlon) + 1
    rel_lons = np.linspace(REL_LON_WEST, REL_LON_EAST, n_rel)

    n_ev = len(events)
    event_means = np.full((n_ev, n_rel), np.nan, dtype=np.float32)
    day_sum = np.zeros(n_rel, dtype=np.float64)
    day_cnt = np.zeros(n_rel, dtype=np.float64)

    for i, (_, ev) in enumerate(events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        fi_mask = (ts_time >= ts) & (ts_time <= te)
        si_mask = (time_step3 >= ts) & (time_step3 <= te)
        fi_idx = np.where(fi_mask)[0]
        si_idx = np.where(si_mask)[0]
        if len(fi_idx) == 0:
            continue
        samples = []
        for fi in fi_idx:
            t_val = ts_time[fi]
            si_match = [si for si in si_idx
                        if abs((time_step3[si] - t_val).total_seconds()) < 43200]
            if not si_match:
                continue
            si = si_match[0]
            c, a = center_lon_all[si], amp_all[si]
            if not np.isfinite(c) or not np.isfinite(a) or a < AMP_THRESHOLD:
                continue
            s = _align_1d(ts_data[fi], lons, c, rel_lons, dlon)
            samples.append(s)
            finite = np.isfinite(s)
            day_sum += np.where(finite, s, 0)
            day_cnt += finite
        if samples:
            event_means[i] = np.nanmean(np.array(samples), axis=0)

    day_cnt[day_cnt == 0] = np.nan
    daywise_mean = (day_sum / day_cnt).astype(np.float32)
    return event_means, daywise_mean, rel_lons


def correlate_1d(event_means, phase_speed):
    """Pearson correlation at each longitude."""
    nX = event_means.shape[1]
    r = np.full(nX, np.nan)
    p = np.full(nX, np.nan)
    for j in range(nX):
        v = event_means[:, j]
        ok = np.isfinite(v) & np.isfinite(phase_speed)
        if ok.sum() < 10:
            continue
        r[j], p[j] = stats.pearsonr(v[ok], phase_speed[ok])
    return r, p


def group_diff_1d(event_means, fast_mask, slow_mask):
    """Compute Fast-Slow difference and Welch t-test."""
    fast_mean = np.nanmean(event_means[fast_mask], axis=0)
    slow_mean = np.nanmean(event_means[slow_mask], axis=0)
    diff = fast_mean - slow_mean
    nX = event_means.shape[1]
    p_vals = np.full(nX, np.nan)
    for j in range(nX):
        fa = event_means[fast_mask, j]
        sl = event_means[slow_mask, j]
        ok_f, ok_s = np.isfinite(fa), np.isfinite(sl)
        if ok_f.sum() >= 5 and ok_s.sum() >= 5:
            _, p_vals[j] = stats.ttest_ind(fa[ok_f], sl[ok_s], equal_var=False)
    return fast_mean, slow_mean, diff, p_vals


# ======================
# 1D PLOT FUNCTIONS
# ======================
def _style_ax(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10, direction="in")
    ax.grid(alpha=0.3, ls="--")
    for s in ax.spines.values():
        s.set_linewidth(1.2)


def plot_1d_mean(data, x_vals, title, ylabel, out_path, is_relative=False):
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.plot(x_vals, data, 'k-', lw=1.5)
    ymin, ymax = ax.get_ylim()
    if ymin < 0 < ymax:
        ax.axhline(0, color='gray', lw=0.8, ls='--')
    if is_relative:
        ax.axvline(0, color='limegreen', lw=2, ls='--', alpha=0.7)
    xlabel = "Relative Longitude (deg)" if is_relative else "Longitude (deg E)"
    _style_ax(ax, xlabel, ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_1d_corr(r, sig, x_vals, title, out_path, is_relative=False):
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.plot(x_vals, r, 'k-', lw=1.2, alpha=0.5)
    sig_idx = np.where(sig)[0]
    if len(sig_idx) > 0:
        ax.scatter(x_vals[sig_idx], r[sig_idx], c='tab:red', s=18, zorder=5,
                   label=f"Sig ({len(sig_idx)}/{np.sum(np.isfinite(r))})")
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    if is_relative:
        ax.axvline(0, color='limegreen', lw=2, ls='--', alpha=0.7)
    xlabel = "Relative Longitude (deg)" if is_relative else "Longitude (deg E)"
    _style_ax(ax, xlabel, "Pearson r")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_1d_pair(fast_data, slow_data, n_fast, n_slow, x_vals, title, ylabel,
                 out_path, is_relative=False):
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.plot(x_vals, fast_data, 'tab:red', lw=1.5, label=f"Fast (N={n_fast})")
    ax.plot(x_vals, slow_data, 'tab:blue', lw=1.5, label=f"Slow (N={n_slow})")
    ymin, ymax = ax.get_ylim()
    if ymin < 0 < ymax:
        ax.axhline(0, color='gray', lw=0.8, ls='--')
    if is_relative:
        ax.axvline(0, color='limegreen', lw=2, ls='--', alpha=0.7)
    xlabel = "Relative Longitude (deg)" if is_relative else "Longitude (deg E)"
    _style_ax(ax, xlabel, ylabel)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_1d_diff(diff, sig, x_vals, title, ylabel, n_fast, n_slow,
                 out_path, is_relative=False):
    fig, ax = plt.subplots(figsize=(14, 4), dpi=150)
    ax.plot(x_vals, diff, 'k-', lw=1.5)
    ax.fill_between(x_vals, diff, 0, where=diff > 0, alpha=0.15, color='tab:red')
    ax.fill_between(x_vals, diff, 0, where=diff < 0, alpha=0.15, color='tab:blue')
    sig_idx = np.where(sig)[0]
    if len(sig_idx) > 0:
        ax.scatter(x_vals[sig_idx], diff[sig_idx], c='black', s=12, zorder=5, marker='+')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    if is_relative:
        ax.axvline(0, color='limegreen', lw=2, ls='--', alpha=0.7)
    n_sig = len(sig_idx)
    n_total = np.sum(np.isfinite(diff))
    pct = n_sig / n_total * 100 if n_total > 0 else 0
    xlabel = "Relative Longitude (deg)" if is_relative else "Longitude (deg E)"
    _style_ax(ax, xlabel, ylabel)
    ax.set_title(f"{title}  Sig(p<0.05): {n_sig}/{n_total} ({pct:.1f}%)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


# ======================
# ANALYSIS PIPELINE
# ======================
def run_analysis(var_key, var_long, var_unit, ts_data, ts_time, lons,
                 events, phase_speed, fast_mask, slow_mask,
                 center_lon_all, time_step3, amp_all, out_dir):
    """Run full 1D analysis for one variable."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = var_key

    print(f"\n  === {var_long} ===")

    # --- Background ---
    print(f"    Background ...")
    ev_bg, dw_bg = compute_bg_1d(ts_data, ts_time, events, lons)
    plot_1d_mean(dw_bg, lons, f"Background Mean: {var_long} (daywise avg)",
                 f"{var_unit}", out_dir / f"bg_mean_{prefix}.png")

    r_bg, p_bg = correlate_1d(ev_bg, phase_speed)
    sig_bg = (p_bg < SIG_ALPHA) & np.isfinite(p_bg)
    n_sig = int(sig_bg.sum())
    n_tot = int(np.sum(np.isfinite(r_bg)))
    pct = n_sig / n_tot * 100 if n_tot > 0 else 0
    plot_1d_corr(r_bg, sig_bg, lons,
                 f"Background: Corr({var_long}, Phase Speed)  Sig: {n_sig}/{n_tot} ({pct:.1f}%)",
                 out_dir / f"bg_corr_{prefix}.png")

    fast_bg, slow_bg, diff_bg, p_diff_bg = group_diff_1d(ev_bg, fast_mask, slow_mask)
    sig_diff_bg = (p_diff_bg < SIG_ALPHA) & np.isfinite(p_diff_bg)
    plot_1d_pair(fast_bg, slow_bg, int(fast_mask.sum()), int(slow_mask.sum()),
                 lons, f"Background: {var_long} \u2013 Fast vs Slow",
                 f"{var_unit}", out_dir / f"bg_pair_{prefix}.png")
    plot_1d_diff(diff_bg, sig_diff_bg, lons,
                 f"Background Diff (Fast\u2212Slow): {var_long}",
                 f"Delta ({var_unit})",
                 int(fast_mask.sum()), int(slow_mask.sum()),
                 out_dir / f"bg_diff_{prefix}.png")

    # --- Perturbation (MJO aligned) ---
    print(f"    Perturbation ...")
    ev_mjo, dw_mjo, rel_lons = compute_mjo_1d(
        ts_data, ts_time, events, center_lon_all, time_step3, amp_all, lons)
    plot_1d_mean(dw_mjo, rel_lons,
                 f"MJO Mean: {var_long} (OLR-aligned, daywise avg)",
                 f"{var_unit}", out_dir / f"mjo_mean_{prefix}.png",
                 is_relative=True)

    r_mjo, p_mjo = correlate_1d(ev_mjo, phase_speed)
    sig_mjo = (p_mjo < SIG_ALPHA) & np.isfinite(p_mjo)
    n_sig_m = int(sig_mjo.sum())
    n_tot_m = int(np.sum(np.isfinite(r_mjo)))
    pct_m = n_sig_m / n_tot_m * 100 if n_tot_m > 0 else 0
    plot_1d_corr(r_mjo, sig_mjo, rel_lons,
                 f"MJO: Corr({var_long}, Phase Speed)  Sig: {n_sig_m}/{n_tot_m} ({pct_m:.1f}%)",
                 out_dir / f"mjo_corr_{prefix}.png", is_relative=True)

    fast_mjo, slow_mjo, diff_mjo, p_diff_mjo = group_diff_1d(ev_mjo, fast_mask, slow_mask)
    sig_diff_mjo = (p_diff_mjo < SIG_ALPHA) & np.isfinite(p_diff_mjo)
    plot_1d_pair(fast_mjo, slow_mjo, int(fast_mask.sum()), int(slow_mask.sum()),
                 rel_lons, f"MJO: {var_long} \u2013 Fast vs Slow",
                 f"{var_unit}", out_dir / f"mjo_pair_{prefix}.png",
                 is_relative=True)
    plot_1d_diff(diff_mjo, sig_diff_mjo, rel_lons,
                 f"MJO Diff (Fast\u2212Slow): {var_long}",
                 f"Delta ({var_unit})",
                 int(fast_mask.sum()), int(slow_mask.sum()),
                 out_dir / f"mjo_diff_{prefix}.png", is_relative=True)


# ======================
# MAIN
# ======================
def main():
    print("=" * 70)
    print("06d: OLR & Column-Integrated Analysis vs Phase Speed")
    print("=" * 70)

    # Load step3 for alignment
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    amp_all = ds3["amp"].values.astype(float)
    time_step3 = pd.to_datetime(ds3["time"].values)
    ds3.close()

    # Load events & phase speed
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)

    ps_valid = phase_speed[np.isfinite(phase_speed)]
    mu, sigma = np.mean(ps_valid), np.std(ps_valid)
    fast_mask = phase_speed > mu + GROUP_SIGMA * sigma
    slow_mask = phase_speed < mu - GROUP_SIGMA * sigma
    print(f"  Events: {len(merged)}, Fast={fast_mask.sum()}, Slow={slow_mask.sum()}")

    # --- 1. OLR ---
    print("\n" + "=" * 50)
    print("Loading OLR ...")
    olr_data, olr_time, olr_lons = load_olr_timeseries()
    run_analysis("olr", "OLR", "W/m\u00b2 / amp", olr_data, olr_time, olr_lons,
                 merged, phase_speed, fast_mask, slow_mask,
                 center_lon_all, time_step3, amp_all,
                 FIG_DIR / "olr")

    # --- 2. Column q ---
    print("\n" + "=" * 50)
    print("Loading Column Moisture ...")
    colq_data, colq_time, colq_lons = load_column_q_timeseries()
    run_analysis("column_q", "Column Moisture (q)", "kg/m\u00b2 / amp",
                 colq_data, colq_time, colq_lons,
                 merged, phase_speed, fast_mask, slow_mask,
                 center_lon_all, time_step3, amp_all,
                 FIG_DIR / "column_integrated")

    # --- 3. Column MSE ---
    print("\n" + "=" * 50)
    print("Loading Column MSE ...")
    mse_data, mse_time, mse_lons = load_column_mse_timeseries()
    run_analysis("column_mse", "Column MSE (CpT+Lvq)", "J/m\u00b2 / amp",
                 mse_data, mse_time, mse_lons,
                 merged, phase_speed, fast_mask, slow_mask,
                 center_lon_all, time_step3, amp_all,
                 FIG_DIR / "column_integrated")

    print(f"\nAll done!")


if __name__ == "__main__":
    main()
