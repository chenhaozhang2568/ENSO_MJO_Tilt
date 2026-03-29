# -*- coding: utf-8 -*-
"""
06j_precipitation_analysis.py
降水(TP)与MJO相速度的1D分析 — 背景坐标 + 扰动坐标
输出到 precipitation/ 文件夹
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
SL_DIR = Path(r"E:\Datas\ERA5\raw\single_level\daily_mean")
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")
OUT_DIR = FIG_DIR / "precipitation"

AMP_THRESHOLD = 0.5
SIG_ALPHA = 0.05
GROUP_SIGMA = 0.7


# ======================
# HELPERS
# ======================



def _align_1d(data_day, lons, center_lon, rel_lons, dlon):
    lon_360 = np.mod(lons, 360)
    c360 = np.mod(center_lon, 360)
    sample = np.full(len(rel_lons), np.nan, dtype=np.float32)
    for j, rl in enumerate(rel_lons):
        tlon = np.mod(c360 + rl, 360)
        k = np.argmin(np.abs(lon_360 - tlon))
        if np.abs(lon_360[k] - tlon) < dlon:
            sample[j] = data_day[k]
    return sample


# ======================
# DATA LOADING
# ======================
def load_precip_timeseries():
    """Load daily precipitation, lat-average → (n_days, n_lon)."""
    files = sorted(SL_DIR.glob("era5_sl_dailymean_*.nc"))
    all_data, all_time = [], []
    lons = None
    for f in files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        vals = ds['tp'].values  # mm/day
        vals_latavg = np.nanmean(vals, axis=1)
        all_data.append(vals_latavg)
        all_time.append(pd.to_datetime(ds[tdim].values))
        if lons is None:
            lons = ds['longitude'].values
        ds.close()
    data = np.concatenate(all_data, axis=0).astype(np.float32)
    times = pd.DatetimeIndex(np.concatenate(all_time))
    return data, times, lons


# ======================
# ANALYSIS
# ======================
def compute_bg_1d(ts_data, ts_time, events, lons):
    n_ev = len(events)
    nX = len(lons)
    event_means = np.full((n_ev, nX), np.nan, dtype=np.float32)
    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (ts_time >= pd.Timestamp(ev["start_date"])) & \
               (ts_time <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        event_means[i] = np.nanmean(ts_data[mask], axis=0)
    grand_mean = np.nanmean(event_means, axis=0)
    return event_means, grand_mean


def compute_mjo_1d(ts_data, ts_time, events, center_lon_all, time_step3,
                   amp_all, lons):
    dlon = np.abs(lons[1] - lons[0])
    n_rel = int(360 / dlon) + 1
    rel_lons = np.linspace(-180, 180, n_rel)

    n_ev = len(events)
    event_means = np.full((n_ev, n_rel), np.nan, dtype=np.float32)

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
        if samples:
            event_means[i] = np.nanmean(np.array(samples), axis=0)

    grand_mean = np.nanmean(event_means, axis=0)
    return event_means, grand_mean, rel_lons


def correlate_1d(event_means, phase_speed):
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
# PLOTTING
# ======================
def _style_ax(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10, direction="in")
    ax.grid(alpha=0.3, ls="--")
    for s in ax.spines.values():
        s.set_linewidth(1.2)


def plot_mean(data, x_vals, title, ylabel, out_path, is_relative=False):
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


def plot_corr(r, sig, x_vals, title, out_path, is_relative=False):
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


def plot_pair(fast_data, slow_data, n_fast, n_slow, x_vals, title, ylabel,
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


def plot_diff(diff, sig, x_vals, title, ylabel, out_path, is_relative=False):
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
# MAIN
# ======================
def main():
    print("=" * 70)
    print("06j: Precipitation Analysis vs Phase Speed")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # Load precipitation
    print("\nLoading precipitation ...")
    tp_data, tp_time, lons = load_precip_timeseries()
    print(f"  Shape: {tp_data.shape}, time: {tp_time[0]} ~ {tp_time[-1]}")

    # ---- Background ----
    print("\n  --- Background ---")
    ev_bg, grand_bg = compute_bg_1d(tp_data, tp_time, merged, lons)

    plot_mean(grand_bg, lons, "Background Mean: Precipitation",
              "mm/day", OUT_DIR / "bg_mean_precip.png")

    r_bg, p_bg = correlate_1d(ev_bg, phase_speed)
    sig_bg = (p_bg < SIG_ALPHA) & np.isfinite(p_bg)
    n_sig = int(sig_bg.sum()); n_tot = int(np.sum(np.isfinite(r_bg)))
    pct = n_sig / n_tot * 100 if n_tot > 0 else 0
    plot_corr(r_bg, sig_bg, lons,
              f"Background: Corr(Precipitation, Phase Speed)  Sig(p<0.05): {n_sig}/{n_tot} ({pct:.1f}%)",
              OUT_DIR / "bg_corr_precip.png")

    fast_bg, slow_bg, diff_bg, p_diff_bg = group_diff_1d(ev_bg, fast_mask, slow_mask)
    sig_diff_bg = (p_diff_bg < SIG_ALPHA) & np.isfinite(p_diff_bg)
    plot_pair(fast_bg, slow_bg, int(fast_mask.sum()), int(slow_mask.sum()),
              lons, "Background: Precipitation \u2013 Fast vs Slow",
              "mm/day", OUT_DIR / "bg_pair_precip.png")
    plot_diff(diff_bg, sig_diff_bg, lons,
              "Background Diff (Fast\u2212Slow): Precipitation",
              "Delta (mm/day)", OUT_DIR / "bg_diff_precip.png")

    # ---- Perturbation (MJO-aligned) ----
    print("\n  --- Perturbation ---")
    ev_mjo, grand_mjo, rel_lons = compute_mjo_1d(
        tp_data, tp_time, merged, center_lon_all, time_step3, amp_all, lons)

    plot_mean(grand_mjo, rel_lons,
              "MJO Mean: Precipitation (OLR-aligned)",
              "mm/day", OUT_DIR / "mjo_mean_precip.png", is_relative=True)

    r_mjo, p_mjo = correlate_1d(ev_mjo, phase_speed)
    sig_mjo = (p_mjo < SIG_ALPHA) & np.isfinite(p_mjo)
    n_sig_m = int(sig_mjo.sum()); n_tot_m = int(np.sum(np.isfinite(r_mjo)))
    pct_m = n_sig_m / n_tot_m * 100 if n_tot_m > 0 else 0
    plot_corr(r_mjo, sig_mjo, rel_lons,
              f"MJO: Corr(Precipitation, Phase Speed)  Sig(p<0.05): {n_sig_m}/{n_tot_m} ({pct_m:.1f}%)",
              OUT_DIR / "mjo_corr_precip.png", is_relative=True)

    fast_mjo, slow_mjo, diff_mjo, p_diff_mjo = group_diff_1d(ev_mjo, fast_mask, slow_mask)
    sig_diff_mjo = (p_diff_mjo < SIG_ALPHA) & np.isfinite(p_diff_mjo)
    plot_pair(fast_mjo, slow_mjo, int(fast_mask.sum()), int(slow_mask.sum()),
              rel_lons, "MJO: Precipitation \u2013 Fast vs Slow",
              "mm/day", OUT_DIR / "mjo_pair_precip.png", is_relative=True)
    plot_diff(diff_mjo, sig_diff_mjo, rel_lons,
              "MJO Diff (Fast\u2212Slow): Precipitation",
              "Delta (mm/day)", OUT_DIR / "mjo_diff_precip.png", is_relative=True)

    print(f"\nAll done! Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
