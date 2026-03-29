# -*- coding: utf-8 -*-
"""
06i_surface_flux_analysis.py
表面通量(LHF, SHF, 净辐射) + SST 与 MJO 相速度的 1D 分析
输出到 surface_flux/ 文件夹
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
SST_DIR = Path(r"E:\Datas\ERA5\raw\single_level\sst_daily")
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")
OUT_DIR = FIG_DIR / "surface_flux"

AMP_THRESHOLD = 0.5
SIG_ALPHA = 0.05
GROUP_SIGMA = 0.7

# Only LHF and SST have significant background signals → do perturbation too
PERTURB_VARS = ["lhf", "sst"]

# Variables to analyze
SURFACE_VARS = {
    "lhf":  {"sl_var": "slhf", "label": "Latent Heat Flux (LHF)",
             "unit": "W/m²", "sign": -1},      # ERA5 LHF < 0 (upward), flip sign
    "shf":  {"sl_var": "sshf", "label": "Sensible Heat Flux (SHF)",
             "unit": "W/m²", "sign": -1},
    "qrad": {"sl_var": ["ssr", "str"], "label": "Net Radiation (SW+LW)",
             "unit": "W/m²", "sign": 1},         # SSR>0, STR<0, sum = net
    "sst":  {"sl_var": "sst", "label": "Sea Surface Temperature (SST)",
             "unit": "K", "sign": 1, "source": "sst"},
}


# ======================
# HELPERS
# ======================



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


# ======================
# DATA LOADING
# ======================
def load_surface_timeseries(var_key, var_info):
    """
    Load daily surface variable, lat-average → (n_days, n_lon).
    Returns (data, times, lons).
    """
    src = var_info.get("source", "sl")
    data_dir = SST_DIR if src == "sst" else SL_DIR
    pattern = "era5_sst_dailymean_*.nc" if src == "sst" else "era5_sl_dailymean_*.nc"

    files = sorted(data_dir.glob(pattern))
    all_data, all_time = [], []

    for f in files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'

        # Extract and combine variables
        sl_var = var_info["sl_var"]
        if isinstance(sl_var, list):
            # Sum multiple variables (e.g. ssr + str for net radiation)
            vals = sum(ds[v].values for v in sl_var if v in ds.data_vars)
        else:
            if sl_var not in ds.data_vars:
                ds.close()
                continue
            vals = ds[sl_var].values

        vals = vals * var_info["sign"]

        # Lat-average: (time, lat, lon) → (time, lon)
        vals_latavg = np.nanmean(vals, axis=1)

        times = pd.to_datetime(ds[tdim].values)
        all_data.append(vals_latavg)
        all_time.append(times)

        lons = ds['longitude'].values
        ds.close()

    data = np.concatenate(all_data, axis=0).astype(np.float32)
    times = np.concatenate(all_time)
    return data, pd.DatetimeIndex(times), lons


# ======================
# ANALYSIS
# ======================
def compute_bg_1d(ts_data, ts_time, events, lons):
    """Background: event-mean 1D field in absolute coordinates."""
    n_ev = len(events)
    nX = len(lons)
    event_means = np.full((n_ev, nX), np.nan, dtype=np.float32)

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (ts_time >= pd.Timestamp(ev["start_date"])) & \
               (ts_time <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        event_means[i] = np.nanmean(ts_data[mask], axis=0)

    # Grand mean (daywise)
    grand_mean = np.nanmean(event_means, axis=0)
    return event_means, grand_mean


def compute_mjo_1d(ts_data, ts_time, events, center_lon_all, time_step3,
                   amp_all, lons):
    """Perturbation: aligned 1D field in relative coordinates."""
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


def plot_1d_pair(fast_data, slow_data, n_fast, n_slow, x_vals, title, ylabel, out_path, is_relative=False):
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


def plot_1d_diff(diff, sig, x_vals, title, ylabel, out_path, is_relative=False):
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
# RUN ONE VARIABLE
# ======================
def run_var(var_key, var_info, events, phase_speed, fast_mask, slow_mask,
            center_lon_all=None, time_step3=None, amp_all=None):
    """Run 1D analysis: background for all vars, + perturbation for LHF/SST."""
    label = var_info["label"]
    unit = var_info["unit"]
    print(f"\n  === {label} ===")

    print(f"    Loading ...")
    ts_data, ts_time, lons = load_surface_timeseries(var_key, var_info)
    print(f"    Data shape: {ts_data.shape}, time: {ts_time[0]} ~ {ts_time[-1]}")

    ev_bg, grand_mean = compute_bg_1d(ts_data, ts_time, events, lons)

    # --- Background ---
    plot_1d_mean(grand_mean, lons,
                 f"Background Mean: {label}",
                 unit, OUT_DIR / f"bg_mean_{var_key}.png")

    r_bg, p_bg = correlate_1d(ev_bg, phase_speed)
    sig_bg = (p_bg < SIG_ALPHA) & np.isfinite(p_bg)
    n_sig = int(sig_bg.sum())
    n_tot = int(np.sum(np.isfinite(r_bg)))
    pct = n_sig / n_tot * 100 if n_tot > 0 else 0
    plot_1d_corr(r_bg, sig_bg, lons,
                 f"Background: Corr({label}, Phase Speed)  Sig(p<0.05): {n_sig}/{n_tot} ({pct:.1f}%)",
                 OUT_DIR / f"bg_corr_{var_key}.png")

    fast_bg, slow_bg, diff_bg, p_diff = group_diff_1d(ev_bg, fast_mask, slow_mask)
    sig_diff = (p_diff < SIG_ALPHA) & np.isfinite(p_diff)
    plot_1d_pair(fast_bg, slow_bg, int(fast_mask.sum()), int(slow_mask.sum()),
                 lons, f"Background: {label} \u2013 Fast vs Slow",
                 unit, OUT_DIR / f"bg_pair_{var_key}.png")
    plot_1d_diff(diff_bg, sig_diff, lons,
                 f"Background Diff (Fast\u2212Slow): {label}",
                 f"Delta ({unit})",
                 OUT_DIR / f"bg_diff_{var_key}.png")
    print(f"    BG done: corr sig={pct:.1f}%, diff sig={sig_diff.sum()}/{n_tot}")

    # --- Perturbation (MJO-aligned) --- only for significant vars
    if var_key in PERTURB_VARS and center_lon_all is not None:
        print(f"    Perturbation (MJO-aligned) ...")
        ev_mjo, grand_mjo, rel_lons = compute_mjo_1d(
            ts_data, ts_time, events, center_lon_all, time_step3, amp_all, lons)

        plot_1d_mean(grand_mjo, rel_lons,
                     f"MJO Mean: {label} (OLR-aligned)",
                     unit, OUT_DIR / f"mjo_mean_{var_key}.png", is_relative=True)

        r_mjo, p_mjo = correlate_1d(ev_mjo, phase_speed)
        sig_mjo = (p_mjo < SIG_ALPHA) & np.isfinite(p_mjo)
        n_sig_m = int(sig_mjo.sum())
        n_tot_m = int(np.sum(np.isfinite(r_mjo)))
        pct_m = n_sig_m / n_tot_m * 100 if n_tot_m > 0 else 0
        plot_1d_corr(r_mjo, sig_mjo, rel_lons,
                     f"MJO: Corr({label}, Phase Speed)  Sig(p<0.05): {n_sig_m}/{n_tot_m} ({pct_m:.1f}%)",
                     OUT_DIR / f"mjo_corr_{var_key}.png", is_relative=True)

        fast_mjo, slow_mjo, diff_mjo, p_diff_m = group_diff_1d(ev_mjo, fast_mask, slow_mask)
        sig_diff_m = (p_diff_m < SIG_ALPHA) & np.isfinite(p_diff_m)
        plot_1d_pair(fast_mjo, slow_mjo, int(fast_mask.sum()), int(slow_mask.sum()),
                     rel_lons, f"MJO: {label} \u2013 Fast vs Slow",
                     unit, OUT_DIR / f"mjo_pair_{var_key}.png", is_relative=True)
        plot_1d_diff(diff_mjo, sig_diff_m, rel_lons,
                     f"MJO Diff (Fast\u2212Slow): {label}",
                     f"Delta ({unit})",
                     OUT_DIR / f"mjo_diff_{var_key}.png", is_relative=True)
        print(f"    MJO done: corr sig={pct_m:.1f}%")


# ======================
# MAIN
# ======================
def main():
    print("=" * 70)
    print("06i: Surface Flux + SST Analysis vs Phase Speed")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load step3 for MJO alignment
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

    for var_key, var_info in SURFACE_VARS.items():
        run_var(var_key, var_info, merged, phase_speed, fast_mask, slow_mask,
                center_lon_all, time_step3, amp_all)

    print(f"\nAll done! Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
