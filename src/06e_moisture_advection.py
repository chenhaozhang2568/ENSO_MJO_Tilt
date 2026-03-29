# -*- coding: utf-8 -*-
"""
06e_moisture_advection.py
水汽平流项 -u·∂q/∂x 与MJO相速度的关系分析 (背景场+扰动场+分组对比)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
from scipy.interpolate import interp1d
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
OUT_DIR = FIG_DIR / "moisture_advection"

REL_LON_WEST, REL_LON_EAST = -180.0, 180.0
AMP_THRESHOLD = 0.5
SIG_ALPHA = 0.05
GROUP_SIGMA = 0.7
R_EARTH = 6.371e6  # m

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}
TARGET_HEIGHTS = np.linspace(0.5, 12, 24)
P_TICKS = [1000, 850, 700, 500, 400, 300, 200]


# ======================
# HELPERS
# ======================
def _rename_level(da):
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    return da


def interp_to_height(data_2d, levels):
    heights = np.array([LEVEL_TO_HEIGHT.get(int(p), 5.0) for p in levels])
    n_x = data_2d.shape[1]
    out = np.full((len(TARGET_HEIGHTS), n_x), np.nan)
    for j in range(n_x):
        valid = np.isfinite(data_2d[:, j])
        if valid.sum() >= 2:
            f = interp1d(heights[valid], data_2d[valid, j], kind='linear',
                         bounds_error=False, fill_value=np.nan)
            out[:, j] = f(TARGET_HEIGHTS)
    return out





def _align_day(data_day, lons, center_lon, rel_lons, dlon):
    lon_360 = np.mod(lons, 360)
    c360 = np.mod(center_lon, 360)
    nL = data_day.shape[0]
    sample = np.full((nL, len(rel_lons)), np.nan, dtype=np.float32)
    for j, rl in enumerate(rel_lons):
        tlon = np.mod(c360 + rl, 360)
        k = np.argmin(np.abs(lon_360 - tlon))
        if np.abs(lon_360[k] - tlon) < dlon:
            sample[:, j] = data_day[:, k]
    return sample


def correlate_with_phase_speed(field_all, phase_speed):
    _, nL, nX = field_all.shape
    r = np.full((nL, nX), np.nan, dtype=np.float32)
    p = np.full((nL, nX), np.nan, dtype=np.float32)
    for k in range(nL):
        for j in range(nX):
            v = field_all[:, k, j]
            ok = np.isfinite(v) & np.isfinite(phase_speed)
            if ok.sum() < 10:
                continue
            r[k, j], p[k, j] = stats.pearsonr(v[ok], phase_speed[ok])
    return r, p


# ======================
# DATA: LOAD u & q, COMPUTE ADVECTION
# ======================
def _load_uq():
    """Load u and q normalized recon fields."""
    ds_u = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_u_norm_1979-2022.nc")
    ds_q = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_q_norm_1979-2022.nc")
    da_u = _rename_level(ds_u["u_mjo_recon_norm"])
    da_q = _rename_level(ds_q["q_mjo_recon_norm"])
    time_all = pd.to_datetime(da_u["time"].values)
    levels = da_u["level"].values
    lons = da_u["lon"].values
    data_u = da_u.values
    data_q = da_q.values
    ds_u.close(); ds_q.close()
    return data_u, data_q, time_all, levels, lons


def _compute_advection(u_block, q_block, dx_m):
    """Compute -u * dq/dx for a (n_time, nL, nX) block."""
    dq_dx = np.gradient(q_block, dx_m, axis=-1)
    return -u_block * dq_dx


def compute_bg_advection(events):
    data_u, data_q, time_all, levels, lons = _load_uq()
    dlon = np.abs(lons[1] - lons[0])
    dx_m = dlon * np.pi / 180 * R_EARTH

    n_ev = len(events)
    nL, nX = len(levels), len(lons)
    event_means = np.full((n_ev, nL, nX), np.nan, dtype=np.float32)
    day_sum = np.zeros((nL, nX), dtype=np.float64)
    day_cnt = np.zeros((nL, nX), dtype=np.float64)

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_all >= pd.Timestamp(ev["start_date"])) & \
               (time_all <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        adv = _compute_advection(data_u[mask], data_q[mask], dx_m)
        event_means[i] = np.nanmean(adv, axis=0)
        finite = np.isfinite(adv)
        day_sum += np.where(finite, adv, 0).sum(axis=0)
        day_cnt += finite.sum(axis=0)

    day_cnt[day_cnt == 0] = np.nan
    daywise_mean = (day_sum / day_cnt).astype(np.float32)
    return event_means, daywise_mean, levels, lons


def compute_mjo_advection(events, center_lon_all, time_step3, amp_all):
    data_u, data_q, time_field, levels, lons = _load_uq()
    dlon = np.abs(lons[1] - lons[0])
    n_rel = int((REL_LON_EAST - REL_LON_WEST) / dlon) + 1
    rel_lons = np.linspace(REL_LON_WEST, REL_LON_EAST, n_rel)
    dx_rel = np.abs(rel_lons[1] - rel_lons[0]) * np.pi / 180 * R_EARTH

    n_ev = len(events)
    nL = len(levels)
    event_means = np.full((n_ev, nL, n_rel), np.nan, dtype=np.float32)
    day_sum = np.zeros((nL, n_rel), dtype=np.float64)
    day_cnt = np.zeros((nL, n_rel), dtype=np.float64)

    for i, (_, ev) in enumerate(events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        fi_mask = (time_field >= ts) & (time_field <= te)
        si_mask = (time_step3 >= ts) & (time_step3 <= te)
        fi_idx = np.where(fi_mask)[0]
        si_idx = np.where(si_mask)[0]
        if len(fi_idx) == 0:
            continue
        samples = []
        for fi in fi_idx:
            t_val = time_field[fi]
            si_match = [si for si in si_idx
                        if abs((time_step3[si] - t_val).total_seconds()) < 43200]
            if not si_match:
                continue
            si = si_match[0]
            c, a = center_lon_all[si], amp_all[si]
            if not np.isfinite(c) or not np.isfinite(a) or a < AMP_THRESHOLD:
                continue
            u_aligned = _align_day(data_u[fi], lons, c, rel_lons, dlon)
            q_aligned = _align_day(data_q[fi], lons, c, rel_lons, dlon)
            dq_dx = np.gradient(q_aligned, dx_rel, axis=-1)
            adv = -u_aligned * dq_dx
            samples.append(adv)
            finite = np.isfinite(adv)
            day_sum += np.where(finite, adv, 0)
            day_cnt += finite
        if samples:
            event_means[i] = np.nanmean(np.array(samples), axis=0)

    day_cnt[day_cnt == 0] = np.nan
    daywise_mean = (day_sum / day_cnt).astype(np.float32)
    return event_means, daywise_mean, levels, rel_lons


# ======================
# PLOT FUNCTIONS
# ======================
def _setup_height_axis(ax, xlabel="Longitude (deg E)"):
    ax.set_ylim(0.1, 12.5)
    ax.set_ylabel("Height (km)", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=10,
                   direction="in", top=True, right=False, length=5)
    for s in ax.spines.values():
        s.set_linewidth(1.2)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in P_TICKS])
    ax2.set_yticklabels([str(p) for p in P_TICKS], fontsize=9)
    ax2.set_ylabel("hPa", fontsize=11)
    ax2.tick_params(direction="in", length=4)
    return ax2


def _plot_2d(field_h, x_vals, title, cbar_label, out_path,
             sig_h=None, is_relative=False, vmax_override=None):
    fig, ax = plt.subplots(figsize=(14 if not is_relative else 16, 5.5), dpi=150)
    vmax = vmax_override or np.nanmax(np.abs(field_h))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    clevs = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(x_vals, TARGET_HEIGHTS, field_h,
                     levels=clevs, cmap="RdBu_r", norm=norm, extend="both")
    if sig_h is not None:
        skip = 3 if is_relative else 2
        for i in range(len(TARGET_HEIGHTS)):
            for j in range(0, len(x_vals), skip):
                if sig_h[i, j]:
                    ax.plot(x_vals[j], TARGET_HEIGHTS[i], 'k+',
                            markersize=5, markeredgewidth=1.0, alpha=0.9)
    if is_relative:
        ax.axvline(0, color="limegreen", lw=2.5, ls="--", alpha=0.9)
        ax.set_xlim(REL_LON_WEST, REL_LON_EAST)
        ax.set_xticks(np.arange(REL_LON_WEST, REL_LON_EAST + 1, 30))
    xlabel = "Relative Longitude (deg)" if is_relative else "Longitude (deg E)"
    _setup_height_axis(ax, xlabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    cbar_ax = fig.add_axes([0.12, 0.04, 0.78, 0.025])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=11)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def _plot_pair(fast_h, slow_h, n_fast, n_slow, x_vals, title, cbar_label,
               out_path, is_relative=False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), dpi=150, sharey=True)
    vmax = max(np.nanmax(np.abs(fast_h)), np.nanmax(np.abs(slow_h)))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    clevs = np.linspace(-vmax, vmax, 21)
    for ax, data, label in zip(axes, [fast_h, slow_h],
                               [f"(a) Fast (N={n_fast})", f"(b) Slow (N={n_slow})"]):
        cf = ax.contourf(x_vals, TARGET_HEIGHTS, data,
                         levels=clevs, cmap="RdBu_r", norm=norm, extend="both")
        if is_relative:
            ax.axvline(0, color="limegreen", lw=2.5, ls="--", alpha=0.9)
            ax.set_xlim(REL_LON_WEST, REL_LON_EAST)
            ax.set_xticks(np.arange(REL_LON_WEST, REL_LON_EAST + 1, 60))
        xlabel = "Relative Longitude (deg)" if is_relative else "Longitude (deg E)"
        _setup_height_axis(ax, xlabel)
        ax.set_title(label, fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    cbar_ax = fig.add_axes([0.12, 0.04, 0.78, 0.025])
    fig.colorbar(cf, cax=cbar_ax, orientation="horizontal").set_label(cbar_label, fontsize=11)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


# ======================
# MAIN
# ======================
def main():
    print("=" * 70)
    print("06e: Moisture Advection (-u dq/dx) vs Phase Speed")
    print("=" * 70)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds3 = xr.open_dataset(STEP3_NC)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    amp_all = ds3["amp"].values.astype(float)
    time_step3 = pd.to_datetime(ds3["time"].values)
    ds3.close()

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)
    print(f"  Events: {len(merged)}")

    # --- Grouping ---
    ps_valid = phase_speed[np.isfinite(phase_speed)]
    mu, sigma = np.mean(ps_valid), np.std(ps_valid)
    fast_mask = phase_speed > mu + GROUP_SIGMA * sigma
    slow_mask = phase_speed < mu - GROUP_SIGMA * sigma
    print(f"  Fast N={fast_mask.sum()}, Slow N={slow_mask.sum()}")

    ADV_LONG = "Moisture Advection (\u2212u\u00b7\u2202q/\u2202x)"
    ADV_UNIT = "(kg/kg)·(m/s)/m / amp"

    # ==============================
    # A. BACKGROUND FIELD
    # ==============================
    print("\n--- Background Advection ---")
    ev_means, dw_mean, levels, lons = compute_bg_advection(merged)
    mean_h = interp_to_height(dw_mean, levels)
    _plot_2d(mean_h, lons, f"Background Mean: {ADV_LONG}",
             ADV_UNIT, OUT_DIR / "bg_mean_advection.png")

    r_map, p_map = correlate_with_phase_speed(ev_means, phase_speed)
    sig_mask = (p_map < SIG_ALPHA) & np.isfinite(p_map)
    r_h = interp_to_height(r_map, levels)
    sig_h = interp_to_height(sig_mask.astype(float), levels) > 0.5
    n_sig = int(np.sum(sig_h & np.isfinite(r_h)))
    n_total = int(np.sum(np.isfinite(r_h)))
    pct = n_sig / n_total * 100 if n_total > 0 else 0
    vmax_r = min(max(0.3, np.nanmax(np.abs(r_h)) * 1.1), 1.0)
    _plot_2d(r_h, lons,
             f"Background: Corr({ADV_LONG}, Phase Speed)  Sig(p<0.05): {n_sig}/{n_total} ({pct:.1f}%)",
             "Pearson r", OUT_DIR / "bg_corr_advection.png",
             sig_h=sig_h, vmax_override=vmax_r)

    # Group comparison
    fast_means_h = interp_to_height(np.nanmean(ev_means[fast_mask], axis=0), levels)
    slow_means_h = interp_to_height(np.nanmean(ev_means[slow_mask], axis=0), levels)
    _plot_pair(fast_means_h, slow_means_h, int(fast_mask.sum()), int(slow_mask.sum()),
               lons, f"Background: {ADV_LONG} \u2013 Fast vs Slow",
               ADV_UNIT, OUT_DIR / "bg_pair_advection.png")

    diff = np.nanmean(ev_means[fast_mask], axis=0) - np.nanmean(ev_means[slow_mask], axis=0)
    nL_, nX_ = diff.shape
    p_diff = np.full(diff.shape, np.nan)
    for k in range(nL_):
        for j in range(nX_):
            fa = ev_means[fast_mask, k, j]
            sl = ev_means[slow_mask, k, j]
            ok_f, ok_s = np.isfinite(fa), np.isfinite(sl)
            if ok_f.sum() >= 5 and ok_s.sum() >= 5:
                _, p_diff[k, j] = stats.ttest_ind(fa[ok_f], sl[ok_s], equal_var=False)
    sig_diff = (p_diff < SIG_ALPHA) & np.isfinite(p_diff)
    diff_h = interp_to_height(diff, levels)
    sig_diff_h = interp_to_height(sig_diff.astype(float), levels) > 0.5
    n_sig_d = int(np.sum(sig_diff_h & np.isfinite(diff_h)))
    n_total_d = int(np.sum(np.isfinite(diff_h)))
    pct_d = n_sig_d / n_total_d * 100 if n_total_d > 0 else 0
    _plot_2d(diff_h, lons,
             f"Background Diff (Fast\u2212Slow): {ADV_LONG}  Sig(p<0.05): {n_sig_d}/{n_total_d} ({pct_d:.1f}%)",
             f"Delta {ADV_UNIT}", OUT_DIR / "bg_diff_advection.png",
             sig_h=sig_diff_h)

    # ==============================
    # B. PERTURBATION (MJO) FIELD
    # ==============================
    print("\n--- MJO Perturbation Advection ---")
    ev_means_m, dw_mean_m, levels_m, rel_lons = compute_mjo_advection(
        merged, center_lon_all, time_step3, amp_all)
    mean_h_m = interp_to_height(dw_mean_m, levels_m)
    _plot_2d(mean_h_m, rel_lons, f"MJO Mean: {ADV_LONG} (OLR-aligned)",
             ADV_UNIT, OUT_DIR / "mjo_mean_advection.png", is_relative=True)

    r_m, p_m = correlate_with_phase_speed(ev_means_m, phase_speed)
    sig_m = (p_m < SIG_ALPHA) & np.isfinite(p_m)
    r_h_m = interp_to_height(r_m, levels_m)
    sig_h_m = interp_to_height(sig_m.astype(float), levels_m) > 0.5
    n_sig_m = int(np.sum(sig_h_m & np.isfinite(r_h_m)))
    n_total_m = int(np.sum(np.isfinite(r_h_m)))
    pct_m = n_sig_m / n_total_m * 100 if n_total_m > 0 else 0
    vmax_rm = min(max(0.3, np.nanmax(np.abs(r_h_m)) * 1.1), 1.0)
    _plot_2d(r_h_m, rel_lons,
             f"MJO Perturbation: Corr({ADV_LONG}, Phase Speed)  Sig(p<0.05): {n_sig_m}/{n_total_m} ({pct_m:.1f}%)",
             "Pearson r", OUT_DIR / "mjo_corr_advection.png",
             sig_h=sig_h_m, is_relative=True, vmax_override=vmax_rm)

    # Group comparison
    fast_m_h = interp_to_height(np.nanmean(ev_means_m[fast_mask], axis=0), levels_m)
    slow_m_h = interp_to_height(np.nanmean(ev_means_m[slow_mask], axis=0), levels_m)
    _plot_pair(fast_m_h, slow_m_h, int(fast_mask.sum()), int(slow_mask.sum()),
               rel_lons, f"MJO Perturbation: {ADV_LONG} \u2013 Fast vs Slow",
               ADV_UNIT, OUT_DIR / "mjo_pair_advection.png", is_relative=True)

    diff_m = np.nanmean(ev_means_m[fast_mask], axis=0) - np.nanmean(ev_means_m[slow_mask], axis=0)
    nL_m, nX_m = diff_m.shape
    p_diff_m = np.full(diff_m.shape, np.nan)
    for k in range(nL_m):
        for j in range(nX_m):
            fa = ev_means_m[fast_mask, k, j]
            sl = ev_means_m[slow_mask, k, j]
            ok_f, ok_s = np.isfinite(fa), np.isfinite(sl)
            if ok_f.sum() >= 5 and ok_s.sum() >= 5:
                _, p_diff_m[k, j] = stats.ttest_ind(fa[ok_f], sl[ok_s], equal_var=False)
    sig_diff_m = (p_diff_m < SIG_ALPHA) & np.isfinite(p_diff_m)
    diff_h_m = interp_to_height(diff_m, levels_m)
    sig_diff_h_m = interp_to_height(sig_diff_m.astype(float), levels_m) > 0.5
    n_sig_dm = int(np.sum(sig_diff_h_m & np.isfinite(diff_h_m)))
    n_total_dm = int(np.sum(np.isfinite(diff_h_m)))
    pct_dm = n_sig_dm / n_total_dm * 100 if n_total_dm > 0 else 0
    _plot_2d(diff_h_m, rel_lons,
             f"MJO Diff (Fast\u2212Slow): {ADV_LONG}  Sig(p<0.05): {n_sig_dm}/{n_total_dm} ({pct_dm:.1f}%)",
             f"Delta {ADV_UNIT}", OUT_DIR / "mjo_diff_advection.png",
             sig_h=sig_diff_h_m, is_relative=True)

    print(f"\nAll plots saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
