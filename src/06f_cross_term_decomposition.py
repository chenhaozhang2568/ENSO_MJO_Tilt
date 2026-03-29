# -*- coding: utf-8 -*-
"""
06f_cross_term_decomposition.py
水汽平流交叉项分解: 将 -u·∂q/∂x 分解为
  Term A: -<ū>·∂q'ᵢ/∂x  (平均风 × 事件异常水汽梯度)
  Term B: -u'ᵢ·∂<q̄>/∂x  (事件异常风 × 平均水汽梯度)
在背景场和扰动场两个坐标系下分别分析
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

DERIVED_DIR = Path(r"E:\Datas\Derived")
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")
OUT_DIR = FIG_DIR / "cross_terms"

REL_LON_WEST, REL_LON_EAST = -180.0, 180.0
AMP_THRESHOLD = 0.5
SIG_ALPHA = 0.05
GROUP_SIGMA = 0.7
R_EARTH = 6.371e6

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}
TARGET_HEIGHTS = np.linspace(0.5, 12, 24)
P_TICKS = [1000, 850, 700, 500, 400, 300, 200]


# ====== HELPERS ======
def _rename_level(da):
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    return da

def interp_to_height(data_2d, levels):
    heights = np.array([LEVEL_TO_HEIGHT.get(int(p), 5.0) for p in levels])
    out = np.full((len(TARGET_HEIGHTS), data_2d.shape[1]), np.nan)
    for j in range(data_2d.shape[1]):
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

def correlate_2d(field_all, phase_speed):
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

def group_diff_2d(field_all, fast_mask, slow_mask):
    diff = np.nanmean(field_all[fast_mask], axis=0) - np.nanmean(field_all[slow_mask], axis=0)
    nL, nX = diff.shape
    p_vals = np.full(diff.shape, np.nan)
    for k in range(nL):
        for j in range(nX):
            fa = field_all[fast_mask, k, j]
            sl = field_all[slow_mask, k, j]
            ok_f, ok_s = np.isfinite(fa), np.isfinite(sl)
            if ok_f.sum() >= 5 and ok_s.sum() >= 5:
                _, p_vals[k, j] = stats.ttest_ind(fa[ok_f], sl[ok_s], equal_var=False)
    return diff, p_vals


# ====== COMPUTE FUNCTIONS ======
def _load_uq():
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


def compute_bg_event_means(events):
    """Background: per-event mean u and q in absolute coords."""
    data_u, data_q, time_all, levels, lons = _load_uq()
    dx_m = np.abs(lons[1] - lons[0]) * np.pi / 180 * R_EARTH
    n_ev = len(events)
    nL, nX = len(levels), len(lons)
    u_means = np.full((n_ev, nL, nX), np.nan, dtype=np.float32)
    q_means = np.full((n_ev, nL, nX), np.nan, dtype=np.float32)
    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_all >= pd.Timestamp(ev["start_date"])) & \
               (time_all <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        u_means[i] = np.nanmean(data_u[mask], axis=0)
        q_means[i] = np.nanmean(data_q[mask], axis=0)
    return u_means, q_means, levels, lons, dx_m


def compute_mjo_event_means(events, center_lon_all, time_step3, amp_all):
    """Perturbation: per-event mean u and q in aligned coords."""
    data_u, data_q, time_field, levels, lons = _load_uq()
    dlon = np.abs(lons[1] - lons[0])
    n_rel = int((REL_LON_EAST - REL_LON_WEST) / dlon) + 1
    rel_lons = np.linspace(REL_LON_WEST, REL_LON_EAST, n_rel)
    dx_m = np.abs(rel_lons[1] - rel_lons[0]) * np.pi / 180 * R_EARTH

    n_ev = len(events)
    nL = len(levels)
    u_means = np.full((n_ev, nL, n_rel), np.nan, dtype=np.float32)
    q_means = np.full((n_ev, nL, n_rel), np.nan, dtype=np.float32)

    for i, (_, ev) in enumerate(events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        fi_mask = (time_field >= ts) & (time_field <= te)
        si_mask = (time_step3 >= ts) & (time_step3 <= te)
        fi_idx = np.where(fi_mask)[0]
        si_idx = np.where(si_mask)[0]
        if len(fi_idx) == 0:
            continue
        u_list, q_list = [], []
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
            u_list.append(_align_day(data_u[fi], lons, c, rel_lons, dlon))
            q_list.append(_align_day(data_q[fi], lons, c, rel_lons, dlon))
        if u_list:
            u_means[i] = np.nanmean(np.array(u_list), axis=0)
            q_means[i] = np.nanmean(np.array(q_list), axis=0)

    return u_means, q_means, levels, rel_lons, dx_m


def compute_cross_terms(u_means, q_means, dx_m):
    """
    Decompose -u·∂q/∂x into cross terms.
    u_means, q_means: (n_ev, nL, nX)
    Returns: termA, termB, termC (each n_ev, nL, nX)
      termA = -<ū>·∂q'ᵢ/∂x  (mean wind × anomalous moisture grad)
      termB = -u'ᵢ·∂<q̄>/∂x  (anomalous wind × mean moisture grad)
      termC = -u'ᵢ·∂q'ᵢ/∂x  (nonlinear)
    """
    # Grand means (over events with valid data)
    valid_mask = np.isfinite(u_means[:, 0, 0])
    u_bar = np.nanmean(u_means[valid_mask], axis=0)  # (nL, nX)
    q_bar = np.nanmean(q_means[valid_mask], axis=0)

    n_ev = u_means.shape[0]
    termA = np.full_like(u_means, np.nan)
    termB = np.full_like(u_means, np.nan)
    termC = np.full_like(u_means, np.nan)

    dq_bar_dx = np.gradient(q_bar, dx_m, axis=-1)  # mean moisture gradient

    for i in range(n_ev):
        if not np.isfinite(u_means[i, 0, 0]):
            continue
        u_prime = u_means[i] - u_bar
        q_prime = q_means[i] - q_bar
        dq_prime_dx = np.gradient(q_prime, dx_m, axis=-1)
        termA[i] = -u_bar * dq_prime_dx
        termB[i] = -u_prime * dq_bar_dx
        termC[i] = -u_prime * dq_prime_dx

    return termA, termB, termC


# ====== PLOT ======
def _setup_height_axis(ax, xlabel):
    ax.set_ylim(0.1, 12.5)
    ax.set_ylabel("Height (km)", fontsize=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=10, direction="in", top=True, right=False)
    for s in ax.spines.values():
        s.set_linewidth(1.2)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in P_TICKS])
    ax2.set_yticklabels([str(p) for p in P_TICKS], fontsize=9)
    ax2.set_ylabel("hPa", fontsize=11)
    ax2.tick_params(direction="in", length=4)

def plot_2d(field_h, x_vals, title, cbar_label, out_path,
            sig_h=None, is_rel=False, vmax_ov=None):
    fig, ax = plt.subplots(figsize=(14 if not is_rel else 16, 5.5), dpi=150)
    vmax = vmax_ov or np.nanmax(np.abs(field_h))
    if vmax == 0: vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    clevs = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(x_vals, TARGET_HEIGHTS, field_h,
                     levels=clevs, cmap="RdBu_r", norm=norm, extend="both")
    if sig_h is not None:
        skip = 3 if is_rel else 2
        for i in range(len(TARGET_HEIGHTS)):
            for j in range(0, len(x_vals), skip):
                if sig_h[i, j]:
                    ax.plot(x_vals[j], TARGET_HEIGHTS[i], 'k+',
                            markersize=5, markeredgewidth=1.0, alpha=0.9)
    if is_rel:
        ax.axvline(0, color="limegreen", lw=2.5, ls="--", alpha=0.9)
        ax.set_xlim(REL_LON_WEST, REL_LON_EAST)
        ax.set_xticks(np.arange(REL_LON_WEST, REL_LON_EAST + 1, 30))
    xlabel = "Relative Longitude (deg)" if is_rel else "Longitude (deg E)"
    _setup_height_axis(ax, xlabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    cbar_ax = fig.add_axes([0.12, 0.04, 0.78, 0.025])
    fig.colorbar(cf, cax=cbar_ax, orientation="horizontal").set_label(cbar_label, fontsize=11)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def run_cross_term_analysis(term_data, levels, x_vals, phase_speed,
                            fast_mask, slow_mask, prefix, label,
                            out_dir, is_rel=False):
    """Run mean + correlation + pair + diff for one cross term."""
    UNIT = "(kg/kg)\u00b7(m/s)/m / amp"

    # Mean
    grand_mean = np.nanmean(term_data, axis=0)
    mean_h = interp_to_height(grand_mean, levels)
    plot_2d(mean_h, x_vals,
            f"Mean: {label}",
            UNIT, out_dir / f"{prefix}_mean.png",
            is_rel=is_rel)

    # Correlation
    r, p = correlate_2d(term_data, phase_speed)
    sig = (p < SIG_ALPHA) & np.isfinite(p)
    r_h = interp_to_height(r, levels)
    sig_h = interp_to_height(sig.astype(float), levels) > 0.5
    n_sig = int(np.sum(sig_h & np.isfinite(r_h)))
    n_tot = int(np.sum(np.isfinite(r_h)))
    pct = n_sig / n_tot * 100 if n_tot > 0 else 0
    vmax_r = min(max(0.3, np.nanmax(np.abs(r_h)) * 1.1), 1.0)
    plot_2d(r_h, x_vals,
            f"Corr({label}, Phase Speed)  Sig(p<0.05): {n_sig}/{n_tot} ({pct:.1f}%)",
            "Pearson r", out_dir / f"{prefix}_corr.png",
            sig_h=sig_h, is_rel=is_rel, vmax_ov=vmax_r)

    # Fast/Slow pair
    fast_mean = np.nanmean(term_data[fast_mask], axis=0)
    slow_mean = np.nanmean(term_data[slow_mask], axis=0)
    fast_h = interp_to_height(fast_mean, levels)
    slow_h = interp_to_height(slow_mean, levels)
    # Plot pair as two subplots
    fig, axes = plt.subplots(1, 2, figsize=(22 if is_rel else 20, 5.5), dpi=150)
    vmax_p = max(np.nanmax(np.abs(fast_h)), np.nanmax(np.abs(slow_h)))
    if vmax_p == 0: vmax_p = 1.0
    for ax, data_h, grp_label in [(axes[0], fast_h, f"Fast (N={fast_mask.sum()})"),
                                   (axes[1], slow_h, f"Slow (N={slow_mask.sum()})")]:
        norm = TwoSlopeNorm(vmin=-vmax_p, vcenter=0, vmax=vmax_p)
        clevs = np.linspace(-vmax_p, vmax_p, 21)
        cf = ax.contourf(x_vals, TARGET_HEIGHTS, data_h,
                         levels=clevs, cmap="RdBu_r", norm=norm, extend="both")
        if is_rel:
            ax.axvline(0, color="limegreen", lw=2.5, ls="--", alpha=0.9)
            ax.set_xlim(REL_LON_WEST, REL_LON_EAST)
        xlabel = "Relative Longitude (deg)" if is_rel else "Longitude (deg E)"
        _setup_height_axis(ax, xlabel)
        ax.set_title(f"{grp_label}: {label}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    cbar_ax = fig.add_axes([0.12, 0.04, 0.78, 0.025])
    fig.colorbar(cf, cax=cbar_ax, orientation="horizontal").set_label(UNIT, fontsize=11)
    plt.savefig(out_dir / f"{prefix}_pair.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {prefix}_pair.png")

    # Group diff
    diff, p_diff = group_diff_2d(term_data, fast_mask, slow_mask)
    sig_diff = (p_diff < SIG_ALPHA) & np.isfinite(p_diff)
    diff_h = interp_to_height(diff, levels)
    sig_diff_h = interp_to_height(sig_diff.astype(float), levels) > 0.5
    n_sig_d = int(np.sum(sig_diff_h & np.isfinite(diff_h)))
    n_tot_d = int(np.sum(np.isfinite(diff_h)))
    pct_d = n_sig_d / n_tot_d * 100 if n_tot_d > 0 else 0
    plot_2d(diff_h, x_vals,
            f"Diff (Fast\u2212Slow) {label}  Sig(p<0.05): {n_sig_d}/{n_tot_d} ({pct_d:.1f}%)",
            f"Delta {UNIT}", out_dir / f"{prefix}_diff.png",
            sig_h=sig_diff_h, is_rel=is_rel)

    return pct, pct_d


# ====== MAIN ======
def main():
    print("=" * 70)
    print("06f: Cross-Term Decomposition of Moisture Advection")
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
    ps_valid = phase_speed[np.isfinite(phase_speed)]
    mu, sigma = np.mean(ps_valid), np.std(ps_valid)
    fast_mask = phase_speed > mu + GROUP_SIGMA * sigma
    slow_mask = phase_speed < mu - GROUP_SIGMA * sigma
    print(f"  Events: {len(merged)}, Fast={fast_mask.sum()}, Slow={slow_mask.sum()}")

    results = {}

    # --- Background Frame ---
    print("\n--- Background Frame ---")
    u_bg, q_bg, levels, lons, dx_bg = compute_bg_event_means(merged)
    tA_bg, tB_bg, tC_bg = compute_cross_terms(u_bg, q_bg, dx_bg)

    r1 = run_cross_term_analysis(
        tA_bg, levels, lons, phase_speed, fast_mask, slow_mask,
        "bg_termA", "Bg TermA (\u2212\u016b\u00b7\u2202q\'/\u2202x)", OUT_DIR)
    r2 = run_cross_term_analysis(
        tB_bg, levels, lons, phase_speed, fast_mask, slow_mask,
        "bg_termB", "Bg TermB (\u2212u\'\u00b7\u2202\u0071\u0304/\u2202x)", OUT_DIR)

    results["bg_termA"] = r1
    results["bg_termB"] = r2

    # --- Perturbation Frame ---
    print("\n--- Perturbation (MJO-aligned) Frame ---")
    u_mjo, q_mjo, levels_m, rel_lons, dx_mjo = compute_mjo_event_means(
        merged, center_lon_all, time_step3, amp_all)
    tA_mjo, tB_mjo, tC_mjo = compute_cross_terms(u_mjo, q_mjo, dx_mjo)

    r3 = run_cross_term_analysis(
        tA_mjo, levels_m, rel_lons, phase_speed, fast_mask, slow_mask,
        "mjo_termA", "MJO TermA (\u2212\u016b\u00b7\u2202q\'/\u2202x)", OUT_DIR, is_rel=True)
    r4 = run_cross_term_analysis(
        tB_mjo, levels_m, rel_lons, phase_speed, fast_mask, slow_mask,
        "mjo_termB", "MJO TermB (\u2212u\'\u00b7\u2202\u0071\u0304/\u2202x)", OUT_DIR, is_rel=True)

    results["mjo_termA"] = r3
    results["mjo_termB"] = r4

    # --- Summary Bar Chart ---
    print("\n--- Summary ---")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    labels = ["TermA\n(\u2212\u016b\u00b7\u2202q\'/\u2202x)", "TermB\n(\u2212u\'\u00b7\u2202q\u0304/\u2202x)"]
    colors = ['#2196F3', '#FF5722']

    # Correlation sig %
    ax = axes[0]
    bg_corr = [results["bg_termA"][0], results["bg_termB"][0]]
    mjo_corr = [results["mjo_termA"][0], results["mjo_termB"][0]]
    x = np.arange(2)
    ax.bar(x - 0.18, bg_corr, 0.35, label="Background", color=colors[0], alpha=0.8)
    ax.bar(x + 0.18, mjo_corr, 0.35, label="Perturbation", color=colors[1], alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Sig Grid Points (%)", fontsize=11)
    ax.set_title("Corr with Phase Speed", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ylim_top = ax.get_ylim()[1]
    offset = ylim_top * 0.02 if ylim_top > 0 else 0.1
    ax.set_ylim(0, ylim_top * 1.1)
    for i, v in enumerate(bg_corr):
        ax.text(i - 0.18, v + offset, f"{v:.1f}%", ha='center', fontsize=9)
    for i, v in enumerate(mjo_corr):
        ax.text(i + 0.18, v + offset, f"{v:.1f}%", ha='center', fontsize=9)

    # Group diff sig %
    ax = axes[1]
    bg_diff = [results["bg_termA"][1], results["bg_termB"][1]]
    mjo_diff = [results["mjo_termA"][1], results["mjo_termB"][1]]
    ax.bar(x - 0.18, bg_diff, 0.35, label="Background", color=colors[0], alpha=0.8)
    ax.bar(x + 0.18, mjo_diff, 0.35, label="Perturbation", color=colors[1], alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Sig Grid Points (%)", fontsize=11)
    ax.set_title("Fast\u2212Slow Diff", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ylim_top = ax.get_ylim()[1]
    offset = ylim_top * 0.02 if ylim_top > 0 else 0.1
    ax.set_ylim(0, ylim_top * 1.1)
    for i, v in enumerate(bg_diff):
        ax.text(i - 0.18, v + offset, f"{v:.1f}%", ha='center', fontsize=9)
    for i, v in enumerate(mjo_diff):
        ax.text(i + 0.18, v + offset, f"{v:.1f}%", ha='center', fontsize=9)

    plt.suptitle("Cross-Term Decomposition Summary", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "summary_cross_terms.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: summary_cross_terms.png")

    print(f"\nAll plots saved to: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
