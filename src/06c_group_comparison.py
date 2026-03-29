# -*- coding: utf-8 -*-
"""
06c_group_comparison.py
高相速度 vs 低相速度分组对比 (v3)
4 plots x 5 vars = 20 plots
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
from scipy.ndimage import gaussian_filter
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# ======================
# PATHS
# ======================
DERIVED_DIR = Path(r"E:\Datas\Derived")
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")
GRP_BG_DIR = FIG_DIR / "group_background"
GRP_MJO_DIR = FIG_DIR / "group_perturbation"

VARIABLES = ["u", "v", "w", "q", "t"]
VAR_LONG = {"u": "Zonal Wind (u)", "v": "Meridional Wind (v)",
            "w": "Vertical Velocity (omega)", "q": "Specific Humidity (q)",
            "t": "Temperature (T)"}
VAR_UNIT = {"u": "m/s / amp", "v": "m/s / amp", "w": "Pa/s / amp",
            "q": "kg/kg / amp", "t": "K / amp"}

REL_LON_WEST, REL_LON_EAST = -180.0, 180.0
AMP_THRESHOLD = 0.5
GROUP_THRESHOLD_STD = 0.7
SIG_ALPHA = 0.05

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}
TARGET_HEIGHTS = np.linspace(0.5, 12, 24)
P_TICKS = [1000, 850, 700, 500, 400, 300, 200]

U_QUIV_SCALE = 120
W_VERT_SCALE = 300
W_QUIV_SCALE = 80
QUIV_WIDTH = 0.003
QUIV_SKIP_X_BG = 5
QUIV_SKIP_X_MJO = 8
QUIV_SKIP_Y = 2


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


def load_var_data(var):
    ds = xr.open_dataset(DERIVED_DIR / f"era5_mjo_recon_{var}_norm_1979-2022.nc")
    vn = [k for k in ds.data_vars if var in k.lower()][0]
    da = _rename_level(ds[vn])
    time_field = pd.to_datetime(da["time"].values)
    levels = da["level"].values
    lons = da["lon"].values
    data = da.values
    dlon = np.abs(lons[1] - lons[0])
    return time_field, levels, lons, data, dlon, ds


def compute_group_data(var, events, group_mask, mode="bg",
                       center_lon_all=None, time_step3=None, amp_all=None):
    time_field, levels, lons, data, dlon, ds = load_var_data(var)

    group_events = events[group_mask]
    n_ev = len(group_events)
    nL = len(levels)

    if mode == "mjo":
        n_x = int((REL_LON_EAST - REL_LON_WEST) / dlon) + 1
        x_axis = np.linspace(REL_LON_WEST, REL_LON_EAST, n_x)
    else:
        n_x = len(lons)
        x_axis = lons

    event_means = np.full((n_ev, nL, n_x), np.nan, dtype=np.float32)
    day_sum = np.zeros((nL, n_x), dtype=np.float64)
    day_cnt = np.zeros((nL, n_x), dtype=np.float64)

    for i, (_, ev) in enumerate(group_events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        fi_mask = (time_field >= ts) & (time_field <= te)
        fi_idx = np.where(fi_mask)[0]
        if len(fi_idx) == 0:
            continue

        if mode == "bg":
            block = data[fi_idx, :, :]
            event_means[i] = np.nanmean(block, axis=0)
            finite = np.isfinite(block)
            day_sum += np.where(finite, block, 0).sum(axis=0)
            day_cnt += finite.sum(axis=0)
        else:
            si_mask_t = (time_step3 >= ts) & (time_step3 <= te)
            si_idx = np.where(si_mask_t)[0]
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
                s = _align_day(data[fi], lons, c, x_axis, dlon)
                samples.append(s)
                finite = np.isfinite(s)
                day_sum += np.where(finite, s, 0)
                day_cnt += finite
            if samples:
                event_means[i] = np.nanmean(np.array(samples), axis=0)

    day_cnt[day_cnt == 0] = np.nan
    daywise_mean = (day_sum / day_cnt).astype(np.float32)
    ds.close()
    return daywise_mean, event_means, levels, x_axis


def group_diff_ttest(em_hi, em_lo, nL, n_x):
    p_map = np.full((nL, n_x), np.nan, dtype=np.float32)
    for k in range(nL):
        for j in range(n_x):
            h, l = em_hi[:, k, j], em_lo[:, k, j]
            vh, vl = np.isfinite(h), np.isfinite(l)
            if vh.sum() >= 3 and vl.sum() >= 3:
                _, p_map[k, j] = stats.ttest_ind(h[vh], l[vl], equal_var=False)
    return p_map


# ======================
# PLOT FUNCTIONS
# ======================
def _setup_height_axis(ax, xlabel):
    ax.set_ylim(0.1, 12.5)
    ax.set_ylabel("Height (km)", fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.tick_params(axis="both", which="major", labelsize=9,
                   direction="in", top=True, right=False, length=4)
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in P_TICKS])
    ax2.set_yticklabels([str(p) for p in P_TICKS], fontsize=8)
    ax2.set_ylabel("hPa", fontsize=9)
    ax2.tick_params(direction="in", length=3)
    return ax2


def _add_var_arrows(ax, x_vals, field_h, var, skip_x):
    if var not in ("u", "w"):
        return
    sm = gaussian_filter(np.nan_to_num(field_h, nan=0), sigma=1.0)
    nm = np.isnan(field_h)
    sm[nm] = np.nan
    X, Y = np.meshgrid(x_vals, TARGET_HEIGHTS)
    sx, sy = skip_x, QUIV_SKIP_Y
    if var == "u":
        ax.quiver(X[::sy, ::sx], Y[::sy, ::sx],
                  sm[::sy, ::sx], np.zeros_like(sm[::sy, ::sx]),
                  color='black', scale=U_QUIV_SCALE, width=QUIV_WIDTH,
                  headwidth=2.5, headlength=2, headaxislength=1.8,
                  pivot='middle', alpha=1.0)
    elif var == "w":
        w_arr = -sm * W_VERT_SCALE
        ax.quiver(X[::sy, ::sx], Y[::sy, ::sx],
                  np.zeros_like(w_arr[::sy, ::sx]), w_arr[::sy, ::sx],
                  color='black', scale=W_QUIV_SCALE, width=QUIV_WIDTH,
                  headwidth=2.5, headlength=2, headaxislength=1.8,
                  pivot='middle', alpha=1.0)


def plot_pair(hi_h, lo_h, x_vals, var, n_hi, n_lo, out_path,
              xlabel, title_mode="Background", center_line=None, skip_x=5):
    fig, axes = plt.subplots(1, 2, figsize=(16 if 'Longitude' in xlabel else 18, 5.5), dpi=150)
    vmax = max(np.nanmax(np.abs(hi_h)), np.nanmax(np.abs(lo_h)))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    clevs = np.linspace(-vmax, vmax, 21)

    for idx, (ax, data_h, label, n) in enumerate([
        (axes[0], hi_h, "Fast", n_hi), (axes[1], lo_h, "Slow", n_lo),
    ]):
        cf = ax.contourf(x_vals, TARGET_HEIGHTS, data_h,
                         levels=clevs, cmap="RdBu_r", norm=norm, extend="both")
        _add_var_arrows(ax, x_vals, data_h, var, skip_x)
        if center_line is not None:
            ax.axvline(center_line, color="limegreen", lw=2, ls="--", alpha=0.8)
        if 'Relative' in xlabel:
            ax.set_xlim(REL_LON_WEST, REL_LON_EAST)
            ax.set_xticks(np.arange(REL_LON_WEST, REL_LON_EAST + 1, 30))
        _setup_height_axis(ax, xlabel)
        ax.set_title(f"({'a' if idx == 0 else 'b'}) {label} (N={n})",
                     fontsize=12, fontweight="bold")
        if idx == 1:
            ax.set_ylabel("")

    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.025])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"{VAR_LONG[var]} ({VAR_UNIT[var]})", fontsize=9)
    fig.suptitle(f"{title_mode}: {VAR_LONG[var]} - Fast vs Slow",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.subplots_adjust(bottom=0.12, wspace=0.30)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_diff(diff_h, sig_h, x_vals, var, n_hi, n_lo, out_path,
              xlabel, title_mode="Background", center_line=None):
    fig, ax = plt.subplots(figsize=(14 if 'Longitude' not in xlabel or 'Relative' not in xlabel else 16, 5.5), dpi=150)
    vmax = np.nanmax(np.abs(diff_h))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    clevs = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(x_vals, TARGET_HEIGHTS, diff_h,
                     levels=clevs, cmap="RdBu_r", norm=norm, extend="both")
    for i in range(len(TARGET_HEIGHTS)):
        step = 2 if len(x_vals) < 200 else 3
        for j in range(0, len(x_vals), step):
            if sig_h[i, j]:
                ax.plot(x_vals[j], TARGET_HEIGHTS[i], 'k+',
                        markersize=5, markeredgewidth=1.0, alpha=0.9)
    if center_line is not None:
        ax.axvline(center_line, color="limegreen", lw=2, ls="--", alpha=0.8)
    if 'Relative' in xlabel:
        ax.set_xlim(REL_LON_WEST, REL_LON_EAST)
        ax.set_xticks(np.arange(REL_LON_WEST, REL_LON_EAST + 1, 30))
    _setup_height_axis(ax, xlabel)
    n_sig = int(np.sum(sig_h & np.isfinite(diff_h)))
    n_total = int(np.sum(np.isfinite(diff_h)))
    pct = n_sig / n_total * 100 if n_total > 0 else 0
    ax.set_title(
        f"{title_mode} Diff (Fast-Slow): {VAR_LONG[var]}  "
        f"Fast N={n_hi}, Slow N={n_lo}  Sig(p<0.05): {n_sig}/{n_total} ({pct:.1f}%)",
        fontsize=11, fontweight="bold")
    plt.subplots_adjust(bottom=0.18)
    cbar_ax = fig.add_axes([0.12, 0.06, 0.78, 0.03])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"Delta ({VAR_UNIT[var]})", fontsize=11)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


# ======================
# MAIN
# ======================
def main():
    print("=" * 70)
    print("06c: Fast vs Slow Phase Speed Group Comparison (v3)")
    print("=" * 70)
    GRP_BG_DIR.mkdir(parents=True, exist_ok=True)
    GRP_MJO_DIR.mkdir(parents=True, exist_ok=True)

    ds3 = xr.open_dataset(STEP3_NC)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    amp_all = ds3["amp"].values.astype(float)
    time_step3 = pd.to_datetime(ds3["time"].values)
    ds3.close()

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    speed = merged["phase_speed_m_s"].values.astype(float)
    sp_mean, sp_std = np.nanmean(speed), np.nanstd(speed)
    hi_thr = sp_mean + GROUP_THRESHOLD_STD * sp_std
    lo_thr = sp_mean - GROUP_THRESHOLD_STD * sp_std
    hi_mask = speed > hi_thr
    lo_mask = speed < lo_thr
    n_hi, n_lo = int(hi_mask.sum()), int(lo_mask.sum())
    print(f"  Speed: mean={sp_mean:.2f}, std={sp_std:.2f}")
    print(f"  Fast (>{hi_thr:.2f}): N={n_hi}")
    print(f"  Slow (<{lo_thr:.2f}): N={n_lo}")

    for var in VARIABLES:
        print(f"\n{'='*50}\nProcessing: {var}\n{'='*50}")

        # --- Background ---
        print("  [BG]...")
        hi_dm_bg, hi_em_bg, levels, lons = compute_group_data(var, merged, hi_mask, "bg")
        lo_dm_bg, lo_em_bg, _, _ = compute_group_data(var, merged, lo_mask, "bg")
        hi_h_bg = interp_to_height(hi_dm_bg, levels)
        lo_h_bg = interp_to_height(lo_dm_bg, levels)

        plot_pair(hi_h_bg, lo_h_bg, lons, var, n_hi, n_lo,
                  GRP_BG_DIR / f"bg_{var}_pair.png",
                  xlabel="Longitude (deg E)", title_mode="Background",
                  skip_x=QUIV_SKIP_X_BG)

        diff_bg = hi_dm_bg - lo_dm_bg
        p_bg = group_diff_ttest(hi_em_bg, lo_em_bg, len(levels), len(lons))
        sig_bg = (p_bg < SIG_ALPHA) & np.isfinite(p_bg)
        diff_h_bg = interp_to_height(diff_bg, levels)
        sig_h_bg = interp_to_height(sig_bg.astype(float), levels) > 0.5
        plot_diff(diff_h_bg, sig_h_bg, lons, var, n_hi, n_lo,
                  GRP_BG_DIR / f"bg_{var}_diff.png",
                  xlabel="Longitude (deg E)", title_mode="Background")

        # --- MJO Aligned ---
        print("  [MJO]...")
        hi_dm_mjo, hi_em_mjo, levels, rel_lons = compute_group_data(
            var, merged, hi_mask, "mjo", center_lon_all, time_step3, amp_all)
        lo_dm_mjo, lo_em_mjo, _, _ = compute_group_data(
            var, merged, lo_mask, "mjo", center_lon_all, time_step3, amp_all)
        hi_h_mjo = interp_to_height(hi_dm_mjo, levels)
        lo_h_mjo = interp_to_height(lo_dm_mjo, levels)

        plot_pair(hi_h_mjo, lo_h_mjo, rel_lons, var, n_hi, n_lo,
                  GRP_MJO_DIR / f"mjo_{var}_pair.png",
                  xlabel="Relative Longitude (deg)", title_mode="MJO Perturbation",
                  center_line=0, skip_x=QUIV_SKIP_X_MJO)

        diff_mjo = hi_dm_mjo - lo_dm_mjo
        p_mjo = group_diff_ttest(hi_em_mjo, lo_em_mjo, len(levels), len(rel_lons))
        sig_mjo = (p_mjo < SIG_ALPHA) & np.isfinite(p_mjo)
        diff_h_mjo = interp_to_height(diff_mjo, levels)
        sig_h_mjo = interp_to_height(sig_mjo.astype(float), levels) > 0.5
        plot_diff(diff_h_mjo, sig_h_mjo, rel_lons, var, n_hi, n_lo,
                  GRP_MJO_DIR / f"mjo_{var}_diff.png",
                  xlabel="Relative Longitude (deg)", title_mode="MJO Perturbation",
                  center_line=0)

    print(f"\nAll 20 plots saved.\nDone!")


if __name__ == "__main__":
    main()
