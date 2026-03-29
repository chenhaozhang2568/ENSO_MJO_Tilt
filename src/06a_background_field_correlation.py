# -*- coding: utf-8 -*-
"""
06a_background_field_correlation.py
背景场（绝对坐标）与MJO相速度的逐格点相关性分析 (v3)
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
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")
CORR_DIR = FIG_DIR / "background"
OUT_NC = DERIVED_DIR / "field_bg_correlation_1979-2022.nc"

VARIABLES = ["u", "v", "w", "q", "t"]
VAR_LONG = {"u": "Zonal Wind (u)", "v": "Meridional Wind (v)",
            "w": "Vertical Velocity (omega)", "q": "Specific Humidity (q)",
            "t": "Temperature (T)"}
VAR_UNIT = {"u": "m/s / amp", "v": "m/s / amp", "w": "Pa/s / amp",
        "q": "kg/kg / amp", "t": "K / amp"}

SIG_ALPHA = 0.05

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}
TARGET_HEIGHTS = np.linspace(0.5, 12, 24)
P_TICKS = [1000, 850, 700, 500, 400, 300, 200]

# Arrow display parameters
U_QUIV_SCALE = 120
W_VERT_SCALE = 300      # multiply omega for vertical arrow display
W_QUIV_SCALE = 80
QUIV_WIDTH = 0.003
QUIV_SKIP_X = 5
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





def compute_bg_data(var, events):
    ds = xr.open_dataset(DERIVED_DIR / f"era5_mjo_recon_{var}_norm_1979-2022.nc")
    
    # 针对 u/v/w/q/t 获取变量名
    vn = [k for k in ds.data_vars if var in k.lower()][0]
    da = _rename_level(ds[vn])
    time_all = pd.to_datetime(da["time"].values)
    levels = da["level"].values
    lons = da["lon"].values
    data = da.values

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
        block = data[mask, :, :]
        event_means[i] = np.nanmean(block, axis=0)
        finite = np.isfinite(block)
        day_sum += np.where(finite, block, 0).sum(axis=0)
        day_cnt += finite.sum(axis=0)

    day_cnt[day_cnt == 0] = np.nan
    daywise_mean = (day_sum / day_cnt).astype(np.float32)
    ds.close()
    return event_means, daywise_mean, levels, lons


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
    ax2.set_ylabel("Pressure (hPa)", fontsize=11)
    ax2.tick_params(direction="in", length=4)
    return ax2


def _add_var_arrows(ax, x_vals, field_h, var):
    """Add variable-specific arrows: u=horizontal, w=vertical, others=none."""
    if var not in ("u", "w"):
        return
    sm = gaussian_filter(np.nan_to_num(field_h, nan=0), sigma=1.0)
    nm = np.isnan(field_h)
    sm[nm] = np.nan
    X, Y = np.meshgrid(x_vals, TARGET_HEIGHTS)
    sx, sy = QUIV_SKIP_X, QUIV_SKIP_Y

    if var == "u":
        ax.quiver(X[::sy, ::sx], Y[::sy, ::sx],
                  sm[::sy, ::sx], np.zeros_like(sm[::sy, ::sx]),
                  color='black', scale=U_QUIV_SCALE, width=QUIV_WIDTH,
                  headwidth=2.5, headlength=2, headaxislength=1.8,
                  pivot='middle', alpha=1.0)
        q = ax.quiver([], [], [], [], color='black', scale=U_QUIV_SCALE,
                       width=QUIV_WIDTH, headwidth=2.5, pivot='middle')
        ax.quiverkey(q, 0.88, 0.03, 2, '2 m/s', labelpos='E',
                     coordinates='axes', fontproperties={'size': 8})
    elif var == "w":
        w_arr = -sm * W_VERT_SCALE
        ax.quiver(X[::sy, ::sx], Y[::sy, ::sx],
                  np.zeros_like(w_arr[::sy, ::sx]), w_arr[::sy, ::sx],
                  color='black', scale=W_QUIV_SCALE, width=QUIV_WIDTH,
                  headwidth=2.5, headlength=2, headaxislength=1.8,
                  pivot='middle', alpha=1.0)
        q = ax.quiver([], [], [], [], color='black', scale=W_QUIV_SCALE,
                       width=QUIV_WIDTH, headwidth=2.5, pivot='middle')
        ref_val = 0.01 * W_VERT_SCALE
        ax.quiverkey(q, 0.88, 0.03, ref_val, '0.01 Pa/s', labelpos='E',
                     coordinates='axes', fontproperties={'size': 8})


def plot_corr_map(r_h, sig_h, x_vals, var, out_path):
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    vmax = min(max(0.3, np.nanmax(np.abs(r_h)) * 1.1), 1.0)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    clevs = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(x_vals, TARGET_HEIGHTS, r_h,
                     levels=clevs, cmap="RdBu_r", norm=norm, extend="both")
    for i in range(len(TARGET_HEIGHTS)):
        for j in range(0, len(x_vals), 2):
            if sig_h[i, j]:
                ax.plot(x_vals[j], TARGET_HEIGHTS[i], 'k+',
                        markersize=5, markeredgewidth=1.0, alpha=0.9)
    _setup_height_axis(ax)
    n_sig = int(np.sum(sig_h & np.isfinite(r_h)))
    n_total = int(np.sum(np.isfinite(r_h)))
    pct = n_sig / n_total * 100 if n_total > 0 else 0
    ax.set_title(
        f"Background: Corr({VAR_LONG[var]}, Phase Speed)  "
        f"Sig(p<0.05): {n_sig}/{n_total} ({pct:.1f}%)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    cbar_ax = fig.add_axes([0.12, 0.04, 0.78, 0.025])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Pearson r", fontsize=11)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_mean_field(field_h, x_vals, var, out_path):
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    vmax = np.nanmax(np.abs(field_h))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    clevs = np.linspace(-vmax, vmax, 21)
    cf = ax.contourf(x_vals, TARGET_HEIGHTS, field_h,
                     levels=clevs, cmap="RdBu_r", norm=norm, extend="both")
    _add_var_arrows(ax, x_vals, field_h, var)
    _setup_height_axis(ax)
    ax.set_title(f"Background Mean: {VAR_LONG[var]} (daywise avg)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.24)
    cbar_ax = fig.add_axes([0.12, 0.04, 0.78, 0.025])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(f"{VAR_UNIT[var]}", fontsize=11)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


# ======================
# MAIN
# ======================
def main():
    print("=" * 70)
    print("06a: Background Field vs Phase Speed Correlation (v3)")
    print("=" * 70)
    CORR_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)
    print(f"  Events: {len(merged)}, valid speed: {np.sum(np.isfinite(phase_speed))}")

    nc_vars = {}

    for var in VARIABLES:
        print(f"\n{'='*50}\nProcessing: {var}\n{'='*50}")
        event_means, daywise_mean, levels, lons = compute_bg_data(var, merged)

        r_map, p_map = correlate_with_phase_speed(event_means, phase_speed)
        sig_mask = (p_map < SIG_ALPHA) & np.isfinite(p_map)
        n_sig = int(np.sum(sig_mask & np.isfinite(r_map)))
        print(f"  r range: [{np.nanmin(r_map):.3f}, {np.nanmax(r_map):.3f}], "
              f"p<0.05 sig: {n_sig}/{np.sum(np.isfinite(r_map)).astype(int)}")

        r_h = interp_to_height(r_map, levels)
        sig_h = interp_to_height(sig_mask.astype(float), levels) > 0.5
        mean_h = interp_to_height(daywise_mean, levels)

        plot_corr_map(r_h, sig_h, lons, var, CORR_DIR / f"bg_{var}_corr.png")
        plot_mean_field(mean_h, lons, var, CORR_DIR / f"bg_mean_{var}.png")

        nc_vars[f"bg_mean_{var}"] = xr.DataArray(
            daywise_mean, dims=("level", "lon"),
            coords={"level": levels, "lon": lons})
        nc_vars[f"bg_corr_{var}"] = xr.DataArray(
            r_map, dims=("level", "lon"),
            coords={"level": levels, "lon": lons})
        nc_vars[f"bg_pval_{var}"] = xr.DataArray(
            p_map, dims=("level", "lon"),
            coords={"level": levels, "lon": lons})
        nc_vars[f"bg_sig_{var}"] = xr.DataArray(
            sig_mask.astype(np.int8), dims=("level", "lon"),
            coords={"level": levels, "lon": lons})

    ds_out = xr.Dataset(nc_vars, attrs={
        "description": "Background field correlation (v3)",
        "sig_threshold": str(SIG_ALPHA), "n_events": str(len(merged)),
    })
    ds_out.to_netcdf(OUT_NC)
    print(f"\nSaved: {OUT_NC}\nDone!")


if __name__ == "__main__":
    main()
