# -*- coding: utf-8 -*-
"""
07a_2d_latlon_correlation.py
2D lat×lon MJO重构归一场 — 事件平均数值 + 与相速度逐格点相关分析

变量清单:
  气压层 (u,v,w,q,t × 9层):  从 _norm_3d 文件读取
  单层   (sst,lhf,shf,tp,olr): 从 _norm_2d 文件读取
  柱积分 (CWV, MSE):           从 q,t 的 _norm_3d 做在线柱积分

每变量每层 2 张图:
  1. mean  — 事件平均值填色 + 与相速度的显著性打点 (p<0.05)
  2. corr  — Pearson r 填色 + p<0.05 打点
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from scipy import stats
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
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
FIG_ROOT = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\2d_latlon_corr")

SIG_ALPHA = 0.05
REL_LON_WEST, REL_LON_EAST = -180.0, 180.0

# Physical constants for column integration
CP = 1004.0    # J/(kg·K)
LV = 2.501e6   # J/kg
G  = 9.81      # m/s²

# Pressure level variables
PL_VARIABLES = ["u", "v", "w", "q", "t"]
PL_LONG = {
    "u": "Zonal Wind (u)", "v": "Meridional Wind (v)",
    "w": "Vertical Velocity (ω)", "q": "Specific Humidity (q)",
    "t": "Temperature (T)"
}
PL_UNIT = {
    "u": "m/s / amp", "v": "m/s / amp", "w": "Pa/s / amp",
    "q": "kg/kg / amp", "t": "K / amp"
}

# Surface variables
SFC_VARIABLES = ["sst", "lhf", "shf", "tp", "olr"]
SFC_LONG = {
    "sst": "Sea Surface Temperature", "lhf": "Latent Heat Flux",
    "shf": "Sensible Heat Flux", "tp": "Precipitation",
    "olr": "OLR"
}
SFC_UNIT = {
    "sst": "K / amp", "lhf": "W/m² / amp", "shf": "W/m² / amp",
    "tp": "mm/day / amp", "olr": "W/m² / amp"
}


# ======================
# DATA LOADING
# ======================
def load_pl_norm_3d(var):
    """Load 3D normalized field → DataArray (time, level, lat, lon)."""
    nc = DERIVED_DIR / f"era5_mjo_recon_{var}_norm_3d_1979-2022.nc"
    if not nc.exists():
        raise FileNotFoundError(f"Missing: {nc}\nRun 02_mvEOF.py reconstruct_era5_fields_3d() first.")
    ds = xr.open_dataset(nc)
    vn = [k for k in ds.data_vars][0]
    da = ds[vn]
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    return da


def load_sfc_norm_2d(var):
    """Load 2D normalized field → DataArray (time, lat, lon)."""
    nc = DERIVED_DIR / f"era5_mjo_recon_{var}_norm_2d_1979-2022.nc"
    if not nc.exists():
        raise FileNotFoundError(f"Missing: {nc}\nRun 02c_reconstruct_surface_2d.py first.")
    ds = xr.open_dataset(nc)
    vn = [k for k in ds.data_vars][0]
    return ds[vn]


# ======================
# ANALYSIS
# ======================
def compute_event_means_2d(data_2d, time_all, events):
    """
    Compute event-mean of a 2D field (time, lat, lon) or (time, lon).
    Returns: (n_events, n_lat, n_lon) or (n_events, n_lon)
    """
    time_pd = pd.to_datetime(time_all)
    n_ev = len(events)
    spatial_shape = data_2d.shape[1:]
    event_means = np.full((n_ev, *spatial_shape), np.nan, dtype=np.float32)

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_pd >= pd.Timestamp(ev["start_date"])) & \
               (time_pd <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        event_means[i] = np.nanmean(data_2d[mask], axis=0)

    return event_means


def _align_day_2d(data_day, lons, center_lon, rel_lons, dlon):
    lon_360 = np.mod(lons, 360)
    c360 = np.mod(center_lon, 360)
    n_lat = data_day.shape[0]
    sample = np.full((n_lat, len(rel_lons)), np.nan, dtype=np.float32)
    for j, rl in enumerate(rel_lons):
        tlon = np.mod(c360 + rl, 360)
        k = np.argmin(np.abs(lon_360 - tlon))
        if np.abs(lon_360[k] - tlon) < dlon:
            sample[:, j] = data_day[:, k]
    return sample


def compute_event_means_aligned(data_2d, time_all, lons, events, center_lon_all, time_step3):
    time_pd = pd.to_datetime(time_all)
    dlon = np.abs(lons[1] - lons[0])
    n_rel = int((REL_LON_EAST - REL_LON_WEST) / dlon) + 1
    rel_lons = np.linspace(REL_LON_WEST, REL_LON_EAST, n_rel)

    n_ev = len(events)
    spatial_shape = data_2d.shape[1:]
    event_means = np.full((n_ev, spatial_shape[0], n_rel), np.nan, dtype=np.float32)

    for i, (_, ev) in enumerate(events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        fi_mask = (time_pd >= ts) & (time_pd <= te)
        si_mask = (time_step3 >= ts) & (time_step3 <= te)
        
        fi_idx = np.where(fi_mask)[0]
        si_idx = np.where(si_mask)[0]
        if len(fi_idx) == 0:
            continue
            
        samples = []
        for fi in fi_idx:
            t_val = time_pd[fi]
            si_match = [si for si in si_idx if abs((time_step3[si] - t_val).total_seconds()) < 43200]
            if not si_match:
                continue
            si = si_match[0]
            c = center_lon_all[si]
            
            if not np.isfinite(c):
                continue
                
            s = _align_day_2d(data_2d[fi], lons, c, rel_lons, dlon)
            samples.append(s)
            
        if samples:
            event_means[i] = np.nanmean(np.array(samples), axis=0)

    return event_means, rel_lons


def correlate_with_speed(event_means, phase_speed):
    """
    Compute Pearson r between event_means and phase_speed at each spatial point.
    event_means shape: (n_events, ...) — arbitrary spatial dims
    Returns: r_map, p_map with same spatial shape
    """
    spatial_shape = event_means.shape[1:]
    r_map = np.full(spatial_shape, np.nan, dtype=np.float32)
    p_map = np.full(spatial_shape, np.nan, dtype=np.float32)

    flat_spatial = np.prod(spatial_shape)
    em_flat = event_means.reshape(len(event_means), flat_spatial)

    for j in range(flat_spatial):
        v = em_flat[:, j]
        ok = np.isfinite(v) & np.isfinite(phase_speed)
        if ok.sum() < 10:
            continue
        r_map.flat[j], p_map.flat[j] = stats.pearsonr(v[ok], phase_speed[ok])

    return r_map, p_map


# ======================
# PLOTTING
# ======================
def plot_2d_field(data_2d, lats, lons, sig_mask, title, cbar_label, out_path,
                  cmap="RdBu_r", symmetric=True, r_map=None, xlabel="Longitude (°E)"):
    """
    Plot a lat×lon filled contour with significance stippling.
    If r_map is provided, use red/blue dots to indicate positive/negative correlation.
    Otherwise use black dots.
    """
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=150)

    vmax = np.nanmax(np.abs(data_2d))
    if vmax == 0:
        vmax = 1.0
    if symmetric:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        levels = np.linspace(-vmax, vmax, 21)
    else:
        norm = None
        levels = 21

    cf = ax.contourf(lons, lats, data_2d, levels=levels, cmap=cmap,
                     norm=norm, extend="both")

    # Significance stippling
    if sig_mask is not None and r_map is not None:
        # Red/blue dots for positive/negative correlation on mean field
        pos_mask = sig_mask & np.isfinite(data_2d) & (r_map > 0)
        neg_mask = sig_mask & np.isfinite(data_2d) & (r_map < 0)
        py, px = np.where(pos_mask)
        ny, nx = np.where(neg_mask)
        if len(py) > 0:
            ax.scatter(lons[px], lats[py], c='tab:red', s=22, marker='.',
                       alpha=0.85, linewidths=0, label=f'+r ({len(py)})')
        if len(ny) > 0:
            ax.scatter(lons[nx], lats[ny], c='tab:blue', s=22, marker='.',
                       alpha=0.85, linewidths=0, label=f'−r ({len(ny)})')
        if len(py) > 0 or len(ny) > 0:
            ax.legend(fontsize=8, loc='lower right', framealpha=0.7,
                      markerscale=2.5, handletextpad=0.3)
    elif sig_mask is not None:
        sig_y, sig_x = np.where(sig_mask & np.isfinite(data_2d))
        if len(sig_y) > 0:
            ax.scatter(lons[sig_x], lats[sig_y], c='black', s=22, marker='.',
                       alpha=0.8, linewidths=0)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.tick_params(labelsize=9, direction="in")
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    for s in ax.spines.values():
        s.set_linewidth(1.0)

    # Significance statistics in title
    if sig_mask is not None:
        n_sig = int(np.sum(sig_mask & np.isfinite(data_2d)))
        n_total = int(np.sum(np.isfinite(data_2d)))
        pct = n_sig / n_total * 100 if n_total > 0 else 0
        title = f"{title}  Sig(p<0.05): {n_sig}/{n_total} ({pct:.1f}%)"

    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    cbar_ax = fig.add_axes([0.12, 0.06, 0.78, 0.03])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=10)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def process_and_plot(data_3d_or_2d, time_all, lats, lons, events, phase_speed,
                     var_key, var_long, var_unit, out_dir, level_label=None,
                     mode="background", center_lon_all=None, time_step3=None):
    """Process one variable (or one level) and generate 2 plots."""
    out_dir_mode = out_dir / mode
    out_dir_mode.mkdir(parents=True, exist_ok=True)

    suffix = f"_{level_label}" if level_label else ""

    if mode == "background":
        # 1. Compute event means
        event_means = compute_event_means_2d(data_3d_or_2d, time_all, events)
        plot_lons = lons
        xlabel = "Longitude (°E)"
        title_prefix = "Event Mean"
        title_prefix_r = "Corr"
    else:
        # Align to OLR center
        event_means, plot_lons = compute_event_means_aligned(
            data_3d_or_2d, time_all, lons, events, center_lon_all, time_step3)
        xlabel = "Relative Longitude (°)"
        title_prefix = "MJO Perturbation"
        title_prefix_r = "MJO Perturbation: Corr"

    # 2. Grand mean (daywise)
    grand_mean = np.nanmean(event_means, axis=0)

    # 3. Correlate with phase speed
    r_map, p_map = correlate_with_speed(event_means, phase_speed)
    sig_mask = (p_map < SIG_ALPHA) & np.isfinite(p_map)

    # Plot 1: Mean field + red/blue stippling for correlation direction
    plot_2d_field(
        grand_mean, lats, plot_lons, sig_mask,
        f"{title_prefix}: {var_long}{(' @ ' + level_label) if level_label else ''}",
        var_unit,
        out_dir_mode / f"mean_{var_key}{suffix}.png",
        r_map=r_map, xlabel=xlabel
    )

    # Plot 2: Correlation map (black dots)
    plot_2d_field(
        r_map, lats, plot_lons, sig_mask,
        f"{title_prefix_r}({var_long}, Phase Speed){(' @ ' + level_label) if level_label else ''}",
        "Pearson r",
        out_dir_mode / f"corr_{var_key}{suffix}.png",
        xlabel=xlabel
    )


# ======================
# COLUMN INTEGRATION
# ======================
def column_integrate_3d(data_4d, levels_hPa):
    """
    Integrate (time, level, lat, lon) → (time, lat, lon).
    Uses trapezoidal rule along pressure axis.
    """
    sort_idx = np.argsort(levels_hPa)
    levels_Pa = levels_hPa[sort_idx] * 100.0
    data_sorted = data_4d[:, sort_idx, :, :]
    return np.abs(np.trapz(data_sorted, x=levels_Pa, axis=1)) / G


# ======================
# MAIN
# ======================
def main():
    print("=" * 70)
    print("07a: 2D lat×lon Correlation Analysis (MJO Recon Norm)")
    print("=" * 70)

    # Load events & phase speed
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    
    # Load step3 data for perturbation alignment
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    time_step3 = pd.to_datetime(ds3["time"].values)
    ds3.close()
    
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)
    print(f"  Events: {len(merged)}, valid speed: {np.sum(np.isfinite(phase_speed))}")

    n_plots = 0

    # ========================================
    # Part 1: Pressure-level variables
    # ========================================
    for var in PL_VARIABLES:
        print(f"\n{'='*50}\nPressure Level: {var}\n{'='*50}")
        try:
            da = load_pl_norm_3d(var)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        time_all = da["time"].values
        levels = da["level"].values
        lats = da["lat"].values
        lons = da["lon"].values
        data = da.values  # (time, level, lat, lon)

        out_dir = FIG_ROOT / var

        for lev_idx, lev in enumerate(levels):
            lev_label = f"{int(lev)}hPa"
            print(f"  Level: {lev_label}")
            process_and_plot(
                data[:, lev_idx, :, :], time_all, lats, lons,
                merged, phase_speed, var, PL_LONG[var], PL_UNIT[var],
                out_dir, lev_label, mode="background"
            )
            process_and_plot(
                data[:, lev_idx, :, :], time_all, lats, lons,
                merged, phase_speed, var, PL_LONG[var], PL_UNIT[var],
                out_dir, lev_label, mode="perturbation",
                center_lon_all=center_lon_all, time_step3=time_step3
            )
            n_plots += 4

        da.close()

    # ========================================
    # Part 2: Surface variables
    # ========================================
    for var in SFC_VARIABLES:
        print(f"\n{'='*50}\nSurface: {var}\n{'='*50}")
        try:
            da = load_sfc_norm_2d(var)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        time_all = da["time"].values
        lats = da["lat"].values
        lons = da["lon"].values
        data = da.values  # (time, lat, lon)

        out_dir = FIG_ROOT / var
        process_and_plot(
            data, time_all, lats, lons,
            merged, phase_speed, var, SFC_LONG[var], SFC_UNIT[var],
            out_dir, mode="background"
        )
        process_and_plot(
            data, time_all, lats, lons,
            merged, phase_speed, var, SFC_LONG[var], SFC_UNIT[var],
            out_dir, mode="perturbation",
            center_lon_all=center_lon_all, time_step3=time_step3
        )
        n_plots += 4
        da.close()

    # ========================================
    # Part 3: Column-integrated variables
    # ========================================
    print(f"\n{'='*50}\nColumn Integrated\n{'='*50}")

    # CWV: ∫q dp/g
    try:
        da_q = load_pl_norm_3d("q")
        levels_q = da_q["level"].values.astype(float)
        lats = da_q["lat"].values
        lons = da_q["lon"].values
        time_all = da_q["time"].values

        print("  Computing CWV (column water vapor) ...")
        cwv = column_integrate_3d(da_q.values, levels_q)  # (time, lat, lon)

        out_dir = FIG_ROOT / "column_integrated"
        process_and_plot(
            cwv, time_all, lats, lons,
            merged, phase_speed, "cwv", "Column Water Vapor (∫q dp/g)",
            "kg/m² / amp", out_dir, mode="background"
        )
        process_and_plot(
            cwv, time_all, lats, lons,
            merged, phase_speed, "cwv", "Column Water Vapor (∫q dp/g)",
            "kg/m² / amp", out_dir, mode="perturbation",
            center_lon_all=center_lon_all, time_step3=time_step3
        )
        n_plots += 4

        # MSE: ∫(CpT + Lvq) dp/g
        da_t = load_pl_norm_3d("t")
        print("  Computing Column MSE ...")
        mse_3d = CP * da_t.values + LV * da_q.values
        col_mse = column_integrate_3d(mse_3d, levels_q)

        process_and_plot(
            col_mse, time_all, lats, lons,
            merged, phase_speed, "mse", "Column MSE (∫(CpT+Lvq) dp/g)",
            "J/m² / amp", out_dir, mode="background"
        )
        process_and_plot(
            col_mse, time_all, lats, lons,
            merged, phase_speed, "mse", "Column MSE (∫(CpT+Lvq) dp/g)",
            "J/m² / amp", out_dir, mode="perturbation",
            center_lon_all=center_lon_all, time_step3=time_step3
        )
        n_plots += 4

        da_q.close()
        da_t.close()

    except FileNotFoundError as e:
        print(f"  [SKIP] Column integration skipped: {e}")

    print(f"\n{'='*70}")
    print(f"Total plots generated: {n_plots}")
    print(f"Output directory: {FIG_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
