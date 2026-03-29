# -*- coding: utf-8 -*-
"""
07b_2d_latlon_group_composite.py

按相速度分组做 2D lat×lon 复合对比分析。

分组规则：
  高相速度组: phase_speed > mean + 0.7 sigma
  低相速度组: phase_speed < mean - 0.7 sigma
  中间事件: 不参与复合

输出结构：
  每个变量目录下新增：
    - background_grouped/
    - perturbation_grouped/

  每层/变量输出：
    - composite_*.png  左: 高相速度组，右: 低相速度组
    - diff_*.png       高减低，并叠加显著性格点
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

warnings.filterwarnings("ignore")

mpl.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
mpl.rcParams["axes.unicode_minus"] = False

# ======================
# PATHS
# ======================
DERIVED_DIR = Path(r"E:\Datas\Derived")
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
FIG_ROOT = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\2d_latlon_corr")
TABLE_ROOT = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\tables")
SUMMARY_CSV = TABLE_ROOT / "2d_latlon_grouped_summary.csv"

# ======================
# SETTINGS
# ======================
SIG_ALPHA = 0.05
GROUP_SIGMA = 0.7
MIN_GROUP_SIZE = 8
REL_LON_WEST, REL_LON_EAST = -180.0, 180.0

# Physical constants for column integration
CP = 1004.0    # J/(kg·K)
LV = 2.501e6   # J/kg
G = 9.81       # m/s²

# Pressure level variables
PL_VARIABLES = ["u", "v", "w", "q", "t"]
PL_LONG = {
    "u": "Zonal Wind (u)",
    "v": "Meridional Wind (v)",
    "w": "Vertical Velocity (ω)",
    "q": "Specific Humidity (q)",
    "t": "Temperature (T)",
}
PL_UNIT = {
    "u": "m/s / amp",
    "v": "m/s / amp",
    "w": "Pa/s / amp",
    "q": "kg/kg / amp",
    "t": "K / amp",
}

# Surface variables
SFC_VARIABLES = ["sst", "lhf", "shf", "tp", "olr"]
SFC_LONG = {
    "sst": "Sea Surface Temperature",
    "lhf": "Latent Heat Flux",
    "shf": "Sensible Heat Flux",
    "tp": "Precipitation",
    "olr": "OLR",
}
SFC_UNIT = {
    "sst": "K / amp",
    "lhf": "W/m² / amp",
    "shf": "W/m² / amp",
    "tp": "mm/day / amp",
    "olr": "W/m² / amp",
}


def load_pl_norm_3d(var: str) -> xr.DataArray:
    """Load 3D normalized field -> DataArray(time, level, lat, lon)."""
    nc = DERIVED_DIR / f"era5_mjo_recon_{var}_norm_3d_1979-2022.nc"
    if not nc.exists():
        raise FileNotFoundError(f"Missing: {nc}")
    ds = xr.open_dataset(nc)
    vn = [k for k in ds.data_vars][0]
    da = ds[vn]
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    return da


def load_sfc_norm_2d(var: str) -> xr.DataArray:
    """Load 2D normalized field -> DataArray(time, lat, lon)."""
    nc = DERIVED_DIR / f"era5_mjo_recon_{var}_norm_2d_1979-2022.nc"
    if not nc.exists():
        raise FileNotFoundError(f"Missing: {nc}")
    ds = xr.open_dataset(nc)
    vn = [k for k in ds.data_vars][0]
    return ds[vn]


def compute_event_means_2d(data_2d: np.ndarray, time_all, events: pd.DataFrame) -> np.ndarray:
    """Compute event mean for each event."""
    time_pd = pd.to_datetime(time_all)
    n_ev = len(events)
    spatial_shape = data_2d.shape[1:]
    event_means = np.full((n_ev, *spatial_shape), np.nan, dtype=np.float32)

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_pd >= pd.Timestamp(ev["start_date"])) & (time_pd <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        event_means[i] = np.nanmean(data_2d[mask], axis=0)

    return event_means


def _align_day_2d(
    data_day: np.ndarray,
    lons: np.ndarray,
    center_lon: float,
    rel_lons: np.ndarray,
    dlon: float,
) -> np.ndarray:
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


def compute_event_means_aligned(
    data_2d: np.ndarray,
    time_all,
    lons: np.ndarray,
    events: pd.DataFrame,
    center_lon_all: np.ndarray,
    time_step3: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray]:
    """Align each event to MJO center longitude, then compute event means."""
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
            c = center_lon_all[si_match[0]]
            if not np.isfinite(c):
                continue
            samples.append(_align_day_2d(data_2d[fi], lons, c, rel_lons, dlon))

        if samples:
            event_means[i] = np.nanmean(np.array(samples), axis=0)

    return event_means, rel_lons


def column_integrate_3d(data_4d: np.ndarray, levels_hpa: np.ndarray) -> np.ndarray:
    """Integrate (time, level, lat, lon) -> (time, lat, lon)."""
    sort_idx = np.argsort(levels_hpa)
    levels_pa = levels_hpa[sort_idx] * 100.0
    data_sorted = data_4d[:, sort_idx, :, :]
    return np.abs(np.trapz(data_sorted, x=levels_pa, axis=1)) / G


def split_speed_groups(phase_speed: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Split events into high/low groups by mean ± GROUP_SIGMA * std."""
    mu = float(np.nanmean(phase_speed))
    sigma = float(np.nanstd(phase_speed, ddof=1))
    high_thr = mu + GROUP_SIGMA * sigma
    low_thr = mu - GROUP_SIGMA * sigma
    high_mask = phase_speed > high_thr
    low_mask = phase_speed < low_thr

    stats_dict = {
        "mean_speed": mu,
        "std_speed": sigma,
        "high_threshold": high_thr,
        "low_threshold": low_thr,
        "n_high": int(np.sum(high_mask)),
        "n_low": int(np.sum(low_mask)),
    }
    return high_mask, low_mask, stats_dict


def welch_ttest_map(high_samples: np.ndarray, low_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute gridpoint Welch t-test between high and low groups."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_map, p_map = stats.ttest_ind(
            high_samples,
            low_samples,
            axis=0,
            equal_var=False,
            nan_policy="omit",
        )
    return t_map.astype(np.float32), p_map.astype(np.float32)


def _get_symmetric_levels(high_mean: np.ndarray, low_mean: np.ndarray) -> tuple[np.ndarray, TwoSlopeNorm]:
    vmax = np.nanmax(np.abs(np.stack([high_mean, low_mean])))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    levels = np.linspace(-vmax, vmax, 21)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    return levels, norm


def _get_diff_levels(diff_map: np.ndarray) -> tuple[np.ndarray, TwoSlopeNorm]:
    vmax = np.nanmax(np.abs(diff_map))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    levels = np.linspace(-vmax, vmax, 21)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    return levels, norm


def _stipple(ax, xvals: np.ndarray, yvals: np.ndarray, sig_mask: np.ndarray) -> None:
    iy, ix = np.where(sig_mask & np.isfinite(sig_mask))
    if len(iy) > 0:
        ax.scatter(xvals[ix], yvals[iy], c="black", s=18, marker=".", alpha=0.8, linewidths=0)


def plot_composite_pair(
    high_mean: np.ndarray,
    low_mean: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    sig_mask: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    n_high: int,
    n_low: int,
    xlabel: str,
) -> None:
    """Plot high/low composites side-by-side using a shared color scale."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), dpi=150, sharey=True)
    levels, norm = _get_symmetric_levels(high_mean, low_mean)

    fields = [
        (high_mean, f"High Speed (n={n_high})"),
        (low_mean, f"Low Speed (n={n_low})"),
    ]
    cf = None
    for ax, (field, panel_title) in zip(axes, fields):
        cf = ax.contourf(lons, lats, field, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
        _stipple(ax, lons, lats, sig_mask)
        ax.set_title(panel_title, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=11)
        ax.tick_params(labelsize=9, direction="in")
        ax.set_xlim(lons.min(), lons.max())
        ax.set_ylim(lats.min(), lats.max())
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    axes[0].set_ylabel("Latitude (°N)", fontsize=11)
    n_sig = int(np.sum(sig_mask))
    n_total = int(np.sum(np.isfinite(high_mean) & np.isfinite(low_mean)))
    pct = n_sig / n_total * 100 if n_total > 0 else 0.0
    fig.suptitle(f"{title}  Sig(p<{SIG_ALPHA:.2f}): {n_sig}/{n_total} ({pct:.1f}%)", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    cbar_ax = fig.add_axes([0.12, 0.06, 0.78, 0.03])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=10)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_diff_map(
    diff_map: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    sig_mask: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    xlabel: str,
) -> None:
    """Plot high-low difference map with significance stippling."""
    fig, ax = plt.subplots(figsize=(14, 4.8), dpi=150)
    levels, norm = _get_diff_levels(diff_map)
    cf = ax.contourf(lons, lats, diff_map, levels=levels, cmap="RdBu_r", norm=norm, extend="both")
    _stipple(ax, lons, lats, sig_mask)

    n_sig = int(np.sum(sig_mask))
    n_total = int(np.sum(np.isfinite(diff_map)))
    pct = n_sig / n_total * 100 if n_total > 0 else 0.0
    ax.set_title(f"{title}  Sig(p<{SIG_ALPHA:.2f}): {n_sig}/{n_total} ({pct:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.tick_params(labelsize=9, direction="in")
    ax.set_xlim(lons.min(), lons.max())
    ax.set_ylim(lats.min(), lats.max())
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    cbar_ax = fig.add_axes([0.12, 0.06, 0.78, 0.03])
    cbar = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=10)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def process_and_plot_grouped(
    data_2d: np.ndarray,
    time_all,
    lats: np.ndarray,
    lons: np.ndarray,
    events: pd.DataFrame,
    high_mask: np.ndarray,
    low_mask: np.ndarray,
    out_dir: Path,
    var_key: str,
    var_long: str,
    var_unit: str,
    group_label: str,
    summary_rows: list[dict],
    level_label: str | None = None,
    mode: str = "background",
    center_lon_all: np.ndarray | None = None,
    time_step3: pd.DatetimeIndex | None = None,
) -> None:
    """Generate grouped composite and diff plots for one field."""
    out_dir_mode = out_dir / f"{mode}_grouped"
    out_dir_mode.mkdir(parents=True, exist_ok=True)

    suffix = f"_{level_label}" if level_label else ""
    if mode == "background":
        event_means = compute_event_means_2d(data_2d, time_all, events)
        plot_lons = lons
        xlabel = "Longitude (°E)"
        title_prefix = "Background Composite"
    else:
        event_means, plot_lons = compute_event_means_aligned(
            data_2d, time_all, lons, events, center_lon_all, time_step3
        )
        xlabel = "Relative Longitude (°)"
        title_prefix = "Perturbation Composite"

    high_samples = event_means[high_mask]
    low_samples = event_means[low_mask]
    high_mean = np.nanmean(high_samples, axis=0)
    low_mean = np.nanmean(low_samples, axis=0)
    diff_map = high_mean - low_mean
    _, p_map = welch_ttest_map(high_samples, low_samples)
    sig_mask = (p_map < SIG_ALPHA) & np.isfinite(p_map)

    title_core = f"{title_prefix}: {var_long}{(' @ ' + level_label) if level_label else ''} | {group_label}"
    plot_composite_pair(
        high_mean,
        low_mean,
        lats,
        plot_lons,
        sig_mask,
        title_core,
        var_unit,
        out_dir_mode / f"composite_{var_key}{suffix}.png",
        int(np.sum(high_mask)),
        int(np.sum(low_mask)),
        xlabel,
    )
    plot_diff_map(
        diff_map,
        lats,
        plot_lons,
        sig_mask,
        f"High - Low: {var_long}{(' @ ' + level_label) if level_label else ''} | {group_label}",
        f"{var_unit} (High - Low)",
        out_dir_mode / f"diff_{var_key}{suffix}.png",
        xlabel,
    )

    n_sig = int(np.sum(sig_mask))
    n_total = int(np.sum(np.isfinite(diff_map)))
    summary_rows.append(
        {
            "var": var_key,
            "level": level_label or "surface",
            "mode": mode,
            "n_high": int(np.sum(high_mask)),
            "n_low": int(np.sum(low_mask)),
            "sig_points": n_sig,
            "total_points": n_total,
            "sig_pct": (n_sig / n_total * 100) if n_total else np.nan,
        }
    )


def main() -> None:
    print("=" * 72)
    print("07b: 2D lat×lon Grouped Composite Analysis (High vs Low Phase Speed)")
    print("=" * 72)

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)

    high_mask, low_mask, group_stats = split_speed_groups(phase_speed)
    group_label = (
        f"μ={group_stats['mean_speed']:.2f}, σ={group_stats['std_speed']:.2f}, "
        f"high>μ+{GROUP_SIGMA:.1f}σ ({group_stats['high_threshold']:.2f}), "
        f"low<μ-{GROUP_SIGMA:.1f}σ ({group_stats['low_threshold']:.2f})"
    )
    print(f"Events: {len(merged)}")
    print(group_label)
    print(f"High group: {group_stats['n_high']}")
    print(f"Low group : {group_stats['n_low']}")
    if group_stats["n_high"] < MIN_GROUP_SIZE or group_stats["n_low"] < MIN_GROUP_SIZE:
        raise ValueError(
            f"Group size too small for stable comparison: high={group_stats['n_high']}, "
            f"low={group_stats['n_low']}"
        )

    ds3 = xr.open_dataset(STEP3_NC)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    time_step3 = pd.to_datetime(ds3["time"].values)
    ds3.close()

    summary_rows: list[dict] = []
    n_plots = 0

    for var in PL_VARIABLES:
        print(f"\n{'=' * 50}\nPressure Level: {var}\n{'=' * 50}")
        try:
            da = load_pl_norm_3d(var)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue

        time_all = da["time"].values
        levels = da["level"].values
        lats = da["lat"].values
        lons = da["lon"].values
        data = da.values
        out_dir = FIG_ROOT / var

        for lev_idx, lev in enumerate(levels):
            lev_label = f"{int(lev)}hPa"
            print(f"  Level: {lev_label}")
            process_and_plot_grouped(
                data[:, lev_idx, :, :],
                time_all,
                lats,
                lons,
                merged,
                high_mask,
                low_mask,
                out_dir,
                var,
                PL_LONG[var],
                PL_UNIT[var],
                group_label,
                summary_rows,
                level_label=lev_label,
                mode="background",
            )
            process_and_plot_grouped(
                data[:, lev_idx, :, :],
                time_all,
                lats,
                lons,
                merged,
                high_mask,
                low_mask,
                out_dir,
                var,
                PL_LONG[var],
                PL_UNIT[var],
                group_label,
                summary_rows,
                level_label=lev_label,
                mode="perturbation",
                center_lon_all=center_lon_all,
                time_step3=time_step3,
            )
            n_plots += 4

        da.close()

    for var in SFC_VARIABLES:
        print(f"\n{'=' * 50}\nSurface: {var}\n{'=' * 50}")
        try:
            da = load_sfc_norm_2d(var)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
            continue

        time_all = da["time"].values
        lats = da["lat"].values
        lons = da["lon"].values
        data = da.values
        out_dir = FIG_ROOT / var

        process_and_plot_grouped(
            data,
            time_all,
            lats,
            lons,
            merged,
            high_mask,
            low_mask,
            out_dir,
            var,
            SFC_LONG[var],
            SFC_UNIT[var],
            group_label,
            summary_rows,
            mode="background",
        )
        process_and_plot_grouped(
            data,
            time_all,
            lats,
            lons,
            merged,
            high_mask,
            low_mask,
            out_dir,
            var,
            SFC_LONG[var],
            SFC_UNIT[var],
            group_label,
            summary_rows,
            mode="perturbation",
            center_lon_all=center_lon_all,
            time_step3=time_step3,
        )
        n_plots += 4
        da.close()

    print(f"\n{'=' * 50}\nColumn Integrated\n{'=' * 50}")
    try:
        da_q = load_pl_norm_3d("q")
        da_t = load_pl_norm_3d("t")

        levels_q = da_q["level"].values.astype(float)
        lats = da_q["lat"].values
        lons = da_q["lon"].values
        time_all = da_q["time"].values
        out_dir = FIG_ROOT / "column_integrated"

        print("  Computing CWV ...")
        cwv = column_integrate_3d(da_q.values, levels_q)
        process_and_plot_grouped(
            cwv,
            time_all,
            lats,
            lons,
            merged,
            high_mask,
            low_mask,
            out_dir,
            "cwv",
            "Column Water Vapor (∫q dp/g)",
            "kg/m² / amp",
            group_label,
            summary_rows,
            mode="background",
        )
        process_and_plot_grouped(
            cwv,
            time_all,
            lats,
            lons,
            merged,
            high_mask,
            low_mask,
            out_dir,
            "cwv",
            "Column Water Vapor (∫q dp/g)",
            "kg/m² / amp",
            group_label,
            summary_rows,
            mode="perturbation",
            center_lon_all=center_lon_all,
            time_step3=time_step3,
        )
        n_plots += 4

        print("  Computing Column MSE ...")
        mse_3d = CP * da_t.values + LV * da_q.values
        col_mse = column_integrate_3d(mse_3d, levels_q)
        process_and_plot_grouped(
            col_mse,
            time_all,
            lats,
            lons,
            merged,
            high_mask,
            low_mask,
            out_dir,
            "mse",
            "Column MSE (∫(CpT+Lvq) dp/g)",
            "J/m² / amp",
            group_label,
            summary_rows,
            mode="background",
        )
        process_and_plot_grouped(
            col_mse,
            time_all,
            lats,
            lons,
            merged,
            high_mask,
            low_mask,
            out_dir,
            "mse",
            "Column MSE (∫(CpT+Lvq) dp/g)",
            "J/m² / amp",
            group_label,
            summary_rows,
            mode="perturbation",
            center_lon_all=center_lon_all,
            time_step3=time_step3,
        )
        n_plots += 4

        da_q.close()
        da_t.close()
    except FileNotFoundError as exc:
        print(f"[SKIP] Column integrated variables skipped: {exc}")

    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 72}")
    print(f"Total plots generated: {n_plots}")
    print(f"Summary saved to: {SUMMARY_CSV}")
    print(f"Output directory: {FIG_ROOT}")
    print("=" * 72)


if __name__ == "__main__":
    main()
