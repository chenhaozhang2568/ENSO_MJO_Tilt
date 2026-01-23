# -*- coding: utf-8 -*-
"""
Lon–Height scatter of DAILY layer west/east 0-boundaries (event days only),
with marker size indicating point density (overlap).

You asked:
- NOT a w cross-section shading plot.
- For EACH event day, compute 4 boundary values:
    (LOW west, LOW east, UP west, UP east)
  where boundary is defined as the w'=0 crossing on each side of the ascent core.
- Plot on a Lon (x) – Height/Pressure (y) figure.
- Loop all event days; points overlap; show "data amount" via marker size (density).

Inputs:
1) ERA5 omega bandpass lat-mean: w_bp(time, level, lon)  (Pa/s)
   (lat-mean over 15S–15N already done in your pipeline)
2) Step3 center track: center_lon_track(time)
3) Event CSV: start_date, end_date (success events by default)

Outputs:
- PNG: lon-height boundary cloud with density-sized markers
- CSV: daily boundaries for all event days used

Run:
python plot_daily_layer_boundaries_lon_height_density.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# -----------------------------
# USER PATHS (EDIT)
# -----------------------------
# ERA5 processed pressure-level product
W_BP_NC = r"E:\Datas\ERA5\processed\pressure_level\era5_w_bp_latmean_1979-2022.nc"
# Step3 + events are derived products
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
FAILED_EVENTS_CSV = r"E:\Datas\Derived\mjo_failed_events_step3_1979-2022.csv"
INCLUDE_FAILED = False

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\lon_height_daily_boundaries_step4_1979-2022")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = FIG_DIR / "daily_zeroboundaries_lon_height_density_step4_1979-2022.png"
OUT_CSV = TABLE_DIR / "daily_zeroboundaries_eventdays_step4_1979-2022.csv"



# -----------------------------
# SETTINGS
# -----------------------------
WINTER_ONLY = True  # Nov–Apr
REL_LON_MIN, REL_LON_MAX, REL_LON_STEP = -120, 120, 1.0

# Define LOW/UP layers by pressure levels (hPa).
# Use the levels you actually use in your tilt definition.
LOW_LEVELS_HPA = [1000, 925,850,700,600]           # example
UP_LEVELS_HPA  = [400,300, 200]      # example

# Core search window around center to locate ascent core (min omega)
CORE_SEARCH_WINDOW = (-30, 30)  # deg rel_lon

# For density sizing
LON_BIN_WIDTH = 2.0    # deg
BASE_SIZE = 10.0       # base marker size for binned/density points
SIZE_SCALE = 6.0       # additional size per count

# Plot x-limits (often you want wider than 60–180; here it's rel_lon)
XLIM = (REL_LON_MIN, REL_LON_MAX)


# -----------------------------
# HELPERS
# -----------------------------
def to_0360(lon: np.ndarray) -> np.ndarray:
    return (lon + 360.0) % 360.0

def is_winter_time(tindex: pd.DatetimeIndex) -> np.ndarray:
    m = tindex.month
    return (m >= 11) | (m <= 4)

def build_eventmask_from_csv(times: pd.DatetimeIndex, csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    mask = np.zeros(times.size, dtype=bool)
    for _, r in df.iterrows():
        m = (times >= r["start_date"]) & (times <= r["end_date"])
        mask |= m
    return mask

def pick_var(ds: xr.Dataset, candidates: list[str]) -> str:
    for c in candidates:
        if c in ds.data_vars:
            return c
    raise KeyError(f"Cannot find variable. Candidates={candidates}, available={list(ds.data_vars)}")

def select_levels_nearest(da: xr.DataArray, target_hpa: list[float]) -> xr.DataArray:
    """Select nearest pressure levels in 'level' coordinate."""
    # ERA5 sometimes uses 'level' in hPa, sometimes Pa; assume hPa here.
    # If your file is in Pa, convert target list accordingly before selecting.
    return da.sel(level=target_hpa, method="nearest")

def shift_day_to_rel_lon(
    w_day: xr.DataArray,          # dims: (level, lon)
    lon_sorted_0360: np.ndarray,  # 0..360 sorted
    center_lon_0360: float,
    rel_lon_grid: np.ndarray,
) -> xr.DataArray:
    """
    Shift (level, lon) to (level, rel_lon) by interpolating on a periodic lon axis.
    """
    # absolute targets
    abs_target = to_0360(center_lon_0360 + rel_lon_grid)

    # ensure sorted lon
    da = w_day.assign_coords(lon=("lon", lon_sorted_0360)).sortby("lon")

    # periodic extension
    lon_ext = np.concatenate([lon_sorted_0360, lon_sorted_0360 + 360.0])
    da_ext = xr.concat([da, da.assign_coords(lon=da.lon + 360.0)], dim="lon")
    da_ext = da_ext.assign_coords(lon=("lon", lon_ext)).sortby("lon")

    # unwrap target around reference center
    ref = center_lon_0360
    abs_target2 = abs_target.copy()
    abs_target2 = np.where(abs_target2 < ref - 180.0, abs_target2 + 360.0, abs_target2)
    abs_target2 = np.where(abs_target2 > ref + 180.0, abs_target2 - 360.0, abs_target2)

    out = da_ext.interp(lon=abs_target2)
    out = out.assign_coords(rel_lon=("lon", rel_lon_grid)).swap_dims({"lon": "rel_lon"}).drop_vars("lon")
    return out  # (level, rel_lon)

def find_core_min_index(rel_lon: np.ndarray, prof: np.ndarray, core_window=(-30, 30)) -> int | None:
    ok = np.isfinite(prof) & (rel_lon >= core_window[0]) & (rel_lon <= core_window[1])
    if ok.sum() < 3:
        return None
    idx = np.where(ok)[0]
    return int(idx[np.nanargmin(prof[idx])])

def west_zero_boundary(rel_lon: np.ndarray, prof: np.ndarray, core_window=(-30, 30)) -> float:
    """West 0-crossing from the ascent core (min omega)."""
    i0 = find_core_min_index(rel_lon, prof, core_window=core_window)
    if i0 is None:
        return np.nan
    for j in range(i0, 0, -1):
        y2 = prof[j]
        y1 = prof[j - 1]
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue
        # crossing between j-1 (west) and j (east): y1>=0, y2<0
        if (y2 < 0.0) and (y1 >= 0.0):
            x1, x2 = rel_lon[j - 1], rel_lon[j]
            frac = (0.0 - y1) / (y2 - y1) if (y2 - y1) != 0 else 0.0
            return float(x1 + frac * (x2 - x1))
        if y2 == 0.0:
            return float(rel_lon[j])
    return np.nan

def east_zero_boundary(rel_lon: np.ndarray, prof: np.ndarray, core_window=(-30, 30)) -> float:
    """East 0-crossing from the ascent core (min omega)."""
    i0 = find_core_min_index(rel_lon, prof, core_window=core_window)
    if i0 is None:
        return np.nan
    for j in range(i0, len(rel_lon) - 1):
        y1 = prof[j]
        y2 = prof[j + 1]
        if not (np.isfinite(y1) and np.isfinite(y2)):
            continue
        # crossing between j (west) and j+1 (east): y1<0, y2>=0
        if (y1 < 0.0) and (y2 >= 0.0):
            x1, x2 = rel_lon[j], rel_lon[j + 1]
            frac = (0.0 - y1) / (y2 - y1) if (y2 - y1) != 0 else 0.0
            return float(x1 + frac * (x2 - x1))
        if y1 == 0.0:
            return float(rel_lon[j])
    return np.nan

def binned_density_points(x: np.ndarray, y_value: float, bin_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin x along longitude and return:
    - x_bin_centers
    - y (same length)
    - counts per bin
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    bins = np.arange(REL_LON_MIN, REL_LON_MAX + bin_width, bin_width)
    counts, edges = np.histogram(x, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    m = counts > 0
    return centers[m], np.full(m.sum(), y_value), counts[m]


# -----------------------------
# MAIN
# -----------------------------
def main():
    rel_lon_grid = np.arange(REL_LON_MIN, REL_LON_MAX + 1e-9, REL_LON_STEP)

    # Load omega
    ds_w = xr.open_dataset(W_BP_NC)
    w_var = pick_var(ds_w, ["w_bp", "omega_bp", "w", "omega", "w_anom"])
    w = ds_w[w_var]
    if not all(d in w.dims for d in ["time", "level", "lon"]):
        raise ValueError(f"{w_var} must have dims (time, level, lon). Got {w.dims}")

    # lon to 0..360 sorted
    lon_w = w["lon"].values.astype(float)
    lon_w = to_0360(lon_w)
    w = w.assign_coords(lon=("lon", lon_w)).sortby("lon")
    lon_sorted = w["lon"].values.astype(float)

    # Load center track
    ds3 = xr.open_dataset(STEP3_NC)
    if "center_lon_track" not in ds3:
        raise KeyError(f"center_lon_track not found. Vars={list(ds3.data_vars)}")
    center = ds3["center_lon_track"].squeeze(drop=True)
    if "time" not in center.dims:
        raise ValueError("center_lon_track must have dim 'time'")

    # Align time
    w, center = xr.align(w, center, join="inner")
    times = pd.to_datetime(w["time"].values)

    # event days mask
    mask = build_eventmask_from_csv(times, EVENTS_CSV)
    if INCLUDE_FAILED and Path(FAILED_EVENTS_CSV).exists():
        mask |= build_eventmask_from_csv(times, FAILED_EVENTS_CSV)

    if WINTER_ONLY:
        mask &= is_winter_time(times)

    sel_times = times[mask]
    if sel_times.size < 10:
        raise RuntimeError(f"Too few selected days: {sel_times.size}")

    # Pre-select layers (nearest levels)
    # We'll compute daily layer-mean profile using these levels.
    w_sel = w.sel(time=sel_times)
    c_sel = center.sel(time=sel_times)

    # Representative y positions for plotting (layer mean pressure)
    low_y = float(np.mean(LOW_LEVELS_HPA))
    up_y = float(np.mean(UP_LEVELS_HPA))

    rows = []
    for t in sel_times:
        c0 = float(c_sel.sel(time=t).values)
        if not np.isfinite(c0):
            continue
        c0 = float(to_0360(np.array([c0]))[0])

        w_day = w_sel.sel(time=t).squeeze(drop=True)  # (level, lon)
        w_rel = shift_day_to_rel_lon(w_day, lon_sorted, c0, rel_lon_grid)  # (level, rel_lon)

        # layer-mean profiles (rel_lon,)
        low_prof = select_levels_nearest(w_rel, LOW_LEVELS_HPA).mean("level", skipna=True).values.astype(float)
        up_prof  = select_levels_nearest(w_rel, UP_LEVELS_HPA).mean("level", skipna=True).values.astype(float)

        # 0-boundaries
        low_w = west_zero_boundary(rel_lon_grid, low_prof, core_window=CORE_SEARCH_WINDOW)
        low_e = east_zero_boundary(rel_lon_grid, low_prof, core_window=CORE_SEARCH_WINDOW)
        up_w  = west_zero_boundary(rel_lon_grid, up_prof,  core_window=CORE_SEARCH_WINDOW)
        up_e  = east_zero_boundary(rel_lon_grid, up_prof,  core_window=CORE_SEARCH_WINDOW)

        rows.append({
            "time": pd.to_datetime(t),
            "low_west_rel": low_w,
            "low_east_rel": low_e,
            "up_west_rel": up_w,
            "up_east_rel": up_e,
            "low_layer_hPa": low_y,
            "up_layer_hPa": up_y,
        })

    df = pd.DataFrame(rows).sort_values("time")
    if df.empty:
        raise RuntimeError("No valid days produced boundaries (check center track and omega data).")
    df.to_csv(OUT_CSV, index=False)

    # -----------------------------
    # PLOT: Lon (rel) vs Height (pressure)
    #  - faint individual points
    #  - binned density markers with size ~ counts
    # -----------------------------
    fig, ax = plt.subplots(figsize=(11, 6))

    # Individual points (low alpha, show spread)
    alpha_pt = 0.08
    s_pt = 8

    # LOW
    ax.scatter(df["low_west_rel"], df["low_layer_hPa"], s=s_pt, alpha=alpha_pt, marker="<", label="LOW west (indiv)")
    ax.scatter(df["low_east_rel"], df["low_layer_hPa"], s=s_pt, alpha=alpha_pt, marker=">", label="LOW east (indiv)")
    # UP
    ax.scatter(df["up_west_rel"], df["up_layer_hPa"],  s=s_pt, alpha=alpha_pt, marker="<", label="UP  west (indiv)")
    ax.scatter(df["up_east_rel"], df["up_layer_hPa"],  s=s_pt, alpha=alpha_pt, marker=">", label="UP  east (indiv)")

    # Binned density points (size encodes count)
    for col, yv, marker, lab in [
        ("low_west_rel", low_y, "<", "LOW west (density)"),
        ("low_east_rel", low_y, ">", "LOW east (density)"),
        ("up_west_rel",  up_y,  "<", "UP  west (density)"),
        ("up_east_rel",  up_y,  ">", "UP  east (density)"),
    ]:
        xb, yb, cnt = binned_density_points(df[col].to_numpy(dtype=float), yv, LON_BIN_WIDTH)
        if xb.size == 0:
            continue
        sizes = BASE_SIZE + SIZE_SCALE * cnt.astype(float)
        ax.scatter(xb, yb, s=sizes, marker=marker, alpha=0.65, edgecolors="k", linewidths=0.3, label=lab)

    # formatting
    ax.axvline(0, linewidth=1.0)
    ax.set_xlim(*XLIM)
    # --- x-axis ticks & grid (denser longitude graticule) ---
    ax.xaxis.set_major_locator(MultipleLocator(10))  # 主刻度每 10°
    ax.xaxis.set_minor_locator(MultipleLocator(5))   # 次刻度每 5°
    ax.grid(True, which="major", axis="x", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.12)

    ax.set_xlabel("Longitude relative to convective center (deg)")
    ax.set_ylabel("Pressure (hPa)")
    ax.invert_yaxis()
    ax.set_title("Event days: daily 0-boundaries (LOW/UP west/east) with density-sized markers")

    # Make legend compact (many entries)
    handles, labels = ax.get_legend_handles_labels()
    # Keep only density labels (optional). Comment out if you want everything.
    keep = [i for i, s in enumerate(labels) if "density" in s]
    if keep:
        handles = [handles[i] for i in keep]
        labels = [labels[i] for i in keep]
    ax.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.88),   # 1.0=右边不变，0.88=往下移（0.88可再调小）
        frameon=True,
        fontsize=9,
        title=f"Bin={LON_BIN_WIDTH}° (size~count)"
    )


    # annotate counts
    ax.text(0.02, 0.02, f"N days = {len(df)}", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    print(f"Saved figure: {OUT_PNG}")
    print(f"Saved data:   {OUT_CSV}")
    print("Note: y positions are layer-mean pressures; points overlap; marker size shows density per lon bin.")

if __name__ == "__main__":
    main()
