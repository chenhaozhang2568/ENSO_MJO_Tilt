# -*- coding: utf-8 -*-
"""
boundary_diagnostics.py — MJO 上升运动区域边界诊断综合分析

功能：
    对 MJO 上升运动区域的边界位置进行多角度诊断，包括长尾统计、分布可视化、
    逐层零点边界密度、西边界外零散上升运动假说验证、以及多阈值边界包络图。
输入：
    mjo_mvEOF_step3_1979-2022.nc, mjo_events_step3_1979-2022.csv,
    tilt_daily_step4_layermean_1979-2022.nc, era5_mjo_recon_w_norm_1979-2022.nc
输出：
    多组诊断图表（checks_step4/, boundary_longtail_step4/, lon_height_daily_boundaries_step4/）
用法：
    python tests/boundary_diagnostics.py           # 全部 5 个分析
    python tests/boundary_diagnostics.py tails     # 长尾统计
    python tests/boundary_diagnostics.py longtail  # 分布可视化
    python tests/boundary_diagnostics.py scatter   # 密度散点
    python tests/boundary_diagnostics.py ascent    # 分散上升
    python tests/boundary_diagnostics.py envelope  # 包络图
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, MaxNLocator

# ======================
# PATHS
# ======================
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
TILT_NC    = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
W_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"

FIG_BASE    = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\boundary")
OUT_DIR_TAILS     = FIG_BASE / "checks"
OUT_DIR_LONGTAIL  = FIG_BASE / "longtail"
OUT_DIR_SCATTER   = FIG_BASE / "lon_height"
OUT_DIR_ASCENT    = FIG_BASE / "ascent"
TABLE_DIR         = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\tables")

# ======================
# SETTINGS
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
TRACK_LON_MIN = 0.0
TRACK_LON_MAX = 180.0

LOW_LAYER = (1000.0, 600.0)
UP_LAYER  = (400.0, 200.0)
LOW_LEVELS_HPA = [1000, 925, 850, 700, 600]
UP_LEVELS_HPA  = [400, 300, 200]

PIVOT_DELTA_DEG = 10.0
MIN_VALID_POINTS = 7
AMP_EPS = 1e-6
SMOOTH_WINDOW = 6
EDGE_N_CONSEC = 1
OLR_MIN_THRESH = -15.0

BEYOND_WEST_DEG = 60.0
FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TAIL_P_LO = 1.0
TAIL_P_HI = 99.0
BINS_HIST = np.arange(-120, 121, 2)

REL_LON_MIN, REL_LON_MAX, REL_LON_STEP = -120, 120, 1.0
LON_BIN_WIDTH = 2.0
BASE_SIZE = 10.0
SIZE_SCALE = 6.0
CORE_SEARCH_WINDOW = (-30, 30)


# ======================
# SHARED HELPERS
# ======================
def _to_0360(lon):
    return (np.asarray(lon, dtype=float) + 360.0) % 360.0


def _winter_mask(time_index):
    return np.isin(pd.to_datetime(time_index).month, list(WINTER_MONTHS)).astype(bool)


def _event_mask(time, csv):
    ev = pd.read_csv(csv, parse_dates=["start_date", "end_date"])
    m = np.zeros(len(time), dtype=bool)
    if ev.empty:
        return m
    tv = pd.to_datetime(time).values.astype("datetime64[ns]")
    for _, r in ev.iterrows():
        s = np.datetime64(pd.Timestamp(r["start_date"]).normalize().to_datetime64())
        e = np.datetime64(pd.Timestamp(r["end_date"]).normalize().to_datetime64())
        i0 = int(np.searchsorted(tv, s, side="left"))
        i1 = int(np.searchsorted(tv, e, side="right")) - 1
        if i1 >= i0:
            m[i0:i1 + 1] = True
    return m


def _pick_var(ds, candidates, required=True):
    for c in candidates:
        if c in ds.data_vars or c in ds.variables:
            return c
    if required:
        raise KeyError(f"None found: {candidates}")
    return None


def _pressure_to_height_km(p_hpa):
    return 44.331 * (1.0 - (np.asarray(p_hpa) / 1013.25) ** 0.19026)


def _find_boundary_half_max(rel_lon, w, frac):
    """半高宽法找边界。Returns (west, east, wmin)。"""
    m = np.isfinite(w) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return np.nan, np.nan, np.nan
    rr = rel_lon[m].astype(float)
    ww = w[m].astype(float)
    if SMOOTH_WINDOW > 1 and len(ww) >= SMOOTH_WINDOW:
        kernel = np.ones(SMOOTH_WINDOW) / SMOOTH_WINDOW
        ww = np.convolve(ww, kernel, mode="same")
    win = (rr >= -PIVOT_DELTA_DEG) & (rr <= PIVOT_DELTA_DEG)
    if win.any():
        pivot_idx = np.where(win)[0][int(np.nanargmin(ww[win]))]
    else:
        pivot_idx = int(np.nanargmin(ww))
    wmin = float(ww[pivot_idx])
    if (not np.isfinite(wmin)) or (wmin >= 0):
        return np.nan, np.nan, wmin
    thr = float(frac) * wmin
    # west
    outside = 0
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if ww[i] >= thr:
            outside += 1
        else:
            outside = 0
        if outside >= EDGE_N_CONSEC:
            west_idx = min(i + EDGE_N_CONSEC, pivot_idx)
            break
    if west_idx is None:
        west_idx = 0
    # east
    outside = 0
    east_idx = None
    for i in range(pivot_idx, len(ww)):
        if ww[i] >= thr:
            outside += 1
        else:
            outside = 0
        if outside >= EDGE_N_CONSEC:
            east_idx = max(i - EDGE_N_CONSEC, pivot_idx)
            break
    if east_idx is None:
        east_idx = len(ww) - 1
    west = float(rr[west_idx])
    east = float(rr[east_idx])
    if not (np.isfinite(west) and np.isfinite(east)) or (west > east):
        return np.nan, np.nan, wmin
    return west, east, wmin


def _find_zero_west_boundary(rel_lon, prof):
    """0% 西边界 (第一个 ω >= 0)。Returns (west, pivot_idx)。"""
    m = np.isfinite(prof) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return np.nan, -1
    win = (rel_lon >= -10) & (rel_lon <= 10)
    if win.any():
        pivot_idx = np.where(win)[0][int(np.nanargmin(prof[win]))]
    else:
        pivot_idx = int(np.nanargmin(prof))
    if prof[pivot_idx] >= 0:
        return np.nan, pivot_idx
    west_idx = None
    for i in range(pivot_idx, -1, -1):
        if prof[i] >= 0:
            west_idx = i + 1
            break
    if west_idx is None:
        west_idx = 0
    west_idx = min(west_idx, pivot_idx)
    return float(rel_lon[west_idx]), pivot_idx


def _zero_crossing(rel_lon, prof, direction, core_window=CORE_SEARCH_WINDOW):
    """west/east 零交叉。direction='west' or 'east'。"""
    ok = np.isfinite(prof) & (rel_lon >= core_window[0]) & (rel_lon <= core_window[1])
    if ok.sum() < 3:
        return np.nan
    idx = np.where(ok)[0]
    i0 = int(idx[np.nanargmin(prof[idx])])
    if direction == 'west':
        for j in range(i0, 0, -1):
            y2, y1 = prof[j], prof[j - 1]
            if np.isfinite(y1) and np.isfinite(y2) and y2 < 0 and y1 >= 0:
                frac = (0.0 - y1) / (y2 - y1) if (y2 - y1) != 0 else 0.0
                return float(rel_lon[j - 1] + frac * (rel_lon[j] - rel_lon[j - 1]))
    else:
        for j in range(i0, len(prof) - 1):
            y1, y2 = prof[j], prof[j + 1]
            if np.isfinite(y1) and np.isfinite(y2) and y1 < 0 and y2 >= 0:
                frac = (0.0 - y1) / (y2 - y1) if (y2 - y1) != 0 else 0.0
                return float(rel_lon[j] + frac * (rel_lon[j + 1] - rel_lon[j]))
    return np.nan


def _count_ascent_patches(prof_seg):
    in_asc = prof_seg < 0
    patches = 0
    was = False
    for v in in_asc:
        if v and not was:
            patches += 1
        was = v
    return patches


def _load_omega_2d():
    """加载 2D (lat-averaged) omega + step3。"""
    ds3 = xr.open_dataset(STEP3_NC).sel(time=slice(START_DATE, END_DATE))
    center = ds3["center_lon_track"].astype(float)
    amp = ds3["amp"].astype(float)
    dsw = xr.open_dataset(W_RECON_NC).sel(time=slice(START_DATE, END_DATE))
    w = dsw["w_mjo_recon_norm"]
    if "pressure_level" in w.dims:
        w = w.rename({"pressure_level": "level"})
    w = w.sel(lon=slice(TRACK_LON_MIN, TRACK_LON_MAX))
    return center, amp, w, ds3


# ============================================================
# 1. TAILS ANALYSIS (from analyze_boundary_tails.py)
# ============================================================
def run_tails():
    """边界长尾统计：直方图 + 散点 + 箱线图"""
    import seaborn as sns
    OUT_DIR_TAILS.mkdir(parents=True, exist_ok=True)
    print("\n[1] Boundary tail analysis...")
    ds = xr.open_dataset(TILT_NC)
    df = ds[["up_west_rel", "up_east_rel", "low_west_rel", "low_east_rel", "tilt"]].to_dataframe().dropna()
    print(f"  Valid samples: {len(df)}")

    plt.figure(figsize=(10, 6), dpi=150)
    sns.histplot(df["low_west_rel"], bins=50, kde=True, color="blue", label="Lower West")
    sns.histplot(df["up_west_rel"], bins=50, kde=True, color="red", label="Upper West")
    plt.title("West Boundary Comparison: Lower vs Upper")
    plt.xlabel("Relative Longitude (deg)")
    plt.legend()
    out = OUT_DIR_TAILS / "west_boundary_comparison_step4_1979-2022.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    plt.figure(figsize=(10, 6), dpi=150)
    sns.scatterplot(data=df, x="up_west_rel", y="tilt", alpha=0.4, s=15, color="purple")
    plt.title("Upper West vs Tilt")
    plt.xlabel("Upper West Relative Longitude (deg)")
    plt.ylabel("Tilt (deg)")
    out = OUT_DIR_TAILS / "upper_west_vs_tilt_scatter_step4_1979-2022.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    plt.figure(figsize=(10, 6), dpi=150)
    melted = df.melt(value_vars=["up_west_rel", "low_west_rel", "tilt"],
                     var_name="Variable", value_name="Value")
    sns.boxplot(data=melted, x="Variable", y="Value", palette="vlag")
    plt.title("Outliers Visualization")
    out = OUT_DIR_TAILS / "outlier_boxplots_step4_1979-2022.png"
    plt.savefig(out, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()

    stats = df[["up_west_rel", "low_west_rel", "tilt"]].describe(percentiles=[.01, .05, .5, .95, .99])
    print(stats)


# ============================================================
# 2. LONGTAIL VISUALIZATION (from visualize_boundary_longtail.py)
# ============================================================
def run_longtail():
    """边界分布 / ECDF / 散点 / 极值日"""
    OUT_DIR_LONGTAIL.mkdir(parents=True, exist_ok=True)
    print("\n[2] Boundary longtail visualization...")
    ds = xr.open_dataset(TILT_NC)

    v_lw = _pick_var(ds, ["low_west_rel", "low_west"])
    v_le = _pick_var(ds, ["low_east_rel", "low_east"])
    v_uw = _pick_var(ds, ["up_west_rel", "upper_west_rel", "up_west"])
    v_ue = _pick_var(ds, ["up_east_rel", "upper_east_rel", "up_east"])

    em_var = _pick_var(ds, ["eventmask", "event_mask"])
    em = ds[em_var].squeeze(drop=True)
    if em.dtype != bool:
        em = em.astype(float).where(np.isfinite(em.astype(float)), 0.0) > 0.5

    time_idx = pd.to_datetime(ds["time"].values)
    mask = pd.Series(em.values, index=time_idx).astype(bool)

    def _to_s(vn):
        da = ds[vn].squeeze(drop=True)
        extra = [d for d in da.dims if d != "time"]
        if extra:
            da = da.mean(extra, skipna=True)
        s = pd.Series(da.values, index=time_idx, name=vn)
        return s[mask]

    s_lw, s_le, s_uw, s_ue = _to_s(v_lw), _to_s(v_le), _to_s(v_uw), _to_s(v_ue)
    sd = {"LOW west": s_lw, "LOW east": s_le, "UP west": s_uw, "UP east": s_ue}

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    edges = np.asarray(BINS_HIST, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_fine = np.linspace(centers.min(), centers.max(), 1200)
    y_max = 0.0
    for name, s in sd.items():
        x = s.dropna().values
        if x.size == 0:
            continue
        density, _ = np.histogram(x, bins=edges, density=True)
        y_fine = np.interp(x_fine, centers, density.astype(float))
        ax.plot(x_fine, y_fine, lw=2, label=name)
        y_max = max(y_max, float(np.max(y_fine)))
    ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.set_xlabel("Boundary position (deg)")
    ax.set_ylabel("Density")
    ax.set_title("Boundary distributions")
    ax.axvline(0, lw=1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR_LONGTAIL / "boundary_hist.png", dpi=300)
    plt.close(fig)

    # ECDF
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, s in sd.items():
        x = np.sort(s.dropna().values)
        if x.size == 0:
            continue
        ax.plot(x, np.arange(1, x.size + 1) / x.size, lw=2, label=name)
    ax.set_xlabel("Boundary position (deg)")
    ax.set_ylabel("ECDF")
    ax.set_title("ECDF (inspect tails)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR_LONGTAIL / "boundary_ecdf.png", dpi=300)
    plt.close(fig)

    # Scatter
    for label, s1, s2 in [("WEST", s_lw, s_uw), ("EAST", s_le, s_ue)]:
        df_s = pd.concat([s1, s2], axis=1).dropna()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(df_s.iloc[:, 0], df_s.iloc[:, 1], s=10)
        ax.set_xlabel(df_s.columns[0])
        ax.set_ylabel(df_s.columns[1])
        ax.set_title(f"{label} boundary: low vs upper")
        ax.axvline(0, lw=1)
        ax.axhline(0, lw=1)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR_LONGTAIL / f"scatter_low_vs_up_{label.lower()}.png", dpi=300)
        plt.close(fig)

    # Tail time series
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, s in sd.items():
        x = s.dropna().values
        if x.size < 20:
            continue
        lo = np.nanpercentile(x, TAIL_P_LO)
        hi = np.nanpercentile(x, TAIL_P_HI)
        tail = s[(s <= lo) | (s >= hi)]
        ax.scatter(tail.index, tail.values, s=12, label=f"{name} tails")
    ax.set_ylabel("Boundary (deg)")
    ax.set_title("Extreme tail points over time")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR_LONGTAIL / "tail_points_time.png", dpi=300)
    plt.close(fig)

    print(f"  Saved 5 figures to {OUT_DIR_LONGTAIL}")
    for k, s in sd.items():
        x = s.dropna().values
        if x.size:
            print(f"  {k}: n={x.size}, p1={np.percentile(x,1):.1f}, p50={np.percentile(x,50):.1f}, p99={np.percentile(x,99):.1f}")


# ============================================================
# 3. SCATTER OMEGA BOUNDARIES (from scatter_omega_boundaries.py)
# ============================================================
def run_scatter():
    """逐层 omega 零点边界密度散点图"""
    OUT_DIR_SCATTER.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[3] Scatter omega boundaries...")

    rel_lon_grid = np.arange(REL_LON_MIN, REL_LON_MAX + 1e-9, REL_LON_STEP)

    ds_w = xr.open_dataset(W_RECON_NC)
    w_var = _pick_var(ds_w, ["w_mjo_recon_norm", "w_mjo_recon", "w"])
    w = ds_w[w_var]
    if "pressure_level" in w.dims:
        w = w.rename({"pressure_level": "level"})
    lon_w = _to_0360(w["lon"].values)
    w = w.assign_coords(lon=("lon", lon_w)).sortby("lon")
    lon_sorted = w["lon"].values.astype(float)

    ds3 = xr.open_dataset(STEP3_NC)
    center = ds3["center_lon_track"].squeeze(drop=True)
    w, center = xr.align(w, center, join="inner")
    times = pd.to_datetime(w["time"].values)

    mask = _event_mask(times, str(EVENTS_CSV)) & _winter_mask(times)
    sel_times = times[mask]
    print(f"  Selected days: {sel_times.size}")

    w_sel = w.sel(time=sel_times)
    c_sel = center.sel(time=sel_times)
    low_y = float(np.mean(LOW_LEVELS_HPA))
    up_y = float(np.mean(UP_LEVELS_HPA))

    def _shift(w_day, c0):
        abs_target = _to_0360(c0 + rel_lon_grid)
        da = w_day.assign_coords(lon=("lon", lon_sorted)).sortby("lon")
        lon_ext = np.concatenate([lon_sorted, lon_sorted + 360.0])
        da_ext = xr.concat([da, da.assign_coords(lon=da.lon + 360.0)], dim="lon")
        da_ext = da_ext.assign_coords(lon=("lon", lon_ext)).sortby("lon")
        ref = c0
        at2 = abs_target.copy()
        at2 = np.where(at2 < ref - 180, at2 + 360, at2)
        at2 = np.where(at2 > ref + 180, at2 - 360, at2)
        return da_ext.interp(lon=at2).assign_coords(rel_lon=("lon", rel_lon_grid))\
            .swap_dims({"lon": "rel_lon"}).drop_vars("lon")

    rows = []
    for t in sel_times:
        c0 = float(c_sel.sel(time=t).values)
        if not np.isfinite(c0):
            continue
        c0 = float(_to_0360(np.array([c0]))[0])
        w_day = w_sel.sel(time=t).squeeze(drop=True)
        w_rel = _shift(w_day, c0)
        low_prof = w_rel.sel(level=LOW_LEVELS_HPA, method="nearest").mean("level", skipna=True).values.astype(float)
        up_prof = w_rel.sel(level=UP_LEVELS_HPA, method="nearest").mean("level", skipna=True).values.astype(float)
        rows.append({
            "time": pd.to_datetime(t),
            "low_west_rel": _zero_crossing(rel_lon_grid, low_prof, 'west'),
            "low_east_rel": _zero_crossing(rel_lon_grid, low_prof, 'east'),
            "up_west_rel": _zero_crossing(rel_lon_grid, up_prof, 'west'),
            "up_east_rel": _zero_crossing(rel_lon_grid, up_prof, 'east'),
        })

    df = pd.DataFrame(rows).sort_values("time")
    csv_out = TABLE_DIR / "daily_zeroboundaries.csv"
    df.to_csv(csv_out, index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    for col, yv, marker, lab in [
        ("low_west_rel", low_y, "<", "LOW west"),
        ("low_east_rel", low_y, ">", "LOW east"),
        ("up_west_rel", up_y, "<", "UP west"),
        ("up_east_rel", up_y, ">", "UP east"),
    ]:
        valid = df[col].dropna()
        ax.scatter(valid.values, np.full(len(valid), yv), s=8, alpha=0.08, marker=marker)
        x_arr = df[col].dropna().values
        if x_arr.size == 0:
            continue
        bins = np.arange(REL_LON_MIN, REL_LON_MAX + LON_BIN_WIDTH, LON_BIN_WIDTH)
        counts, edges = np.histogram(x_arr, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        m = counts > 0
        sizes = BASE_SIZE + SIZE_SCALE * counts[m].astype(float)
        ax.scatter(centers[m], np.full(m.sum(), yv), s=sizes, marker=marker,
                   alpha=0.65, edgecolors="k", linewidths=0.3, label=lab)

    ax.axvline(0, lw=1)
    ax.set_xlim(REL_LON_MIN, REL_LON_MAX)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.grid(True, which="major", axis="x", alpha=0.25)
    ax.set_xlabel("Relative longitude (deg)")
    ax.set_ylabel("Pressure (hPa)")
    ax.invert_yaxis()
    ax.set_title("0-boundaries density scatter")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(0.02, 0.02, f"N={len(df)}", transform=ax.transAxes)
    fig.tight_layout()
    png_out = OUT_DIR_SCATTER / "daily_zeroboundaries_density.png"
    fig.savefig(png_out, dpi=300)
    plt.close(fig)
    print(f"  Saved: {png_out}, {csv_out}")


# ============================================================
# 4. SCATTERED ASCENT (from diag_scattered_ascent.py)
# ============================================================
def run_ascent():
    """西边界外零散上升运动假说验证"""
    OUT_DIR_ASCENT.mkdir(parents=True, exist_ok=True)
    print("\n[4] Scattered ascent analysis...")

    center, amp, w, ds3 = _load_omega_2d()
    w_low = w.sel(level=slice(LOW_LAYER[0], LOW_LAYER[1])).mean("level", skipna=True).transpose("time", "lon")
    w_up = w.sel(level=slice(UP_LAYER[0], UP_LAYER[1])).mean("level", skipna=True).transpose("time", "lon")
    center_a, amp_a, w_low_a, w_up_a = xr.align(center, amp, w_low, w_up, join="inner")

    time = pd.to_datetime(center_a["time"].values)
    winter = _winter_mask(time)
    eventmask = _event_mask(time, str(EVENTS_CSV))
    amp_np = amp_a.values.astype(float)
    amp_ok = np.isfinite(amp_np) & (amp_np > AMP_EPS)

    lon = w_low_a["lon"].values.astype(float)
    c_np = center_a.values.astype(float)
    w_low_np = w_low_a.values.astype(float)
    w_up_np = w_up_a.values.astype(float)
    dlon = float(lon[1] - lon[0])
    rel_grid = np.arange(-90, 90 + dlon / 2, dlon)

    composite_low, composite_up, records = [], [], []
    for i in range(time.size):
        if not (winter[i] and eventmask[i] and amp_ok[i]):
            continue
        c = c_np[i]
        if not np.isfinite(c):
            continue
        rel = lon - float(c)
        for layer_name, w_arr in [("low", w_low_np), ("up", w_up_np)]:
            prof = w_arr[i, :]
            west_bnd, pivot_idx = _find_zero_west_boundary(rel, prof)
            if not np.isfinite(west_bnd) or pivot_idx < 0:
                continue
            beyond_mask = (rel < west_bnd) & (rel >= west_bnd - BEYOND_WEST_DEG)
            beyond_prof = prof[beyond_mask]
            if len(beyond_prof) < 3:
                continue
            n_pts = len(beyond_prof)
            n_ascent = int((beyond_prof < 0).sum())
            records.append({
                "layer": layer_name,
                "frac_ascent": n_ascent / n_pts,
                "n_patches": _count_ascent_patches(beyond_prof),
            })
        composite_low.append(np.interp(rel_grid, rel, w_low_np[i, :], left=np.nan, right=np.nan))
        composite_up.append(np.interp(rel_grid, rel, w_up_np[i, :], left=np.nan, right=np.nan))

    df = pd.DataFrame(records)
    mean_low = np.nanmean(composite_low, axis=0)
    mean_up = np.nanmean(composite_up, axis=0)

    for layer in ["low", "up"]:
        sub = df[df["layer"] == layer]
        label = 'LOW (1000-600)' if layer == 'low' else 'UP (400-200)'
        print(f"  {label}: n={len(sub)}, ascent_frac_mean={sub['frac_ascent'].mean():.3f}, "
              f"patches_mean={sub['n_patches'].mean():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(rel_grid, mean_low * 100, "b-", lw=2, label="Low")
    ax.plot(rel_grid, mean_up * 100, "r-", lw=2, label="Upper")
    ax.axhline(0, color="gray", ls="-", lw=0.8)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlim(-80, 60)
    ax.set_xlabel("Relative longitude (°)")
    ax.set_ylabel("ω (×10⁻² Pa/s)")
    ax.set_title("Composite ω profile")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(10))

    ax = axes[1]
    for layer, color in [("low", "blue"), ("up", "red")]:
        sub = df[df["layer"] == layer]
        ax.hist(sub["frac_ascent"], bins=np.arange(0, 1.02, 0.04), alpha=0.6,
                color=color, label=f"{layer} (mean={sub['frac_ascent'].mean():.3f})", edgecolor="white")
    ax.set_xlabel("Ascending fraction beyond west boundary")
    ax.set_ylabel("Count")
    ax.set_title(f"Ascending fraction in {BEYOND_WEST_DEG}° west of boundary")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = OUT_DIR_ASCENT / "diag_scattered_ascent.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ============================================================
# 5. BOUNDARY ENVELOPE (from plot_boundary_envelope.py)
# ============================================================
def run_envelope():
    """多阈值边界包络图"""
    print("\n[5] Boundary envelope...")

    center, amp, w, ds3 = _load_omega_2d()
    olr_center = ds3["olr_center_track"].astype(float)
    center_a, olr_center_a, amp_a, w_a = xr.align(center, olr_center, amp, w, join="inner")
    time = pd.to_datetime(center_a["time"].values)
    winter = _winter_mask(time)
    active = (olr_center_a.values.astype(float) <= OLR_MIN_THRESH)
    eventmask = _event_mask(time, str(EVENTS_CSV))
    amp_np = amp_a.values.astype(float)
    amp_ok = np.isfinite(amp_np) & (amp_np > AMP_EPS)
    valid = winter & eventmask & amp_ok

    lon = w_a["lon"].values.astype(float)
    levels = w_a["level"].values.astype(float)
    c_np = center_a.values.astype(float)
    w_np = w_a.values.astype(float)

    n_lev = len(levels)
    n_frac = len(FRACTIONS)
    valid_idx = np.where(valid)[0]
    n_valid = len(valid_idx)
    print(f"  Valid days: {n_valid}, levels: {n_lev}, fractions: {n_frac}")

    boundaries = np.full((n_valid, n_lev, n_frac, 2), np.nan)
    for ii, day_idx in enumerate(valid_idx):
        c = c_np[day_idx]
        if not np.isfinite(c):
            continue
        rel = lon - float(c)
        for lev_i in range(n_lev):
            w_profile = w_np[day_idx, lev_i, :]
            for fi, frac in enumerate(FRACTIONS):
                west, east, _ = _find_boundary_half_max(rel, w_profile, frac)
                boundaries[ii, lev_i, fi, 0] = west
                boundaries[ii, lev_i, fi, 1] = east
        if (ii + 1) % 500 == 0:
            print(f"    {ii + 1} / {n_valid}")

    mean_bnd = np.nanmean(boundaries, axis=0)
    mean_west = mean_bnd[:, :, 0]
    mean_east = mean_bnd[:, :, 1]
    heights = _pressure_to_height_km(levels)

    cmap = plt.cm.YlOrRd
    colors = [cmap(0.15 + 0.75 * i / (n_frac - 1)) for i in range(n_frac)]

    fig, ax = plt.subplots(figsize=(10, 7))
    for fi in range(n_frac - 1, -1, -1):
        frac = FRACTIONS[fi]
        color = colors[fi]
        w_vals = mean_west[:, fi]
        e_vals = mean_east[:, fi]
        mask = np.isfinite(w_vals) & np.isfinite(e_vals)
        if mask.sum() >= 2:
            ax.fill_betweenx(heights[mask], w_vals[mask], e_vals[mask], alpha=0.20, color=color)
        if frac == 1.0:
            pivot = 0.5 * (w_vals + e_vals)
            m = np.isfinite(pivot)
            if m.sum() >= 2:
                ax.plot(pivot[m], heights[m], "o-", color=color, lw=2.5, label="100% (pivot)", markersize=5)
        else:
            for vals, label, marker in [(w_vals, f"West {frac*100:.0f}%", "s"),
                                         (e_vals, f"East {frac*100:.0f}%", "^")]:
                m = np.isfinite(vals)
                if m.sum() >= 2:
                    ax.plot(vals[m], heights[m], f"{marker}-", color=color, lw=1.8, label=label, markersize=4)

    ax.set_ylabel("Height (km)", fontsize=13)
    ax.set_ylim(heights.min() - 0.3, heights.max() + 0.3)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(heights)
    ax2.set_yticklabels([f"{int(p)} hPa" for p in levels], fontsize=9)
    ax2.set_ylabel("Pressure (hPa)", fontsize=11)
    ax.set_xlabel("Relative Longitude (°)", fontsize=13)
    ax.axvline(0, color="grey", ls="--", lw=0.8, alpha=0.6)

    low_h = _pressure_to_height_km(np.array([1000.0, 600.0]))
    up_h = _pressure_to_height_km(np.array([400.0, 200.0]))
    ax.axhspan(low_h[0], low_h[1], alpha=0.06, color="blue")
    ax.axhspan(up_h[0], up_h[1], alpha=0.06, color="red")

    ax.set_title(f"MJO Ascent Region Boundary Envelope\n({n_valid} valid days)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), fontsize=8, ncol=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_BASE / "boundary_envelope.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()


# ============================================================
# MAIN
# ============================================================
ANALYSES = {
    "tails": run_tails,
    "longtail": run_longtail,
    "scatter": run_scatter,
    "ascent": run_ascent,
    "envelope": run_envelope,
}


def main():
    print("=" * 70)
    print("Boundary Diagnostics")
    print("=" * 70)

    if len(sys.argv) > 1:
        name = sys.argv[1].lower()
        if name in ANALYSES:
            ANALYSES[name]()
        else:
            print(f"Unknown: {name}. Available: {list(ANALYSES.keys())}")
            sys.exit(1)
    else:
        for name, func in ANALYSES.items():
            func()

    print("\nDone!")


if __name__ == "__main__":
    main()
