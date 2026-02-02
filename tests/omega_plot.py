# -*- coding: utf-8 -*-
"""
omega_plot.py: 垂直速度 (omega) 合成剖面图绘制

================================================================================
功能描述：
    本脚本绘制 MJO 活跃日期合成的纬向-垂直 omega'（滤波异常）剖面图，
    用于可视化 MJO 的垂直环流结构。

主要输出：
    1. 相对经度 vs 气压高度的 omega 填色图
    2. 上升区（负 omega）和下沉区（正 omega）分布
    3. 高低层上升区的西东边界框（用于定义倾斜）

坐标系统：
    - 横轴：相对于 MJO 对流中心的经度偏移（-90° 到 +90°）
    - 纵轴：气压高度（1000-200 hPa）

筛选条件：
    - 冬季月份（11-4月）
    - MJO 事件日
    - MJO 活跃日（OLR < -15 W/m²，可选）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

# ======================
# USER PATHS
# ======================
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"

# ERA5 processed pressure-level product
ERA5_W_LATMEAN = r"E:\Datas\ERA5\processed\pressure_level\era5_w_bp_latmean_1979-2022.nc"

# figure output
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = FIG_DIR / "omega_xsec_active_mean_1979-2022.png"

# ======================
# SETTINGS (match Step3)
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}

# active definition from Step3 track
OLR_MIN_THRESH = -15.0
ACTIVE_ONLY = True  # True: only days with olr_center_track <= thresh; False: ignore active filter

# --- layer definition (hPa) for boxes only ---
LOW_LAYER = (1000.0, 600.0)   # inclusive slice
UP_LAYER  = (400.0, 200.0)    # inclusive slice

# --- relative-lon domain for composite (deg) ---
REL_LON_RANGE = (-90.0, 90.0)    # show like paper

# ======================
# helpers
# ======================
def _winter_np(time_index: pd.DatetimeIndex) -> np.ndarray:
    return np.isin(time_index.month, list(WINTER_MONTHS)).astype(bool)

def _mask_event_days(time: pd.DatetimeIndex, events_csv: str) -> np.ndarray:
    ev = pd.read_csv(events_csv, parse_dates=["start_date", "end_date"])
    m = np.zeros(len(time), dtype=bool)
    if ev.empty:
        return m
    tv = time.values.astype("datetime64[ns]")
    for _, r in ev.iterrows():
        s = np.datetime64(pd.Timestamp(r["start_date"]).normalize().to_datetime64())
        e = np.datetime64(pd.Timestamp(r["end_date"]).normalize().to_datetime64())
        i0 = int(np.searchsorted(tv, s, side="left"))
        i1 = int(np.searchsorted(tv, e, side="right")) - 1
        if i1 >= i0:
            m[i0:i1+1] = True
    return m

def _to_0360(lon: np.ndarray) -> np.ndarray:
    """Convert lon to [0,360) without reordering."""
    return np.mod(lon, 360.0)

def _wrap_rel_lon(lon0360: np.ndarray, c0360: float) -> np.ndarray:
    """Return relative lon in [-180,180)."""
    return (lon0360 - c0360 + 180.0) % 360.0 - 180.0

def _infer_dlon(lon_sorted: np.ndarray) -> float:
    d = np.diff(lon_sorted)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0
    return float(np.median(d))

def _interp_to_rel_grid_1d(rel: np.ndarray, y: np.ndarray, rel_grid: np.ndarray) -> np.ndarray:
    """
    Interpolate y(rel) onto rel_grid.
    rel: [-180,180) but not necessarily sorted.
    y: same length.
    """
    m = np.isfinite(rel) & np.isfinite(y)
    if m.sum() < 4:
        return np.full(rel_grid.shape, np.nan, dtype=float)

    rr = rel[m].astype(float)
    yy = y[m].astype(float)

    idx = np.argsort(rr)
    rr = rr[idx]
    yy = yy[idx]

    rr_u, uidx = np.unique(rr, return_index=True)
    yy_u = yy[uidx]

    return np.interp(rel_grid, rr_u, yy_u, left=np.nan, right=np.nan)

def _contiguous_ascent_extent(rel_grid: np.ndarray, prof: np.ndarray, thr: float = 0.0) -> tuple[float, float]:
    """
    Define ascent where prof < thr (default 0). Take the contiguous region containing the minimum.
    Returns west_rel, east_rel (nan if not found).
    """
    m = np.isfinite(prof)
    if m.sum() < 5:
        return (np.nan, np.nan)

    prof2 = prof.copy()
    prof2[~m] = np.nan
    jmin = int(np.nanargmin(prof2))
    if not np.isfinite(prof2[jmin]):
        return (np.nan, np.nan)

    asc = np.isfinite(prof2) & (prof2 < float(thr))
    if not asc[jmin]:
        return (np.nan, np.nan)

    j0 = jmin
    while j0 - 1 >= 0 and asc[j0 - 1]:
        j0 -= 1
    j1 = jmin
    while j1 + 1 < asc.size and asc[j1 + 1]:
        j1 += 1

    return (float(rel_grid[j0]), float(rel_grid[j1]))

def _sel_lev_mask(lev_arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    a, b = float(lo), float(hi)
    mn, mx = (min(a, b), max(a, b))
    return (lev_arr >= mn) & (lev_arr <= mx)

# ======================
# main
# ======================
def main():
    # --- load Step3 (track) ---
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    if "center_lon_track" not in ds3:
        raise RuntimeError("Step3 nc missing variable: center_lon_track")
    if "olr_center_track" not in ds3:
        raise RuntimeError("Step3 nc missing variable: olr_center_track")

    center = ds3["center_lon_track"].astype(float)
    olr_center = ds3["olr_center_track"].astype(float)

    # --- load ERA5 w (latmean) ---
    dsw = xr.open_dataset(ERA5_W_LATMEAN, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    if "w_bp" not in dsw:
        raise RuntimeError("ERA5 w file missing variable: w_bp")
    if not set(["time", "level", "lon"]).issubset(set(dsw["w_bp"].dims)):
        raise RuntimeError(f"w_bp dims must include time/level/lon, got: {dsw['w_bp'].dims}")

    w = dsw["w_bp"]

    # --- ensure lon is 0..360 increasing ---
    lon_raw = w["lon"].values.astype(float)
    lon0360 = _to_0360(lon_raw)
    sort_idx = np.argsort(lon0360)
    lon0360_sorted = lon0360[sort_idx]
    w = w.isel(lon=xr.DataArray(sort_idx, dims="lon")).assign_coords(lon=lon0360_sorted)

    # --- align time with Step3 (inner intersection) ---
    center_a, olr_center_a, w_a = xr.align(center, olr_center, w, join="inner")
    time = pd.to_datetime(center_a["time"].values)

    winter = _winter_np(time)
    active = (olr_center_a.values.astype(float) <= OLR_MIN_THRESH) & np.isfinite(olr_center_a.values.astype(float))
    eventmask = _mask_event_days(time, EVENTS_CSV)

    # build rel_lon grid based on lon spacing
    lon_sorted = w_a["lon"].values.astype(float)
    dlon = _infer_dlon(lon_sorted)
    rel_grid = np.arange(REL_LON_RANGE[0], REL_LON_RANGE[1] + 0.5 * dlon, dlon).astype(float)

    # arrays for speed
    c_np = _to_0360(center_a.values.astype(float))  # (time,)
    w_np = w_a.transpose("time", "level", "lon").values.astype(float)  # (time, level, lon)
    lev = w_a["level"].values.astype(float)

    # composite accumulator for cross-section (level x rel_lon)
    comp_sum = np.zeros((lev.size, rel_grid.size), dtype=np.float64)
    comp_cnt = np.zeros((lev.size, rel_grid.size), dtype=np.int32)

    ntime = time.size
    used_days = 0

    for i in range(ntime):
        if not winter[i]:
            continue
        if not eventmask[i]:
            continue
        if ACTIVE_ONLY and (not active[i]):
            continue

        c = c_np[i]
        if not np.isfinite(c):
            continue

        rel = _wrap_rel_lon(lon_sorted, float(c))

        # interpolate each level onto rel_grid and accumulate
        day_w = w_np[i, :, :]  # (level, lon)
        day_used = False
        for k in range(lev.size):
            wk = _interp_to_rel_grid_1d(rel, day_w[k, :], rel_grid)
            mk = np.isfinite(wk)
            if mk.any():
                comp_sum[k, mk] += wk[mk]
                comp_cnt[k, mk] += 1
                day_used = True
        if day_used:
            used_days += 1

    comp_mean = np.full_like(comp_sum, np.nan, dtype=np.float64)
    m = comp_cnt > 0
    comp_mean[m] = comp_sum[m] / comp_cnt[m]

    # ---- derive box extents from composite (layer-mean profs) ----
    low_mask = _sel_lev_mask(lev, LOW_LAYER[0], LOW_LAYER[1])
    up_mask  = _sel_lev_mask(lev, UP_LAYER[0],  UP_LAYER[1])

    low_prof = np.nanmean(comp_mean[low_mask, :], axis=0) if low_mask.any() else np.full(rel_grid.shape, np.nan)
    up_prof  = np.nanmean(comp_mean[up_mask,  :], axis=0) if up_mask.any()  else np.full(rel_grid.shape, np.nan)

    low_w, low_e = _contiguous_ascent_extent(rel_grid, low_prof, thr=0.0)
    up_w,  up_e  = _contiguous_ascent_extent(rel_grid, up_prof,  thr=0.0)

    if np.isfinite(low_e) and np.isfinite(up_e):
        print("GreenBoxEast - YellowBoxEast (deg):", float(low_e - up_e))
        print("low_e, up_e:", float(low_e), float(up_e))
    else:
        print("GreenBoxEast - YellowBoxEast (deg): NaN (box edge not found)")

    # ----------------------
    # plot composite cross-section (Fig b-like)
    # ----------------------
    fig = plt.figure(figsize=(8.6, 5.4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    # pressure axis: 1000 bottom -> 200 top (descending pressure)
    order = np.argsort(lev)[::-1]
    lev_plot = lev[order]
    comp_plot = comp_mean[order, :]

    # smooth (paper-like)
    comp_plot_s = gaussian_filter(comp_plot, sigma=(1.5, 2.0), mode="nearest")

    ax.set_xlabel("Relative longitude to convective center (deg)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Composite omega' zonal-vertical cross section (active event days)")
    ax.set_xlim(REL_LON_RANGE[0], REL_LON_RANGE[1])
    ax.set_ylim(np.nanmax(lev_plot), np.nanmin(lev_plot))

    # color levels: 0.005 Pa/s per step
    STEP = 0.005
    absmax = np.nanmax(np.abs(comp_plot_s[np.isfinite(comp_plot_s)])) if np.isfinite(comp_plot_s).any() else 0.04
    if not np.isfinite(absmax) or absmax <= 0:
        absmax = 0.04
    vmax = np.ceil(absmax / STEP) * STEP
    levels = np.arange(-vmax, vmax + 0.5 * STEP, STEP)

    # negative -> blue, positive -> red
    cmap = plt.get_cmap("RdBu_r")

    cf = ax.contourf(rel_grid, lev_plot, comp_plot_s, levels=levels, cmap=cmap, extend="both")
    cs = ax.contour(rel_grid, lev_plot, comp_plot_s, levels=levels, colors="k", linewidths=0.35)
    ax.contour(rel_grid, lev_plot, comp_plot_s, levels=[0.0], colors="k", linewidths=1.2)

    # label contours every 0.01 (avoid clutter)
    # Use actual cs.levels to avoid floating-point mismatch
    if hasattr(cs, 'levels') and len(cs.levels) > 0:
        label_levels = [lv for lv in cs.levels if abs(lv) >= 0.01 - 1e-9]
        ax.clabel(cs, levels=label_levels, inline=True, fontsize=7, fmt="%.3f")

    cbar = fig.colorbar(cf, ax=ax, pad=0.02, ticks=np.arange(-vmax, vmax + 1e-9, 0.01))
    cbar.set_label("Omega' (Pa s$^{-1}$)")

    ax.axvline(0.0, linewidth=0.8)

    # boxes (paper-like): green = lower, yellow = upper
    if np.isfinite(low_w) and np.isfinite(low_e):
        p_top = min(LOW_LAYER); p_bot = max(LOW_LAYER)
        ax.add_patch(Rectangle((low_w, p_top), low_e - low_w, p_bot - p_top,
                               fill=False, linewidth=2.0, edgecolor="limegreen"))
    if np.isfinite(up_w) and np.isfinite(up_e):
        p_top = min(UP_LAYER); p_bot = max(UP_LAYER)
        ax.add_patch(Rectangle((up_w, p_top), up_e - up_w, p_bot - p_top,
                               fill=False, linewidth=2.0, edgecolor="gold"))

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=200)
    plt.close(fig)

    # summary
    print("Saved figure:", str(OUT_FIG))
    print("Used days:", int(used_days))
    print("Filter stats:",
          "winter=", int(winter.sum()),
          "event=", int(eventmask.sum()),
          "active=", int(active.sum()),
          "winter&event=", int(np.sum(winter & eventmask)),
          "winter&event&active=", int(np.sum(winter & eventmask & active)))

if __name__ == "__main__":
    main()
