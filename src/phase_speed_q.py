# -*- coding: utf-8 -*-
"""
phase_speed_q.py: MJO 相速度计算（新定义）

================================================================================
算法：
  对于事件内的每个经度 lon：
    1. 取该经度上事件期间所有日期的 OLR → olr(t)
    2. 找 OLR 最小值日 t_min，记录 olr_min
    3. 阈值 = 0.5 * olr_min（注意 OLR 异常为负值）
    4. 从 t_min 向前后找 olr(t) > threshold 的日 → [t_start, t_end]
    5. 标记 [t_start, t_end] 之间所有日为活跃点

  收集所有活跃 (lon, day_index) → 二次拟合 lon = a*t^2 + b*t + c
  相速度 = 事件中点处的导数 dlon/dt

输出：
  - CSV：逐事件相速度统计
  - 逐事件 Hovmoller 图（OLR 底图 + 中心点 + 50%范围 + 拟合线）
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
import math

# ======================
# PATHS
# ======================
STEP3_NC   = Path(r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc")
EVENTS_CSV = Path(r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv")
OUT_DIR    = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\phase_speed_q")
OUT_CSV    = Path(r"E:\Datas\Derived\phase_speed_q_events.csv")

# ======================
# SETTINGS
# ======================
LON_RANGE = (20, 220)
XTICK_LOCS = [20, 60, 100, 140, 180, 220]
XTICK_LABELS = ["20E", "60E", "100E", "140E", "180", "140W"]
OLR_CONTOUR_LEVEL = -15.0
HALF_MAX_FRAC = 0.5       # 50% 强度阈值
MIN_POINTS_FIT = 10       # 拟合最少标记点数
LON_STEP = 2.5            # 数据经度分辨率

DEG_TO_M = 111320.0
DAY_TO_SEC = 86400.0


def _setup_colormap():
    boundaries = [-75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65]
    n_colors = len(boundaries) - 1
    base_cmap = plt.cm.RdBu_r
    colors = []
    for i in range(n_colors):
        mid = (boundaries[i] + boundaries[i + 1]) / 2.0
        if -5 <= mid <= 5:
            colors.append("white")
        else:
            norm_val = (mid + 75) / (65 + 75)
            colors.append(base_cmap(norm_val))
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def _to_lon360(ds):
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon.values + 360) % 360).sortby("lon")
    return ds


def compute_event_phase_speed(olr_event, lon_arr, day_indices):
    """
    对单个事件计算相速度。

    Parameters:
        olr_event: (n_days, n_lon) OLR 重构场
        lon_arr: (n_lon,) 经度数组 (0-360)
        day_indices: (n_days,) 事件内天数索引 (0, 1, 2, ...)

    Returns:
        dict with phase_speed info, or None if failed
        center_points: list of (lon, day_idx) for center points
        active_points: list of (lon, day_idx) for 50% range points
    """
    n_days, n_lon = olr_event.shape
    center_points = []   # (lon, day_idx)
    active_points = []   # (lon, day_idx)

    for j in range(n_lon):
        olr_col = olr_event[:, j]  # OLR at this lon across all event days

        # skip if all NaN or all positive
        if np.all(~np.isfinite(olr_col)) or np.nanmin(olr_col) >= 0:
            continue

        # find minimum (most negative = strongest convection)
        valid = np.isfinite(olr_col)
        if valid.sum() < 3:
            continue

        min_idx = int(np.nanargmin(olr_col))
        olr_min = float(olr_col[min_idx])

        if olr_min >= 0:
            continue

        center_points.append((float(lon_arr[j]), float(day_indices[min_idx])))

        # threshold = 50% of minimum (less negative)
        threshold = HALF_MAX_FRAC * olr_min

        # search backward from min_idx
        t_start = min_idx
        for t in range(min_idx - 1, -1, -1):
            if not np.isfinite(olr_col[t]) or olr_col[t] > threshold:
                break
            t_start = t

        # search forward from min_idx
        t_end = min_idx
        for t in range(min_idx + 1, n_days):
            if not np.isfinite(olr_col[t]) or olr_col[t] > threshold:
                break
            t_end = t

        # mark active points
        for t in range(t_start, t_end + 1):
            active_points.append((float(lon_arr[j]), float(day_indices[t])))

    if len(active_points) < MIN_POINTS_FIT:
        return None, center_points, active_points

    # Quadratic fit: lon = a*t^2 + b*t + c
    pts = np.array(active_points)
    t_pts = pts[:, 1]
    lon_pts = pts[:, 0]

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(t_pts, lon_pts)
    except Exception:
        return None, center_points, active_points

    r2 = r_value ** 2
    dlon_dt = slope  # deg/day
    speed_m_s = dlon_dt * DEG_TO_M / DAY_TO_SEC

    return {
        "slope": slope,
        "intercept": intercept,
        "dlon_dt": dlon_dt,
        "speed_m_s": speed_m_s,
        "r2": r2,
        "n_points": len(active_points),
        "t_range": (float(t_pts.min()), float(t_pts.max())),
    }, center_points, active_points


def plot_event(ds, df_row, fit_result, center_pts, active_pts, out_dir):
    """绘制单个事件的 Hovmoller 图 + 中心点 + 50%范围 + 拟合线"""
    t0 = pd.Timestamp(df_row["start_date"])
    t1 = pd.Timestamp(df_row["end_date"])

    ds_sub = ds.sel(time=slice(str(t0), str(t1)))
    if ds_sub.sizes["time"] < 2:
        return

    olr = ds_sub["olr_recon"].sel(lon=slice(LON_RANGE[0], LON_RANGE[1]))
    lon = olr.lon.values
    time_vals = olr.time.values

    fig, ax = plt.subplots(figsize=(9, max(5, len(time_vals) * 0.16)))
    cmap, norm = _setup_colormap()
    levels = norm.boundaries
    cf = ax.contourf(lon, time_vals, olr.values, levels=levels,
                     cmap=cmap, norm=norm, extend="both")

    ax.contour(lon, time_vals, olr.values, levels=[OLR_CONTOUR_LEVEL],
               colors=["steelblue"], linewidths=1.2, linestyles="--")

    # 50% active range (scatter, semi-transparent)
    if active_pts:
        act = np.array(active_pts)
        act_times = [np.datetime64(t0 + pd.Timedelta(days=float(d))) for d in act[:, 1]]
        ax.scatter(act[:, 0], act_times, c="cyan", s=12, alpha=0.4,
                   zorder=4, label="50% range", marker="s", edgecolors="none")

    # center points (OLR min per lon)
    if center_pts:
        ctr = np.array(center_pts)
        ctr_times = [np.datetime64(t0 + pd.Timedelta(days=float(d))) for d in ctr[:, 1]]
        ax.scatter(ctr[:, 0], ctr_times, c="black", s=28, zorder=5,
                   label="OLR min (center)", edgecolors="white", linewidths=0.4)

    # quadratic fit curve
    if fit_result is not None:
        t_line = np.array([fit_result["t_range"][0], fit_result["t_range"][1]])
        lon_line = fit_result["intercept"] + fit_result["slope"] * t_line
        time_line = [np.datetime64(t0 + pd.Timedelta(days=float(d))) for d in t_line]
        ax.plot(lon_line, time_line, color="lime", linewidth=2.5, zorder=6,
                label=f"Fit: {fit_result['speed_m_s']:.1f} m/s (R2={fit_result['r2']:.2f})")

    # event boundaries
    for t_edge in [t0, t1]:
        ax.axhline(y=np.datetime64(t_edge), color="gray", linewidth=1.0, linestyle=":")

    # formatting
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(time_vals) // 15)))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xticks(XTICK_LOCS)
    ax.set_xticklabels(XTICK_LABELS, fontsize=9)
    ax.set_xlim(LON_RANGE)
    ax.set_ylabel("Date")

    ax.axvline(x=60, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(x=180, color="black", linewidth=0.6, alpha=0.5)

    eid = int(df_row["event_id"])
    dur = int(df_row["duration_days"])
    speed_str = f"{fit_result['speed_m_s']:.1f} m/s" if fit_result else "N/A"
    ax.set_title(
        f"Event #{eid}: {t0.strftime('%Y-%m-%d')} - {t1.strftime('%Y-%m-%d')} "
        f"({dur}d)  |  Phase speed: {speed_str}",
        fontsize=10, fontweight="bold",
    )

    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, aspect=30, shrink=0.85)
    cbar.set_label("OLR anomaly (W/m2)", fontsize=8)

    plt.tight_layout()
    out_file = out_dir / f"event_{eid:03d}_phase_speed.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_file.name}")


def main():
    print("=" * 60)
    print("phase_speed_q.py - MJO Phase Speed (new definition)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(STEP3_NC)
    ds = _to_lon360(ds)
    df_events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    print(f"  Events: {len(df_events)}")

    all_time = pd.to_datetime(ds["time"].values)
    olr_full = ds["olr_recon"]
    lon_full = olr_full.lon.values

    results = []

    for idx, row in df_events.iterrows():
        eid = int(row["event_id"])
        t0 = pd.Timestamp(row["start_date"])
        t1 = pd.Timestamp(row["end_date"])

        # select event time window
        ds_event = olr_full.sel(time=slice(str(t0), str(t1)))
        if ds_event.sizes["time"] < 3:
            print(f"  Event {eid}: too short, skip")
            results.append({"event_id": eid, "phase_speed_m_s": np.nan})
            continue

        # select lon range for this event (use lon_start - margin to lon_end + margin)
        lon_start = max(float(row["lon_start"]) - 10, LON_RANGE[0])
        lon_end = min(float(row["lon_end"]) + 10, LON_RANGE[1])
        ds_event_lon = ds_event.sel(lon=slice(lon_start, lon_end))

        olr_arr = ds_event_lon.values  # (n_days, n_lon)
        lon_arr = ds_event_lon.lon.values
        day_indices = np.arange(olr_arr.shape[0], dtype=float)

        fit, center_pts, active_pts = compute_event_phase_speed(
            olr_arr, lon_arr, day_indices
        )

        if fit is not None:
            speed = fit["speed_m_s"]
            r2 = fit["r2"]
            n_pts = fit["n_points"]
            print(f"  Event {eid}: speed={speed:.2f} m/s, R2={r2:.3f}, "
                  f"N_pts={n_pts}, centers={len(center_pts)}")
        else:
            speed = np.nan
            r2 = np.nan
            n_pts = len(active_pts)
            print(f"  Event {eid}: fit failed (N_active={n_pts})")

        results.append({
            "event_id": eid,
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "duration_days": row["duration_days"],
            "phase_speed_deg_day": fit["dlon_dt"] if fit else np.nan,
            "phase_speed_m_s": speed,
            "r2": r2,
            "n_active_points": n_pts,
            "n_center_points": len(center_pts),
        })

        # plot
        plot_event(ds, row, fit, center_pts, active_pts, OUT_DIR)

    # save CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}")

    # summary
    valid = df_out["phase_speed_m_s"].dropna()
    print(f"\n{'='*50}")
    print(f"SUMMARY: {len(valid)}/{len(df_out)} events with valid fits")
    if len(valid) > 0:
        print(f"  Phase speed: mean={valid.mean():.2f} m/s, "
              f"median={valid.median():.2f} m/s, "
              f"std={valid.std():.2f} m/s")
        print(f"  Range: [{valid.min():.2f}, {valid.max():.2f}] m/s")
    print(f"All figures: {OUT_DIR}")


if __name__ == "__main__":
    main()
