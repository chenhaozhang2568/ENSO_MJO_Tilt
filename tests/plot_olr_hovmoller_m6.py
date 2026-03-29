# -*- coding: utf-8 -*-
"""
plot_olr_hovmoller_m6.py — OLR Hovmöller 时-经度图 (M6相速度拟合线)

基于 plot_olr_hovmoller.py 修改：
  - 趋势线改用 M6 方法（逐经度50%范围线性拟合）
  - 移除红色失败事件线
  - 输出到 hovmoller_m6/ 子文件夹

用法：
    python tests/plot_olr_hovmoller_m6.py
"""

import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from compare_phase_speed_methods import (
    method6_lon_halfmax_lsq, _to_lon360, LON_RANGE as CALC_LON_RANGE
)

# Input/Output paths
STEP3_NC = Path(r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc")
EVENTS_CSV = Path(r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv")
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\hovmoller_m6")

# Settings
START_YEAR = 1979
END_YEAR = 2022
LON_RANGE = (20, 220)
XTICK_LOCS = [20, 60, 100, 140, 180, 220]
XTICK_LABELS = ["20E", "60E", "100E", "140E", "180", "140W"]
CONTOUR_LEVEL = -15.0
FIT_LON_RANGE = (60, 180)


def setup_colormap():
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


def get_winter_window(year):
    start_date = f"{year - 1}-11-01"
    end_date = f"{year}-04-30"
    return start_date, end_date


def plot_m6_trend_lines(ax, ds_full, df_events, win_start, win_end):
    """为每个事件用M6方法计算拟合线并绘制。"""
    mask = (df_events["start_date"] <= win_end) & (df_events["end_date"] >= win_start)
    events_in_window = df_events[mask]

    for _, event in events_in_window.iterrows():
        t0 = max(event["start_date"], win_start)
        t1 = min(event["end_date"], win_end)
        if t0 >= t1:
            continue

        # 选取事件OLR数据
        lon_start = max(float(event["lon_start"]) - 10, CALC_LON_RANGE[0])
        lon_end = min(float(event["lon_end"]) + 10, CALC_LON_RANGE[1])
        ds_event = ds_full["olr_recon"].sel(
            time=slice(str(t0), str(t1)),
            lon=slice(lon_start, lon_end)
        )
        if ds_event.sizes["time"] < 3:
            continue

        olr_arr = ds_event.values
        lon_arr = ds_event.lon.values
        day_indices = np.arange(olr_arr.shape[0], dtype=float)

        # M6 计算
        r6 = method6_lon_halfmax_lsq(olr_arr, lon_arr, day_indices)

        if r6.get("slope") is None or np.isnan(r6.get("speed_m_s", np.nan)):
            continue

        slope = r6["slope"]
        intercept = r6["intercept"]

        # 画拟合线: 在 [60, 180] 范围内
        t_range = np.linspace(day_indices[0], day_indices[-1], 100)
        lon_line = intercept + slope * t_range

        # 转换为日期
        time_line = [t0 + pd.Timedelta(days=float(d)) for d in t_range]

        # 限制经度范围 [60, 180]
        lon_clipped = lon_line.copy()
        for k in range(len(lon_clipped)):
            if lon_clipped[k] < FIT_LON_RANGE[0] or lon_clipped[k] > FIT_LON_RANGE[1]:
                lon_clipped[k] = np.nan

        if np.all(~np.isfinite(lon_clipped)):
            continue

        ax.plot(lon_clipped, time_line, color="black", linewidth=2.0)


def plot_hovmoller(ds_sub, ds_full, df_events, year, output_path):
    data = ds_sub["olr_recon"].sel(lon=slice(LON_RANGE[0], LON_RANGE[1]))
    lon = data.lon.values
    time_vals = data.time.values

    fig, ax = plt.subplots(figsize=(8, 10))
    cmap, norm = setup_colormap()
    levels = norm.boundaries
    cf = ax.contourf(lon, time_vals, data.values, levels=levels,
                     cmap=cmap, norm=norm, extend="both")
    ax.contour(lon, time_vals, data.values, levels=[CONTOUR_LEVEL],
               colors=["blue"], linewidths=1.5)

    # M6 趋势线
    win_start = pd.to_datetime(time_vals[0])
    win_end = pd.to_datetime(time_vals[-1])
    plot_m6_trend_lines(ax, ds_full, df_events, win_start, win_end)

    # 格式
    y_year_start = int(year) - 1
    y_year_end = int(year)
    ticks_dates = [
        pd.Timestamp(f"{y_year_start}-11-01"),
        pd.Timestamp(f"{y_year_start}-12-01"),
        pd.Timestamp(f"{y_year_end}-01-01"),
        pd.Timestamp(f"{y_year_end}-02-01"),
        pd.Timestamp(f"{y_year_end}-03-01"),
        pd.Timestamp(f"{y_year_end}-04-01"),
        pd.Timestamp(f"{y_year_end}-04-30"),
    ]
    ax.set_yticks(ticks_dates)
    ax.set_yticklabels([d.strftime("%b%d").upper() for d in ticks_dates])
    ax.set_xticks(XTICK_LOCS)
    ax.set_xticklabels(XTICK_LABELS)
    ax.set_xlabel("Longitude (degE)")
    ax.set_ylabel("Time")
    ax.set_xlim(LON_RANGE)
    ax.set_ylim(ticks_dates[0], ticks_dates[-1])
    ax.axvline(x=60, color="black", linewidth=1.0)
    ax.axvline(x=180, color="black", linewidth=1.0)

    ax.text(0.02, 1.02, "(a)", transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom")
    ax.text(0.5, 1.02, "OLR", transform=ax.transAxes, fontsize=14,
            fontweight="bold", ha="center", va="bottom")
    ax.text(0.98, 1.02, str(year), transform=ax.transAxes, fontsize=14,
            fontweight="bold", ha="right", va="bottom")

    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, aspect=30)
    cbar.set_label("OLR anomalies (W m$^{-2}$)")

    plt.tight_layout()
    out_file = output_path / f"hovmoller_olr_recon_{year}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {out_file}")


def main():
    print("=" * 60)
    print("plot_olr_hovmoller_m6.py")
    print("Hovmoller图 — M6 (LonHalfMax-LSQ) 趋势线")
    print("=" * 60)

    ds = xr.open_dataset(STEP3_NC)
    if float(ds.lon.min()) < 0:
        ds.coords["lon"] = (ds.coords["lon"] + 360) % 360
        ds = ds.sortby("lon")

    df_events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(START_YEAR, END_YEAR + 1):
        s_date, e_date = get_winter_window(year)
        try:
            ds_win = ds.sel(time=slice(s_date, e_date))
            if ds_win.sizes["time"] < 10:
                print(f"Skipping {year}: insufficient data")
                continue
            plot_hovmoller(ds_win, ds, df_events, year, OUT_DIR)
        except Exception as e:
            print(f"Error {year}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
