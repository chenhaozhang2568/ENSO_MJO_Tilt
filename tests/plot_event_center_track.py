# -*- coding: utf-8 -*-
"""
plot_event_center_track.py — 逐事件 OLR 对流中心经度可视化

功能：
    方案 A：逐事件 Hovmöller 底图 + center_lon_track 散点 + 相速度拟合线（仅成功事件）
    方案 B：成功事件汇总多面板对比图（2×2 每页, x=事件内天数, y=经度）
输入：
    mjo_mvEOF_step3_1979-2022.nc, mjo_events_step3_1979-2022.csv
输出：
    outputs/figures/event_tracks/ 下的逐事件图和汇总图
用法：
    python tests/plot_event_center_track.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path
from scipy import stats
import math

# ======================
# PATHS
# ======================
STEP3_NC         = Path(r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc")
EVENTS_CSV       = Path(r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv")
OUT_DIR          = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\event_tracks")

# ======================
# SETTINGS
# ======================
LON_RANGE = (20, 220)          # 绘图经度范围 (0–360)
XTICK_LOCS = [20, 60, 100, 140, 180, 220]
XTICK_LABELS = ["20°E", "60°E", "100°E", "140°E", "180°", "140°W"]
OLR_CONTOUR_LEVEL = -15.0      # 对流增强等值线
MARGIN_DAYS = 0                # 事件前后多取 N 天作上下文

# 物理常数 (与 phase_speed_tilt_analysis.py 一致)
DEG_TO_M = 111320.0   # 赤道处1度经度 ≈ 111.32 km
DAY_TO_SEC = 86400.0  # 1天 = 86400秒

# ======================
# COLORMAP (与 plot_olr_hovmoller.py 一致)
# ======================
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
    """确保经度在 0–360 范围内。"""
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon.values + 360) % 360).sortby("lon")
    return ds


def _fit_phase_speed(days, lons):
    """
    对事件内 center_lon_track 做线性拟合，返回拟合信息。
    """
    valid = np.isfinite(lons)
    if valid.sum() < 5:
        return None
    d = days[valid].astype(float)
    l = lons[valid].astype(float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(d, l)
    speed_m_s = slope * DEG_TO_M / DAY_TO_SEC
    return {
        "slope_deg_day": slope,
        "speed_m_s": speed_m_s,
        "r2": r_value ** 2,
        "intercept": intercept,
    }


# ======================================================================
# 方案 A：逐事件 Hovmöller + center_lon_track 散点 + 拟合线
# ======================================================================
def plot_single_event(ds, df_row, out_dir):
    """
    为一个成功事件绘制 Hovmöller + 中心经度散点 + 相速度拟合线。
    """
    t0 = pd.Timestamp(df_row["start_date"])
    t1 = pd.Timestamp(df_row["end_date"])
    t0_ext = t0 - pd.Timedelta(days=MARGIN_DAYS)
    t1_ext = t1 + pd.Timedelta(days=MARGIN_DAYS)

    ds_sub = ds.sel(time=slice(str(t0_ext), str(t1_ext)))
    if ds_sub.sizes["time"] < 2:
        return

    # --- OLR 填色底图 ---
    olr = ds_sub["olr_recon"].sel(lon=slice(LON_RANGE[0], LON_RANGE[1]))
    lon = olr.lon.values
    time = olr.time.values

    fig, ax = plt.subplots(figsize=(9, max(5, len(time) * 0.16)))
    cmap, norm = _setup_colormap()
    levels = norm.boundaries
    cf = ax.contourf(lon, time, olr.values, levels=levels, cmap=cmap, norm=norm, extend="both")

    # -15 W/m² 等值线
    ax.contour(lon, time, olr.values, levels=[OLR_CONTOUR_LEVEL],
               colors=["steelblue"], linewidths=1.2, linestyles="--")

    # --- center_lon_track 散点（黑色圆点）---
    trk_center = (ds_sub["center_lon_track"].values.copy() + 360) % 360
    trk_time = ds_sub["time"].values

    valid_c = np.isfinite(trk_center)
    ax.scatter(trk_center[valid_c], trk_time[valid_c],
               c="black", s=28, zorder=5, label="center_lon (min-based)",
               edgecolors="white", linewidths=0.4)

    # --- 相速度拟合线（全事件日期）---
    ds_event = ds.sel(time=slice(str(t0), str(t1)))
    event_lons = (ds_event["center_lon_track"].values.copy() + 360) % 360
    event_times = ds_event["time"].values
    event_days = np.array([(pd.Timestamp(t) - t0).days for t in event_times], dtype=float)

    fit = _fit_phase_speed(event_days, event_lons)
    if fit is not None:
        d_line = np.array([event_days[0], event_days[-1]])
        lon_line = fit["intercept"] + fit["slope_deg_day"] * d_line
        t_line = [np.datetime64(t0 + pd.Timedelta(days=float(d))) for d in d_line]
        ax.plot(lon_line, t_line, color="lime", linewidth=2.5, zorder=6,
                label=f"Fit: {fit['speed_m_s']:.1f} m/s  (R²={fit['r2']:.2f})")

    # --- 事件边界标注 ---
    for t_edge in [t0, t1]:
        ax.axhline(y=np.datetime64(t_edge), color="gray", linewidth=1.0, linestyle=":")

    # --- 格式 ---
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(time) // 15)))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xticks(XTICK_LOCS)
    ax.set_xticklabels(XTICK_LABELS, fontsize=9)
    ax.set_xlim(LON_RANGE)
    ax.set_ylabel("Date")

    # 参考线 60°E, 180°
    ax.axvline(x=60, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(x=180, color="black", linewidth=0.6, alpha=0.5)

    # 标题
    eid = int(df_row["event_id"]) if "event_id" in df_row.index else "?"
    dur = int(df_row["duration_days"]) if "duration_days" in df_row.index else "?"
    ax.set_title(
        f"Event #{eid}:  {t0.strftime('%Y-%m-%d')} → {t1.strftime('%Y-%m-%d')}  "
        f"({dur}d,  {df_row.get('lon_start', '?'):.0f}° → {df_row.get('lon_end', '?'):.0f}°)",
        fontsize=10, fontweight="bold",
    )

    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, aspect=30, shrink=0.85)
    cbar.set_label("OLR anomaly (W m⁻²)", fontsize=8)

    plt.tight_layout()
    out_file = out_dir / f"event_{eid:03d}_hovmoller.png"
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out_file.name}")


# ======================================================================
# 方案 B：成功事件汇总多面板对比图 (2×2 每页)
# ======================================================================
def plot_all_events_panel(ds, df_events, out_dir):
    """
    多面板对比图：每页 2×2 = 4 个子图，仅成功事件。
    x=事件内天数, y=经度, 叠加拟合线。
    """
    n_events = len(df_events)
    if n_events == 0:
        print("[WARN] No events to plot in panel view.")
        return

    n_cols = 2
    n_rows_per_page = 2
    per_page = n_cols * n_rows_per_page  # = 4
    n_pages = math.ceil(n_events / per_page)

    for page in range(n_pages):
        start_idx = page * per_page
        end_idx = min(start_idx + per_page, n_events)
        n_this = end_idx - start_idx
        n_rows = math.ceil(n_this / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.5 * n_rows),
                                  squeeze=False, sharey=True)

        for i in range(n_this):
            ax = axes[i // n_cols, i % n_cols]
            row = df_events.iloc[start_idx + i]
            t0 = pd.Timestamp(row["start_date"])
            t1 = pd.Timestamp(row["end_date"])

            ds_sub = ds.sel(time=slice(str(t0), str(t1)))
            if ds_sub.sizes["time"] < 1:
                ax.set_visible(False)
                continue

            trk_center = (ds_sub["center_lon_track"].values.copy() + 360) % 360
            days = np.arange(len(trk_center))

            # center_lon_track 折线 + 散点
            valid_c = np.isfinite(trk_center)
            ax.plot(days[valid_c], trk_center[valid_c], color="black", linewidth=1.0)
            ax.scatter(days[valid_c], trk_center[valid_c], c="black", s=18, zorder=4)

            # 拟合线
            fit = _fit_phase_speed(days, trk_center)
            if fit is not None:
                d_line = np.linspace(days[0], days[-1], 50)
                lon_line = fit["intercept"] + fit["slope_deg_day"] * d_line
                ax.plot(d_line, lon_line, color="red", linewidth=1.8, linestyle="--",
                        label=f"{fit['speed_m_s']:.1f} m/s (R²={fit['r2']:.2f})")
                ax.legend(fontsize=7, loc="upper left", framealpha=0.7)

            # 参考线
            ax.axhline(y=60, color="gray", linewidth=0.4, linestyle=":")
            ax.axhline(y=180, color="gray", linewidth=0.4, linestyle=":")

            # 标题
            eid = int(row["event_id"])
            dur = int(row.get("duration_days", 0))
            ax.set_title(f"#{eid}  {t0.strftime('%Y/%m/%d')} → {t1.strftime('%m/%d')}  ({dur}d)",
                         fontsize=10, fontweight="bold")
            ax.set_ylim(20, 220)
            ax.set_yticks([60, 100, 140, 180])
            ax.set_yticklabels(["60°E", "100°E", "140°E", "180°"], fontsize=8)
            ax.set_xlabel("Day within event", fontsize=9)
            ax.tick_params(labelsize=8)

        # 隐藏多余子图
        for i in range(n_this, n_rows * n_cols):
            axes[i // n_cols, i % n_cols].set_visible(False)

        fig.supylabel("Longitude", fontsize=12)

        plt.tight_layout()
        suffix = f"_page{page + 1}" if n_pages > 1 else ""
        out_file = out_dir / f"all_events_center_track{suffix}.png"
        plt.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out_file.name}")


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 60)
    print("plot_event_center_track.py — 逐事件 OLR 中心经度可视化")
    print("=" * 60)

    # --- 1. Load ---
    if not STEP3_NC.exists():
        print(f"[ERROR] NC file not found: {STEP3_NC}")
        return
    ds = xr.open_dataset(STEP3_NC)
    ds = _to_lon360(ds)

    df_events = pd.read_csv(EVENTS_CSV) if EVENTS_CSV.exists() else pd.DataFrame()
    print(f"  成功事件: {len(df_events)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 2. 方案 A：逐事件 Hovmöller (仅成功事件) ---
    print("\n--- 方案 A：逐事件 Hovmöller + 中心经度散点 + 拟合线 ---")
    for _, row in df_events.iterrows():
        plot_single_event(ds, row, OUT_DIR)

    # --- 3. 方案 B：成功事件汇总面板 (2×2 每页) ---
    print("\n--- 方案 B：成功事件汇总多面板 (2×2) ---")
    plot_all_events_panel(ds, df_events, OUT_DIR)

    print(f"\n✅ 全部完成, 输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
