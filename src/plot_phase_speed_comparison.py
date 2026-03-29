# -*- coding: utf-8 -*-
"""
plot_phase_speed_comparison.py: 六种MJO相速度计算方法对比可视化

生成图表：
  1. 六种方法各事件相速度分布图
  2. Event18 Hovmoller对比图（六条拟合线叠加）
  3. ONI相关性对比图
  4. 拟合优度R²对比箱线图
  5. 综合评估汇总表

输出目录: E:/Projects/ENSO_MJO_Tilt/outputs/figures/phase_speed_comparison/
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from compare_phase_speed_methods import (
    method1_daily_diff, method2_lon_diff, method3_daily_center_lsq,
    method4_lon_center_lsq, method5_daily_halfmax_lsq, method6_lon_halfmax_lsq,
    _to_lon360, _get_lon_centers, LON_RANGE, DAY_TO_SEC, DEG_TO_M
)

# ======================
# PATHS
# ======================
STEP3_NC   = Path(r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc")
EVENTS_CSV = Path(r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv")
SPEED_CSV  = Path(r"E:\Datas\Derived\phase_speed_6methods.csv")
ONI_TXT    = Path(r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt")
OUT_DIR    = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\phase_speed_comparison")

# ======================
# SETTINGS
# ======================
METHOD_NAMES = {
    "speed_m1": "M1: DailyDiff",
    "speed_m2": "M2: LonDiff",
    "speed_m3": "M3: DailyCenter-LSQ",
    "speed_m4": "M4: LonCenter-LSQ",
    "speed_m5": "M5: DailyHalfMax-LSQ",
    "speed_m6": "M6: LonHalfMax-LSQ",
}
METHOD_COLORS = {
    "speed_m1": "#E74C3C",
    "speed_m2": "#E67E22",
    "speed_m3": "#3498DB",
    "speed_m4": "#2ECC71",
    "speed_m5": "#9B59B6",
    "speed_m6": "#F1C40F",
}
XTICK_LOCS = [20, 60, 100, 140, 180, 220]
XTICK_LABELS = ["20E", "60E", "100E", "140E", "180", "140W"]
OLR_CONTOUR_LEVEL = -15.0
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}
ONI_ELNINO = 0.5
ONI_LANINA = -0.5


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


def parse_oni(path):
    seas_map = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
    }
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        mon = seas_map.get(parts[0])
        if mon is None:
            continue
        data.append({"time": pd.Timestamp(year=int(parts[1]), month=mon, day=1),
                      "oni": float(parts[3])})
    df = pd.DataFrame(data).set_index("time").sort_index()
    return df


def classify_event(start_date, end_date, oni_df):
    s = pd.Timestamp(start_date).replace(day=1)
    e = pd.Timestamp(end_date).replace(day=1) + pd.offsets.MonthEnd(0)
    mask = (oni_df.index >= s) & (oni_df.index <= e)
    sub = oni_df.loc[mask]
    if sub.empty:
        idx = oni_df.index.get_indexer([pd.Timestamp(start_date)], method="nearest")
        if idx[0] == -1:
            return np.nan, "Unknown"
        val = oni_df.iloc[idx[0]]["oni"]
    else:
        val = sub["oni"].mean()
    if val >= ONI_ELNINO:
        cat = "El Nino"
    elif val <= ONI_LANINA:
        cat = "La Nina"
    else:
        cat = "Neutral"
    return val, cat


# =====================================================================
# 图1: 六种方法各事件相速度分布图
# =====================================================================
def plot_distribution(df, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    speed_cols = list(METHOD_NAMES.keys())

    for i, col in enumerate(speed_cols):
        ax = axes[i]
        vals = df[col].dropna()
        color = METHOD_COLORS[col]

        ax.hist(vals, bins=15, color=color, alpha=0.7, edgecolor="white", linewidth=0.8)

        mean_v = vals.mean()
        med_v = vals.median()
        std_v = vals.std()

        ax.axvline(mean_v, color="navy", linestyle="--", linewidth=2, label=f"Mean: {mean_v:.2f}")
        ax.axvline(med_v, color="darkred", linestyle=":", linewidth=2, label=f"Median: {med_v:.2f}")

        stats_text = (f"Mean: {mean_v:.2f} m/s\n"
                      f"Median: {med_v:.2f} m/s\n"
                      f"Std: {std_v:.2f} m/s\n"
                      f"Min: {vals.min():.2f} m/s\n"
                      f"Max: {vals.max():.2f} m/s")
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
                va="top", ha="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray"))

        ax.set_title(f"{METHOD_NAMES[col]}  (N={len(vals)})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Phase Speed (m/s)", fontsize=10)
        ax.set_ylabel("Count (events)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)

    fig.suptitle("Phase Speed Distribution: 6 Methods Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "phase_speed_distribution_6methods.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =====================================================================
# 图2: Event18 Hovmoller 对比图
# =====================================================================
def plot_event18_hovmoller(df, out_dir):
    ds = xr.open_dataset(STEP3_NC)
    ds = _to_lon360(ds)
    df_events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])

    ev_row = df_events[df_events["event_id"] == 18].iloc[0]
    t0 = pd.Timestamp(ev_row["start_date"])
    t1 = pd.Timestamp(ev_row["end_date"])

    ds_sub = ds.sel(time=slice(str(t0), str(t1)))
    olr = ds_sub["olr_recon"].sel(lon=slice(LON_RANGE[0], LON_RANGE[1]))
    lon = olr.lon.values
    time_vals = olr.time.values

    lon_start = max(float(ev_row["lon_start"]) - 10, LON_RANGE[0])
    lon_end = min(float(ev_row["lon_end"]) + 10, LON_RANGE[1])
    ds_event_lon = ds_sub["olr_recon"].sel(lon=slice(lon_start, lon_end))
    olr_arr = ds_event_lon.values
    lon_arr_ev = ds_event_lon.lon.values
    day_indices = np.arange(olr_arr.shape[0], dtype=float)

    center_track = ds["center_lon_track"].values.astype(float)
    center_track = (center_track + 360) % 360
    time_index = pd.to_datetime(ds["time"].values)
    time_mask = (time_index >= t0) & (time_index <= t1)
    daily_center = center_track[time_mask]
    daily_days = np.arange(len(daily_center), dtype=float)

    # 计算各方法
    r1 = method1_daily_diff(daily_center, daily_days)
    r2 = method2_lon_diff(olr_arr, lon_arr_ev, day_indices)
    r3 = method3_daily_center_lsq(daily_center, daily_days)
    r4 = method4_lon_center_lsq(olr_arr, lon_arr_ev, day_indices)
    r5 = method5_daily_halfmax_lsq(olr_arr, lon_arr_ev, day_indices)
    r6 = method6_lon_halfmax_lsq(olr_arr, lon_arr_ev, day_indices)

    # 逐经度中心点
    lon_centers_lon, lon_centers_t = _get_lon_centers(olr_arr, lon_arr_ev, day_indices)

    fig, ax = plt.subplots(figsize=(12, max(6, len(time_vals) * 0.18)))
    cmap, norm = _setup_colormap()
    levels = norm.boundaries
    cf = ax.contourf(lon, time_vals, olr.values, levels=levels,
                     cmap=cmap, norm=norm, extend="both")
    ax.contour(lon, time_vals, olr.values, levels=[OLR_CONTOUR_LEVEL],
               colors=["steelblue"], linewidths=1.2, linestyles="--")

    # M6 active points scatter
    if r6.get("active_points"):
        act = np.array(r6["active_points"])
        act_times = [np.datetime64(t0 + pd.Timedelta(days=float(d))) for d in act[:, 1]]
        ax.scatter(act[:, 0], act_times, c="gold", s=8, alpha=0.2,
                   zorder=3, marker="s", edgecolors="none", label="M6 active range")

    # M5 active points scatter
    if r5.get("active_points"):
        act5 = np.array(r5["active_points"])
        act5_times = [np.datetime64(t0 + pd.Timedelta(days=float(d))) for d in act5[:, 1]]
        ax.scatter(act5[:, 0], act5_times, c="plum", s=8, alpha=0.2,
                   zorder=3, marker="s", edgecolors="none", label="M5 active range")

    # M2/M4/M6 逐经度中心点
    if len(lon_centers_lon) > 0:
        lc_times = [np.datetime64(t0 + pd.Timedelta(days=float(d))) for d in lon_centers_t]
        ax.scatter(lon_centers_lon, lc_times, c="black", s=20, zorder=5,
                   label="M2/M4/M6 center (lon OLR min)", edgecolors="white", linewidths=0.3)

    # M1/M3/M5 逐日中心点
    valid_dc = np.isfinite(daily_center)
    dc_times = [np.datetime64(t0 + pd.Timedelta(days=float(d))) for d in daily_days[valid_dc]]
    ax.scatter(daily_center[valid_dc], dc_times, c="gray", s=20, zorder=5,
               label="M1/M3/M5 center (daily OLR min)", edgecolors="white",
               linewidths=0.3, marker="^")

    # --- 拟合线 ---
    line_styles = ["-", (0, (5, 2)), "--", "-.", ":", (0, (3, 1, 1, 1))]
    method_results = [
        ("speed_m1", r1, "daily_diff"),
        ("speed_m2", r2, "lon_diff"),
        ("speed_m3", r3, "lsq"),
        ("speed_m4", r4, "lsq"),
        ("speed_m5", r5, "lsq"),
        ("speed_m6", r6, "lsq"),
    ]

    for i, (key, result, method_type) in enumerate(method_results):
        color = METHOD_COLORS[key]
        name = METHOD_NAMES[key]
        speed = result.get("speed_m_s", np.nan)

        if method_type == "lsq" and result.get("slope") is not None:
            t_range = np.array([day_indices[0], day_indices[-1]])
            lon_line = result["intercept"] + result["slope"] * t_range
            time_line = [np.datetime64(t0 + pd.Timedelta(days=float(d))) for d in t_range]
            r2_val = result.get("r2", np.nan)
            label = f"{name}: {speed:.1f} m/s (R\u00b2={r2_val:.2f})"
            ax.plot(lon_line, time_line, color=color, linewidth=2.5, zorder=7,
                    linestyle=line_styles[i], label=label)

        elif method_type == "daily_diff":
            # M1: 从第一个有效逐日中心点到最后一个有效中心点画线
            valid_m = np.isfinite(daily_center)
            if valid_m.sum() >= 2:
                first_idx = np.where(valid_m)[0][0]
                last_idx = np.where(valid_m)[0][-1]
                lon_start_pt = daily_center[first_idx]
                lon_end_pt = daily_center[last_idx]
                t_start_pt = np.datetime64(t0 + pd.Timedelta(days=float(daily_days[first_idx])))
                t_end_pt = np.datetime64(t0 + pd.Timedelta(days=float(daily_days[last_idx])))
                label = f"{name}: {speed:.1f} m/s (avg diff)"
                ax.plot([lon_start_pt, lon_end_pt], [t_start_pt, t_end_pt],
                        color=color, linewidth=2.5, zorder=7,
                        linestyle=line_styles[i], label=label)

        elif method_type == "lon_diff":
            # M2: 从第一个逐经度中心点到最后一个中心点画线
            if len(lon_centers_lon) >= 2:
                lon_start_pt = lon_centers_lon[0]
                lon_end_pt = lon_centers_lon[-1]
                t_start_pt = np.datetime64(t0 + pd.Timedelta(days=float(lon_centers_t[0])))
                t_end_pt = np.datetime64(t0 + pd.Timedelta(days=float(lon_centers_t[-1])))
                label = f"{name}: {speed:.1f} m/s (avg diff)"
                ax.plot([lon_start_pt, lon_end_pt], [t_start_pt, t_end_pt],
                        color=color, linewidth=2.5, zorder=7,
                        linestyle=line_styles[i], label=label)

    # 格式
    ax.yaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(time_vals) // 15)))
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.set_xticks(XTICK_LOCS)
    ax.set_xticklabels(XTICK_LABELS, fontsize=9)
    ax.set_xlim(LON_RANGE)
    ax.set_ylabel("Date")
    ax.axvline(x=60, color="black", linewidth=0.6, alpha=0.5)
    ax.axvline(x=180, color="black", linewidth=0.6, alpha=0.5)

    dur = int(ev_row["duration_days"])
    ax.set_title(
        f"Event #18: {t0.strftime('%Y-%m-%d')} - {t1.strftime('%Y-%m-%d')} ({dur}d)\n"
        f"6 Phase Speed Methods Comparison",
        fontsize=11, fontweight="bold")

    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=1)
    cbar = plt.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, aspect=30, shrink=0.85)
    cbar.set_label("OLR anomaly (W/m\u00b2)", fontsize=8)

    plt.tight_layout()
    out = out_dir / "event_018_hovmoller_6methods.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =====================================================================
# 图3: ONI相关性对比图
# =====================================================================
def plot_oni_correlation(df, out_dir):
    oni_df = parse_oni(ONI_TXT)

    oni_vals, enso_cats = [], []
    for _, row in df.iterrows():
        val, cat = classify_event(row["start_date"], row["end_date"], oni_df)
        oni_vals.append(val)
        enso_cats.append(cat)
    df = df.copy()
    df["oni_avg"] = oni_vals
    df["enso_phase"] = enso_cats

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    speed_cols = list(METHOD_NAMES.keys())

    for i, col in enumerate(speed_cols):
        ax = axes[i]
        oni = df["oni_avg"].values
        speed = df[col].values
        valid = np.isfinite(oni) & np.isfinite(speed)
        xv, yv = oni[valid], speed[valid]

        colors = []
        for _, row in df[valid].iterrows():
            colors.append(ENSO_COLORS.get(row["enso_phase"], "gray"))

        ax.scatter(xv, yv, c=colors, s=25, alpha=0.7, edgecolors="black",
                   linewidths=0.3, zorder=3)

        slope, intercept, r_val, p_val, _ = stats.linregress(xv, yv)
        x_line = np.linspace(xv.min() - 0.3, xv.max() + 0.3, 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", lw=2, zorder=2)

        p_str = f"p<0.001" if p_val < 0.001 else f"p={p_val:.4f}"
        ax.text(0.95, 0.98, f"Cor={r_val:.2f}\n{p_str}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="top", ha="right",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        ax.set_xlabel("ONI (\u00b0C)", fontsize=10)
        ax.set_ylabel("Phase Speed (m/s)", fontsize=10)
        ax.set_title(METHOD_NAMES[col], fontsize=11, fontweight="bold")
        ax.axvline(0.5, color="#E74C3C", ls=":", alpha=0.4, lw=1)
        ax.axvline(-0.5, color="#3498DB", ls=":", alpha=0.4, lw=1)
        ax.axvline(0, color="gray", ls="--", alpha=0.3, lw=1)
        ax.grid(alpha=0.2)
        ax.tick_params(direction="in", top=True, right=True)

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                       markersize=10, label=g) for g, c in ENSO_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", fontsize=10, framealpha=0.8,
               ncol=3, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("ONI vs Phase Speed: 6 Methods Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = out_dir / "oni_vs_phase_speed_6methods.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =====================================================================
# 图4: R² 对比箱线图
# =====================================================================
def plot_r2_comparison(df, out_dir):
    r2_cols = ["r2_m3", "r2_m4", "r2_m5", "r2_m6"]
    r2_names = ["M3: DailyCenter", "M4: LonCenter", "M5: DailyHalfMax", "M6: LonHalfMax"]
    r2_colors = [METHOD_COLORS["speed_m3"], METHOD_COLORS["speed_m4"],
                 METHOD_COLORS["speed_m5"], METHOD_COLORS["speed_m6"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    data = [df[c].dropna().values for c in r2_cols]
    bp = ax.boxplot(data, tick_labels=r2_names, patch_artist=True,
                    widths=0.5, medianprops=dict(color="black", lw=2))

    for patch, color in zip(bp["boxes"], r2_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    rng = np.random.default_rng(42)
    for i, (d, c) in enumerate(zip(data, r2_colors)):
        jitter = rng.uniform(-0.15, 0.15, len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d,
                   c=c, s=15, alpha=0.5, zorder=3, edgecolors="none")

    for i, d in enumerate(data):
        mean_v = np.mean(d)
        ax.scatter([i + 1], [mean_v], marker="D", c="white", edgecolors="black",
                   s=50, zorder=5)
        ax.text(i + 1 + 0.25, mean_v, f"mean={mean_v:.3f}", fontsize=8, va="center")

    ax.set_ylabel("R\u00b2 (coefficient of determination)", fontsize=12)
    ax.set_title("Fit Quality (R\u00b2) Comparison Across Methods (M3-M6)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out = out_dir / "r2_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# =====================================================================
# 图5: 综合评估汇总表
# =====================================================================
def plot_summary_table(df, out_dir):
    oni_df = parse_oni(ONI_TXT)
    oni_vals = []
    for _, row in df.iterrows():
        val, _ = classify_event(row["start_date"], row["end_date"], oni_df)
        oni_vals.append(val)
    df = df.copy()
    df["oni_avg"] = oni_vals

    speed_cols = list(METHOD_NAMES.keys())
    r2_cols = [None, None, "r2_m3", "r2_m4", "r2_m5", "r2_m6"]

    rows = []
    for i, col in enumerate(speed_cols):
        vals = df[col].dropna()
        name = METHOD_NAMES[col]

        mean_v = vals.mean()
        std_v = vals.std()
        cv = std_v / mean_v if mean_v != 0 else np.nan

        reasonable = ((vals >= 1) & (vals <= 10)).sum() / len(vals) * 100

        r2_col = r2_cols[i]
        r2_mean = df[r2_col].dropna().mean() if r2_col else np.nan

        valid = np.isfinite(df["oni_avg"]) & np.isfinite(df[col])
        if valid.sum() > 5:
            r_oni, p_oni = stats.pearsonr(df["oni_avg"][valid], df[col][valid])
        else:
            r_oni, p_oni = np.nan, np.nan

        rows.append({
            "Method": name,
            "Mean": f"{mean_v:.2f}",
            "Std": f"{std_v:.2f}",
            "CV": f"{cv:.3f}",
            "Reas%": f"{reasonable:.1f}%",
            "R\u00b2": f"{r2_mean:.3f}" if np.isfinite(r2_mean) else "N/A",
            "ONI r": f"{r_oni:.2f}" if np.isfinite(r_oni) else "N/A",
            "ONI p": f"{p_oni:.4f}" if np.isfinite(p_oni) else "N/A",
        })

    df_summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    for (row_i, col_i), cell in table.get_celld().items():
        if row_i == 0:
            cell.set_facecolor("#34495E")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            if row_i % 2 == 0:
                cell.set_facecolor("#ECF0F1")
            else:
                cell.set_facecolor("white")

    ax.set_title("Comprehensive Evaluation Summary: 6 Phase Speed Methods",
                 fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    out = out_dir / "summary_table.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    csv_out = out_dir / "summary_table.csv"
    df_summary.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("plot_phase_speed_comparison.py")
    print("六种MJO相速度计算方法对比可视化")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SPEED_CSV, parse_dates=["start_date", "end_date"])
    print(f"  Loaded {len(df)} events from {SPEED_CSV}")

    print("\n[1/5] Distribution plots...")
    plot_distribution(df, OUT_DIR)

    print("\n[2/5] Event18 Hovmoller...")
    plot_event18_hovmoller(df, OUT_DIR)

    print("\n[3/5] ONI correlation plots...")
    plot_oni_correlation(df, OUT_DIR)

    print("\n[4/5] R\u00b2 comparison...")
    plot_r2_comparison(df, OUT_DIR)

    print("\n[5/5] Summary table...")
    plot_summary_table(df, OUT_DIR)

    print(f"\n\u2705 All figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
