# -*- coding: utf-8 -*-
"""
diagnose_three_groups.py — 按事件逐日平均 up_west 分三组，逐日 omega 剖面对比
（已改用逐日 up_west_rel 的事件内均值代替平均场 field_up_west）

分组规则（基于逐日 up_west_rel 的事件内均值 event_up_west）：
  G1 极端偏西: event_up_west <= -70°
  G2 中间偏西: -70° < event_up_west <= -45°
  G3 正常范围: event_up_west > -45°

输出（到 upper_west_diagnose/three_groups/）：
  1. daily_upwest_distribution.png — 三组逐日 up_west 分布直方图 + KS/MW 检验
  2. group_statistics_summary.png — 三组统计汇总表
  3. G1_daily_profiles/ — G1 随机20日 omega 剖面图
  4. G2_daily_profiles/ — G2 随机20日 omega 剖面图
  5. G3_daily_profiles/ — G3 随机20日 omega 剖面图
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
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

# === PATHS ===
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
TILT_Q_NC  = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
W_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
SPEED_CSV  = r"E:\Datas\Derived\phase_speed_q_events.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\upper_west_diagnose\three_groups")

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}
SMOOTH_WINDOW = 10
SEED = 42


def _smooth_1d(profile, window):
    if window <= 1:
        return profile
    kernel = np.ones(window) / window
    valid = np.isfinite(profile).astype(float)
    filled = np.where(np.isfinite(profile), profile, 0.0)
    smoothed = np.convolve(filled, kernel, mode='same')
    count = np.convolve(valid, kernel, mode='same')
    count[count < 1e-10] = np.nan
    return smoothed / count


# ==================================================================
# Part 1: 分组 + 逐日 up_west 分布统计
# ==================================================================
def classify_events():
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    df_speed = pd.read_csv(SPEED_CSV)

    # 从逐日数据计算每个事件的 up_west_rel 平均值
    ds = xr.open_dataset(TILT_Q_NC)
    times = pd.to_datetime(ds["time"].values)
    uw = ds["up_west_rel"].values.astype(float)

    event_up_wests = []
    for _, ev in events.iterrows():
        mask = (times >= ev["start_date"]) & (times <= ev["end_date"])
        vals = uw[mask]
        valid = vals[np.isfinite(vals)]
        event_up_wests.append(np.mean(valid) if len(valid) > 0 else np.nan)

    events["event_up_west"] = event_up_wests

    df = events.merge(
        df_speed[["event_id", "phase_speed_m_s"]], on="event_id", how="left")

    # 按三分位数自动分组
    valid_uw = df["event_up_west"].dropna()
    q33 = np.percentile(valid_uw, 33.3)
    q66 = np.percentile(valid_uw, 66.7)
    print(f"  分组阈值 (三分位): q33={q33:.1f}°, q66={q66:.1f}°")

    df["group"] = "G3_normal"
    df.loc[df["event_up_west"] <= q33, "group"] = "G1_extreme"
    df.loc[(df["event_up_west"] > q33) & (df["event_up_west"] <= q66), "group"] = "G2_middle"

    for g in ["G1_extreme", "G2_middle", "G3_normal"]:
        n = (df["group"] == g).sum()
        sub = df[df["group"] == g]
        uw_range = sub["event_up_west"]
        print(f"  {g}: {n} events, event_up_west range: [{uw_range.min():.1f}, {uw_range.max():.1f}]")
    return df, q33, q66


def collect_daily_upwest(df_events):
    """收集三组事件的逐日 up_west 值。"""
    ds = xr.open_dataset(TILT_Q_NC)
    times = pd.to_datetime(ds["time"].values)
    uw = ds["up_west_rel"].values.astype(float)

    daily = {"G1_extreme": [], "G2_middle": [], "G3_normal": []}
    daily_meta = {"G1_extreme": [], "G2_middle": [], "G3_normal": []}

    for _, ev in df_events.iterrows():
        g = ev["group"]
        mask = (times >= ev["start_date"]) & (times <= ev["end_date"])
        idxs = np.where(mask)[0]
        for i in idxs:
            v = uw[i]
            if np.isfinite(v):
                daily[g].append(v)
                daily_meta[g].append({"time_idx": i, "event_id": ev["event_id"],
                                       "date": times[i]})

    for g in daily:
        daily[g] = np.array(daily[g])
        print(f"  {g}: {len(daily[g])} valid daily values")
    return daily, daily_meta


def plot_daily_distribution(daily, fig_dir, q33, q66):
    """图1: 三组逐日 up_west 分布 + 统计检验。"""
    # 只绘制非空组
    active_groups = [g for g in ["G1_extreme", "G2_middle", "G3_normal"] if len(daily[g]) > 0]
    n_panels = len(active_groups)
    if n_panels == 0:
        print("  No groups with data, skipping.")
        return
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    colors = {"G1_extreme": "#E74C3C", "G2_middle": "#F39C12", "G3_normal": "#3498DB"}
    labels = {"G1_extreme": f"G1: West\n(event up_west ≤ {q33:.1f}°)",
              "G2_middle": f"G2: Middle\n({q33:.1f}° < event up_west ≤ {q66:.1f}°)",
              "G3_normal": f"G3: East\n(event up_west > {q66:.1f}°)"}

    for ax, g in zip(axes, active_groups):
        vals = daily[g]
        ax.hist(vals, bins=40, color=colors[g], edgecolor="black", alpha=0.7)
        mean_v = np.mean(vals)
        median_v = np.median(vals)
        ax.axvline(mean_v, color="navy", ls="--", lw=2, label=f"Mean: {mean_v:.1f}°")
        ax.axvline(median_v, color="darkgreen", ls=":", lw=2, label=f"Median: {median_v:.1f}°")
        ax.set_xlabel("Daily Up-West (relative lon, °)", fontsize=11)
        ax.set_ylabel("Count (days)", fontsize=11)
        ax.set_title(f"{labels[g]}\n(N={len(vals)})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        stats_text = (f"Mean: {mean_v:.1f}°\nMedian: {median_v:.1f}°\n"
                      f"Std: {np.std(vals):.1f}°\n"
                      f"[{np.min(vals):.1f}, {np.max(vals):.1f}]")
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    out = fig_dir / "daily_upwest_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # 统计检验
    print("\n  === 组间差异检验 ===")
    groups = ["G1_extreme", "G2_middle", "G3_normal"]
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            g1, g2 = groups[i], groups[j]
            v1, v2 = daily[g1], daily[g2]
            ks_stat, ks_p = stats.ks_2samp(v1, v2)
            mw_stat, mw_p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
            t_stat, t_p = stats.ttest_ind(v1, v2, equal_var=False)
            print(f"  {g1} vs {g2}:")
            print(f"    KS test: D={ks_stat:.3f}, p={ks_p:.4e}")
            print(f"    MW-U:    U={mw_stat:.0f}, p={mw_p:.4e}")
            print(f"    Welch-t: t={t_stat:.2f}, p={t_p:.4e}")
            print(f"    Mean diff: {np.mean(v1):.1f} vs {np.mean(v2):.1f}")


def plot_summary_table(df_events, daily, fig_dir):
    """图2: 统计汇总表。"""
    df_speed = pd.read_csv(SPEED_CSV)
    rows = []
    for g in ["G1_extreme", "G2_middle", "G3_normal"]:
        sub = df_events[df_events["group"] == g]
        spd = sub["phase_speed_m_s"].dropna()
        d = daily[g]
        rows.append([
            g.split("_")[0],
            str(len(sub)),
            str(len(d)),
            f"{sub['event_up_west'].mean():.1f}",
            f"{np.mean(d):.1f} ± {np.std(d):.1f}",
            f"{np.median(d):.1f}",
            f"[{np.min(d):.1f}, {np.max(d):.1f}]",
            f"{spd.mean():.2f} ± {spd.std():.2f}" if len(spd) > 0 else "N/A",
        ])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")
    col_labels = ["Group", "N events", "N days", "Event Mean\nUp-West",
                  "Daily Mean±Std", "Daily Median", "Daily Range", "Phase Speed\n(m/s)"]
    table = ax.table(cellText=rows, colLabels=col_labels,
                     colColours=["#D6EAF8"]*8, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # 行着色
    cell_colors = [["#FADBD8"]*8, ["#FDEBD0"]*8, ["#D5F5E3"]*8]
    for i, row_color in enumerate(cell_colors):
        for j in range(8):
            table[i+1, j].set_facecolor(row_color[j])

    ax.set_title("Group Statistics Summary", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    out = fig_dir / "group_statistics_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ==================================================================
# Part 2: 逐日 omega 剖面图
# ==================================================================
def plot_daily_omega_profiles(daily_meta, fig_dir):
    """为三组各随机挑选20天，画 omega 垂直剖面。"""
    print("\n  Loading omega data for profiles...")
    ds_w = xr.open_dataset(W_NORM_NC)
    ds3 = xr.open_dataset(STEP3_NC)
    ds_tilt = xr.open_dataset(TILT_Q_NC)

    w_all = ds_w["w_mjo_recon_norm"].values
    levels = ds_w["pressure_level"].values if "pressure_level" in ds_w else ds_w["level"].values
    lon = ds_w["lon"].values
    lon_360 = np.where(lon < 0, lon + 360, lon)
    sort_lon = np.argsort(lon_360)
    lon_360 = lon_360[sort_lon]

    time_w = pd.to_datetime(ds_w["time"].values)
    center_lon = ds3["center_lon_track"].values.astype(float)
    time_s3 = pd.to_datetime(ds3["time"].values)

    uw_daily = ds_tilt["up_west_rel"].values.astype(float)
    ue_daily = ds_tilt["up_east_rel"].values.astype(float)
    qmax_daily = ds_tilt["q_max_rel"].values.astype(float)
    time_tq = pd.to_datetime(ds_tilt["time"].values)

    heights = np.array([LEVEL_TO_HEIGHT[int(p)] for p in levels])

    rng = np.random.default_rng(SEED)

    for g, meta_list in daily_meta.items():
        if len(meta_list) == 0:
            continue
        out_dir = fig_dir / f"{g}_daily_profiles"
        out_dir.mkdir(parents=True, exist_ok=True)

        n_pick = min(20, len(meta_list))
        chosen = rng.choice(len(meta_list), size=n_pick, replace=False)
        print(f"  {g}: plotting {n_pick} profiles...")

        for ci in chosen:
            info = meta_list[ci]
            date = info["date"]
            eid = info["event_id"]

            # 找对应 time index
            w_idx = np.where(time_w == date)[0]
            s3_idx = np.where(time_s3 == date)[0]
            tq_idx = np.where(time_tq == date)[0]
            if len(w_idx) == 0 or len(s3_idx) == 0 or len(tq_idx) == 0:
                continue
            w_idx, s3_idx, tq_idx = w_idx[0], s3_idx[0], tq_idx[0]

            c = center_lon[s3_idx]
            if not np.isfinite(c):
                continue

            uw = uw_daily[tq_idx]
            ue = ue_daily[tq_idx]
            qm = qmax_daily[tq_idx]

            # 相对经度
            rel_lon = lon_360 - c
            mask_lon = (rel_lon >= -90) & (rel_lon <= 90)
            rel_lons = rel_lon[mask_lon]

            w_day = w_all[w_idx, :, :][:, sort_lon][:, mask_lon]

            # 平滑
            w_sm = np.full_like(w_day, np.nan)
            for k in range(len(levels)):
                w_sm[k, :] = _smooth_1d(w_day[k, :], SMOOTH_WINDOW)

            # 插值到高度坐标
            target_h = np.linspace(0.0, 12.0, 120)
            w_interp = np.full((len(target_h), len(rel_lons)), np.nan)
            for j in range(len(rel_lons)):
                col = w_sm[:, j]
                valid = np.isfinite(col)
                if valid.sum() >= 2:
                    f = interp1d(heights[valid], col[valid], kind="linear",
                                 bounds_error=False, fill_value=np.nan)
                    w_interp[:, j] = f(target_h)

            # 高层(400-200hPa)层平均剖面（用于标注零线）
            up_mask = (levels >= 200) & (levels <= 400)
            w_up_mean = np.nanmean(w_sm[up_mask, :], axis=0)

            # --- 绘图 ---
            fig, ax = plt.subplots(figsize=(14, 7))
            X, Y = np.meshgrid(rel_lons, target_h)

            vmax = np.nanmax(np.abs(w_interp)) * 0.8
            if vmax < 1e-6 or not np.isfinite(vmax):
                vmax = 0.01
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cf = ax.contourf(X, Y, w_interp, levels=np.linspace(-vmax, vmax, 21),
                             cmap="RdBu_r", norm=norm, extend="both")
            ax.contour(X, Y, w_interp, levels=[0], colors="black", linewidths=2.0)

            # 标注 up_west, up_east, q_max 点
            up_h_mid = (LEVEL_TO_HEIGHT[400] + LEVEL_TO_HEIGHT[200]) / 2.0
            low_h_mid = (LEVEL_TO_HEIGHT[1000] + LEVEL_TO_HEIGHT[850]) / 2.0

            if np.isfinite(uw) and np.isfinite(qm):
                tilt_val = qm - uw
                ax.plot([uw, qm], [up_h_mid, low_h_mid], "o-", color="gold",
                        markersize=12, markeredgecolor="black", markeredgewidth=1.5,
                        lw=3, zorder=10, label=f"Tilt_q = {tilt_val:.1f}°")
                ax.annotate(f"Upper W: {uw:.1f}°", (uw, up_h_mid),
                            textcoords="offset points", xytext=(10, 10),
                            fontsize=9, color="darkgoldenrod", fontweight="bold")
                ax.annotate(f"q_max: {qm:.1f}°", (qm, low_h_mid),
                            textcoords="offset points", xytext=(10, -15),
                            fontsize=9, color="darkgoldenrod", fontweight="bold")

            if np.isfinite(ue):
                ax.plot(ue, up_h_mid, "s", color="cyan", markersize=10,
                        markeredgecolor="black", zorder=10, label=f"Upper E: {ue:.1f}°")

            ax.axvline(0, color="limegreen", lw=2.5, alpha=0.8, label="Conv. Center")

            ax.set_ylim(0, 12)
            ax.set_xlim(-90, 90)
            ax.set_ylabel("Height (km)", fontsize=12)
            ax.set_xlabel("Relative Longitude (°)", fontsize=12)

            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
            ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in pticks])
            ax2.set_yticklabels([str(p) for p in pticks])
            ax2.set_ylabel("Pressure (hPa)", fontsize=12)

            cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.12, shrink=0.8)
            cbar.set_label("ω (normalized)", fontsize=10)

            title = (f"[{g}] Event #{eid} — {pd.Timestamp(date).strftime('%Y-%m-%d')}\n"
                     f"Center: {c:.1f}°E, Up-West: {uw:.1f}°")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)

            out = out_dir / f"daily_{pd.Timestamp(date).strftime('%Y-%m-%d')}_event_{eid:03d}.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()

        print(f"    Saved {n_pick} profiles to {out_dir}")


# ==================================================================
# Part 3: 组间差异诊断
# ==================================================================
def diagnose_differences(df_events, daily, fig_dir):
    """分析三组差异的可能原因。"""
    print("\n  === 组间差异诊断 ===")

    ds3 = xr.open_dataset(STEP3_NC)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    amp_all = ds3["amp"].values.astype(float)
    time_s3 = pd.to_datetime(ds3["time"].values)

    results = {}
    for g in ["G1_extreme", "G2_middle", "G3_normal"]:
        sub = df_events[df_events["group"] == g]
        centers, amps, durations = [], [], []
        for _, ev in sub.iterrows():
            mask = (time_s3 >= ev["start_date"]) & (time_s3 <= ev["end_date"])
            c = center_lon_all[mask]
            a = amp_all[mask]
            c_valid = c[np.isfinite(c)]
            a_valid = a[np.isfinite(a)]
            if len(c_valid) > 0:
                centers.append(np.mean(c_valid))
            if len(a_valid) > 0:
                amps.append(np.mean(a_valid))
            durations.append(float(ev["duration_days"]))

        results[g] = {
            "centers": np.array(centers),
            "amps": np.array(amps),
            "durations": np.array(durations),
            "speeds": sub["phase_speed_m_s"].dropna().values,
            "event_upwest": sub["event_up_west"].values,
        }

        print(f"\n  {g} (N={len(sub)}):")
        print(f"    Center Lon:  mean={np.mean(centers):.1f}°, std={np.std(centers):.1f}°")
        print(f"    Amplitude:   mean={np.mean(amps):.2f}, std={np.std(amps):.2f}")
        print(f"    Duration:    mean={np.mean(durations):.1f}d, std={np.std(durations):.1f}d")
        if len(results[g]["speeds"]) > 0:
            print(f"    Phase Speed: mean={np.mean(results[g]['speeds']):.2f}, "
                  f"std={np.std(results[g]['speeds']):.2f}")

    # 绘制三组对比条形图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    groups = ["G1_extreme", "G2_middle", "G3_normal"]
    colors = ["#E74C3C", "#F39C12", "#3498DB"]
    labels_short = ["G1\n(≤-70°)", "G2\n(-70~-45°)", "G3\n(>-45°)"]

    def _bar_compare(ax, data_lists, ylabel, title):
        means = [np.mean(d) for d in data_lists]
        stds = [np.std(d) for d in data_lists]
        x = np.arange(3)
        bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor="black",
                      capsize=5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_short, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                    f"{m:.1f}", ha="center", fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    _bar_compare(axes[0, 0],
                 [results[g]["centers"] for g in groups],
                 "Longitude (°E)", "Mean Center Longitude")
    _bar_compare(axes[0, 1],
                 [results[g]["amps"] for g in groups],
                 "Amplitude", "Mean MJO Amplitude")
    _bar_compare(axes[1, 0],
                 [results[g]["durations"] for g in groups],
                 "Days", "Mean Duration")
    _bar_compare(axes[1, 1],
                 [results[g]["speeds"] for g in groups],
                 "m/s", "Mean Phase Speed")

    plt.suptitle("Group Characteristics Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = fig_dir / "group_characteristics_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")

    # 组间检验
    print("\n  === 组间特征差异检验 ===")
    for feat, label in [("centers", "CenterLon"), ("amps", "Amplitude"),
                         ("durations", "Duration"), ("speeds", "PhaseSpeed")]:
        print(f"  {label}:")
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                d1 = results[groups[i]][feat]
                d2 = results[groups[j]][feat]
                if len(d1) > 1 and len(d2) > 1:
                    t, p = stats.ttest_ind(d1, d2, equal_var=False)
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                    print(f"    {groups[i]} vs {groups[j]}: "
                          f"mean={np.mean(d1):.2f} vs {np.mean(d2):.2f}, "
                          f"t={t:.2f}, p={p:.4f} {sig}")


# ==================================================================
def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("三组事件逐日 omega 剖面对比诊断")
    print("=" * 60)

    df_events, q33, q66 = classify_events()
    daily, daily_meta = collect_daily_upwest(df_events)

    print("\n[1] 逐日 up_west 分布图 + 检验")
    plot_daily_distribution(daily, FIG_DIR, q33, q66)

    print("\n[2] 统计汇总表")
    plot_summary_table(df_events, daily, FIG_DIR)

    print("\n[3] 逐日 omega 剖面图")
    plot_daily_omega_profiles(daily_meta, FIG_DIR)

    print("\n[4] 组间差异诊断")
    diagnose_differences(df_events, daily, FIG_DIR)

    print("\n" + "=" * 60)
    print(f"All outputs: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
