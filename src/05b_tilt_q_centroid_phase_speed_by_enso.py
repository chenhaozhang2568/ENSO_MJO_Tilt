# -*- coding: utf-8 -*-
"""
05b_tilt_q_centroid_phase_speed_by_enso.py

按 ENSO 背景分类 MJO 事件，统计 tilt_q_centroid 和相速度，
进行组间 t 检验和 Mann-Whitney U 检验。

与 05b_tilt_q_phase_speed_by_enso.py 相同，但使用 centroid 定义的 tilt。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ======================
# PATHS
# ======================
EVENTS_CSV   = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
TILT_Q_NC    = r"E:\Datas\Derived\tilt_q_centroid_daily_1979-2022.nc"
PHASE_CSV    = r"E:\Datas\Derived\phase_speed_q_events.csv"
ONI_TXT      = r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt"
FIG_DIR      = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\enso_tilt_q_centroid")
OUT_CSV      = r"E:\Datas\Derived\tilt_q_centroid_phase_speed_by_enso.csv"

# ======================
# SETTINGS
# ======================
ONI_ELNINO = 0.5
ONI_LANINA = -0.5
ENSO_ORDER = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}


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


def compute_event_mean_tilt_q_centroid():
    import xarray as xr
    ds = xr.open_dataset(TILT_Q_NC)
    df_ev = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    times = pd.to_datetime(ds["time"].values)
    vals = ds["tilt_q_centroid"].values.astype(float)

    rows = []
    for _, r in df_ev.iterrows():
        ts = np.datetime64(r["start_date"])
        te = np.datetime64(r["end_date"])
        m = (times >= ts) & (times <= te)
        v = vals[m]
        v = v[np.isfinite(v)]
        rows.append({
            "event_id": r["event_id"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "mean_tilt_q_centroid": np.mean(v) if len(v) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def print_group_stats(df, var, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"{'Group':<12} {'N':>4} {'Mean':>8} {'Std':>8} {'Median':>8} {'Min':>8} {'Max':>8}")
    print("-" * 55)
    for g in ENSO_ORDER:
        sub = df[df["enso_phase"] == g][var].dropna()
        if len(sub) == 0:
            continue
        print(f"{g:<12} {len(sub):>4} {sub.mean():>8.2f} {sub.std():>8.2f} "
              f"{sub.median():>8.2f} {sub.min():>8.2f} {sub.max():>8.2f}")


def print_significance(df, var, label):
    print(f"\n--- {label}: Significance Tests ---")
    pairs = [("El Nino", "La Nina"), ("El Nino", "Neutral"), ("La Nina", "Neutral")]
    print(f"{'Comparison':<25} {'t-stat':>8} {'t-pval':>9} {'U-stat':>10} {'U-pval':>9}")
    print("-" * 62)
    for g1, g2 in pairs:
        a = df[df["enso_phase"] == g1][var].dropna().values
        b = df[df["enso_phase"] == g2][var].dropna().values
        if len(a) < 2 or len(b) < 2:
            print(f"{g1} vs {g2:<10} {'N/A':>8}")
            continue
        t_stat, t_pval = stats.ttest_ind(a, b, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(a, b, alternative="two-sided")
        sig_t = "***" if t_pval < 0.01 else ("**" if t_pval < 0.05 else ("*" if t_pval < 0.1 else ""))
        print(f"{g1} vs {g2:<10} {t_stat:>8.3f} {t_pval:>8.4f}{sig_t:>2} "
              f"{u_stat:>9.0f} {u_pval:>8.4f}")


def _add_significance_brackets(ax, df, var, enso_order):
    pairs = [(0, 1), (1, 2), (0, 2)]
    pair_labels = [
        (enso_order[0], enso_order[1]),
        (enso_order[1], enso_order[2]),
        (enso_order[0], enso_order[2]),
    ]

    pvals = []
    for g1, g2 in pair_labels:
        a = df[df["enso_phase"] == g1][var].dropna().values
        b = df[df["enso_phase"] == g2][var].dropna().values
        if len(a) >= 2 and len(b) >= 2:
            _, p = stats.ttest_ind(a, b, equal_var=False)
            pvals.append(p)
        else:
            pvals.append(np.nan)

    ymin, ymax = ax.get_ylim()
    y_range = ymax - ymin
    h_step = y_range * 0.06
    y_start = ymax + y_range * 0.02

    for level, ((i, j), p) in enumerate(zip(pairs, pvals)):
        if np.isnan(p):
            continue
        x1, x2 = i + 1, j + 1
        y = y_start + level * h_step * 2
        h = h_step * 0.4

        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.0)

        if p < 0.001:
            txt = f"p<0.001***"
        elif p < 0.01:
            txt = f"p={p:.3f}***"
        elif p < 0.05:
            txt = f"p={p:.3f}**"
        elif p < 0.1:
            txt = f"p={p:.2f}*"
        else:
            txt = f"p={p:.2f}"
        ax.text((x1 + x2) / 2, y + h + h_step * 0.1, txt,
                ha="center", va="bottom", fontsize=8.5, color="black")

    ax.set_ylim(ymin, y_start + len(pairs) * h_step * 2 + h_step * 1.5)


def plot_box_and_bar(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for col_idx, (var, ylabel, title) in enumerate([
        ("mean_tilt_q_centroid", "Tilt_q_centroid (deg)", "Tilt_q_centroid by ENSO Phase"),
        ("phase_speed_m_s", "Phase Speed (m/s)", "Phase Speed by ENSO Phase"),
    ]):
        ax = axes[col_idx]
        data_groups = [df[df["enso_phase"] == g][var].dropna().values for g in ENSO_ORDER]
        bp = ax.boxplot(data_groups, tick_labels=ENSO_ORDER,
                        patch_artist=True, showfliers=True,
                        widths=0.5, medianprops=dict(color="black", lw=2))
        for patch, g in zip(bp["boxes"], ENSO_ORDER):
            patch.set_facecolor(ENSO_COLORS[g])
            patch.set_alpha(0.7)

        for i, g in enumerate(ENSO_ORDER):
            sub = df[df["enso_phase"] == g][var].dropna().values
            n = len(sub)
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, n)
            ax.scatter(np.full(n, i + 1) + jitter, sub,
                       c=ENSO_COLORS[g], s=18, alpha=0.6, zorder=3, edgecolors="none")

        counts = [len(df[df["enso_phase"] == g]) for g in ENSO_ORDER]
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f"{g}\n(n={c})" for g, c in zip(ENSO_ORDER, counts)], fontsize=10)

        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        _add_significance_brackets(ax, df, var, ENSO_ORDER)

    fig.suptitle("MJO Tilt_q_centroid and Phase Speed by ENSO Phase",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = out_dir / "tilt_q_phase_speed_by_enso.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ONI...")
    oni_df = parse_oni(ONI_TXT)

    print("Computing event-mean tilt_q_centroid...")
    df_tilt = compute_event_mean_tilt_q_centroid()

    print("Loading phase speed...")
    df_ps = pd.read_csv(PHASE_CSV)

    # Merge
    df = df_tilt.merge(df_ps[["event_id", "phase_speed_m_s", "r2"]], on="event_id")

    # Classify ENSO
    oni_vals, enso_cats = [], []
    for _, row in df.iterrows():
        val, cat = classify_event(row["start_date"], row["end_date"], oni_df)
        oni_vals.append(val)
        enso_cats.append(cat)
    df["oni_avg"] = oni_vals
    df["enso_phase"] = enso_cats

    # Save
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")

    # Counts
    print(f"\nENSO distribution: {df['enso_phase'].value_counts().to_dict()}")

    # Statistics
    print_group_stats(df, "mean_tilt_q_centroid", "Tilt_q_centroid by ENSO")
    print_group_stats(df, "phase_speed_m_s", "Phase Speed by ENSO")

    # Significance
    print_significance(df, "mean_tilt_q_centroid", "Tilt_q_centroid")
    print_significance(df, "phase_speed_m_s", "Phase Speed")

    # Plot
    plot_box_and_bar(df, FIG_DIR)


if __name__ == "__main__":
    main()
