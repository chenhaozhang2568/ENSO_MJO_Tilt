# -*- coding: utf-8 -*-
"""
plot_centroid_correlation.py
平均场水汽重心(centroid)、上层西边界、相速度三者两两相关性散点图
使用平均场上直接计算的值
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

PHASE_SPEED_CSV = r"E:\Datas\Derived\phase_speed_q_events.csv"
FIELD_CSV = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_diagnose\event_mean_field_values.csv"
OUT_DIR = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_diagnose"


def plot_scatter(x, y, xlabel, ylabel, caption_x, caption_y, out_path):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, c="black", s=25, zorder=3)
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, zorder=2)

    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.text(0.95, 0.98, f"Cor={r_val:.2f}",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="right")

    ax.tick_params(axis="both", which="major", labelsize=12,
                   direction="in", top=True, right=True, length=6)
    ax.tick_params(axis="both", which="minor",
                   direction="in", top=True, right=True, length=3)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    sig = "99%" if p_val < 0.01 else ("95%" if p_val < 0.05 else "not sig.")
    caption = (
        f"Scatter diagram of {caption_y} (y) vs.\n"
        f"{caption_x} (x) for {len(x)} MJO events.\n"
        f"Red line: least squares fit.\n"
        f"Cor = {r_val:.2f} (p = {p_val:.4f}),\n"
        f"exceeding the {sig} confidence level."
    )
    fig.text(0.5, -0.10, caption, ha="center", fontsize=10, style="italic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Cor={r_val:.2f}, p={p_val:.4f}, N={len(x)} → {out_path}")


def main():
    from pathlib import Path
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    df_field = pd.read_csv(FIELD_CSV)
    df_ps = pd.read_csv(PHASE_SPEED_CSV)
    df = df_ps.merge(df_field, on="event_id")

    speed = df["phase_speed_m_s"].values
    up_west = df["field_up_west"].values
    centroid = df["field_centroid"].values

    print("[1] q Centroid vs Phase speed:")
    plot_scatter(speed, centroid,
                 "Speed (m/s)", "q Centroid Position (°)",
                 "phase speed", "q centroid (mean-field)",
                 f"{OUT_DIR}/centroid_vs_speed.png")

    print("[2] ω West Boundary vs Phase speed:")
    plot_scatter(speed, up_west,
                 "Speed (m/s)", "ω West Boundary (°)",
                 "phase speed", "ω west boundary (mean-field)",
                 f"{OUT_DIR}/omega_west_vs_speed.png")

    print("[3] ω West Boundary vs q Centroid:")
    plot_scatter(up_west, centroid,
                 "ω West Boundary (°)", "q Centroid Position (°)",
                 "ω west boundary (mean-field)", "q centroid (mean-field)",
                 f"{OUT_DIR}/omega_west_vs_centroid.png")

    print("Done.")


if __name__ == "__main__":
    main()
