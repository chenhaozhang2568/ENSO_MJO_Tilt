# -*- coding: utf-8 -*-
"""
plot_oni_vs_tilt_centroid_phase_speed.py
ONI 与 tilt_q_centroid / phase_speed 的连续回归散点图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CSV = r"E:\Datas\Derived\tilt_q_centroid_phase_speed_by_enso.csv"
OUT = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\enso_tilt_q_centroid\oni_vs_tilt_phase_speed.png"

ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}


def main():
    df = pd.read_csv(CSV)
    oni = df["oni_avg"].values
    tilt = df["mean_tilt_q_centroid"].values
    speed = df["phase_speed_m_s"].values

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, y, ylabel, title in [
        (axes[0], tilt, "Tilt_q_centroid (deg)", "ONI vs Tilt_q_centroid"),
        (axes[1], speed, "Phase Speed (m/s)", "ONI vs Phase Speed"),
    ]:
        valid = np.isfinite(oni) & np.isfinite(y)
        xv, yv = oni[valid], y[valid]

        # color by ENSO phase
        colors = []
        for _, row in df[valid].iterrows():
            colors.append(ENSO_COLORS.get(row["enso_phase"], "gray"))

        ax.scatter(xv, yv, c=colors, s=30, alpha=0.7, edgecolors="black",
                   linewidths=0.3, zorder=3)

        # linear fit
        slope, intercept, r_val, p_val, _ = stats.linregress(xv, yv)
        x_line = np.linspace(xv.min() - 0.3, xv.max() + 0.3, 100)
        ax.plot(x_line, slope * x_line + intercept, "r-", lw=2, zorder=2)

        # annotation
        ax.text(0.95, 0.98, f"Cor={r_val:.2f}\np={p_val:.4f}",
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                va="top", ha="right",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

        ax.set_xlabel("ONI (°C)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axvline(0.5, color="#E74C3C", ls=":", alpha=0.4, lw=1)
        ax.axvline(-0.5, color="#3498DB", ls=":", alpha=0.4, lw=1)
        ax.axvline(0, color="gray", ls="--", alpha=0.3, lw=1)
        ax.grid(alpha=0.2)

        ax.tick_params(direction="in", top=True, right=True)

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                       markersize=8, label=g)
               for g, c in ENSO_COLORS.items()]
    axes[1].legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    from pathlib import Path
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
