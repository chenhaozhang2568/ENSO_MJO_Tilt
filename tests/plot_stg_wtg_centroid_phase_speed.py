# -*- coding: utf-8 -*-
"""
plot_stg_wtg_centroid_phase_speed.py
按 tilt_q_centroid 强弱分组（STG/WTG），比较两组相速度差异

分组标准：
  STG (Strong Tilting): tilt_q_centroid > mean + 0.7*std
  WTG (Weak Tilting):   tilt_q_centroid < mean - 0.7*std
  中间组不参与比较
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

CSV = r"E:\Datas\Derived\tilt_q_centroid_phase_speed_by_enso.csv"
OUT_DIR = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\enso_tilt_q_centroid"
SIGMA_THRESH = 0.7


def main():
    df = pd.read_csv(CSV)
    tilt = df["mean_tilt_q_centroid"].values
    speed = df["phase_speed_m_s"].values

    mean_t, std_t = np.nanmean(tilt), np.nanstd(tilt)
    hi = mean_t + SIGMA_THRESH * std_t
    lo = mean_t - SIGMA_THRESH * std_t

    stg = df[df["mean_tilt_q_centroid"] >= hi]
    wtg = df[df["mean_tilt_q_centroid"] <= lo]
    mid = df[(df["mean_tilt_q_centroid"] > lo) & (df["mean_tilt_q_centroid"] < hi)]

    print(f"Tilt_q_centroid: mean={mean_t:.2f}, std={std_t:.2f}")
    print(f"Thresholds: STG >= {hi:.2f}, WTG <= {lo:.2f}")
    print(f"STG: {len(stg)}, Mid: {len(mid)}, WTG: {len(wtg)}")

    s_stg = stg["phase_speed_m_s"].dropna()
    s_wtg = wtg["phase_speed_m_s"].dropna()

    t_stat, t_pval = stats.ttest_ind(s_stg, s_wtg, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(s_stg, s_wtg, alternative="two-sided")

    print(f"\nPhase Speed comparison:")
    print(f"  STG: mean={s_stg.mean():.2f}, std={s_stg.std():.2f}, n={len(s_stg)}")
    print(f"  WTG: mean={s_wtg.mean():.2f}, std={s_wtg.std():.2f}, n={len(s_wtg)}")
    print(f"  t-test: t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={u_pval:.4f}")

    # === Plot: 2 panels ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- Left: scatter tilt_q_centroid vs speed, colored by group ---
    ax = axes[0]
    ax.scatter(mid["mean_tilt_q_centroid"], mid["phase_speed_m_s"],
               c="#CCCCCC", s=25, alpha=0.5, label="Middle", edgecolors="none")
    ax.scatter(wtg["mean_tilt_q_centroid"], wtg["phase_speed_m_s"],
               c="#3498DB", s=40, alpha=0.8, label=f"WTG (n={len(wtg)})",
               edgecolors="black", linewidths=0.3)
    ax.scatter(stg["mean_tilt_q_centroid"], stg["phase_speed_m_s"],
               c="#E74C3C", s=40, alpha=0.8, label=f"STG (n={len(stg)})",
               edgecolors="black", linewidths=0.3)

    ax.axvline(hi, color="#E74C3C", ls="--", alpha=0.6, lw=1.5)
    ax.axvline(lo, color="#3498DB", ls="--", alpha=0.6, lw=1.5)

    # fit line (all data)
    valid = np.isfinite(tilt) & np.isfinite(speed)
    slope, intercept, r, p, _ = stats.linregress(tilt[valid], speed[valid])
    x_line = np.linspace(tilt.min() - 1, tilt.max() + 1, 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5, alpha=0.6)
    ax.text(0.05, 0.95, f"Cor={r:.2f}\np={p:.4f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel("Tilt_q_centroid (deg)", fontsize=12)
    ax.set_ylabel("Phase Speed (m/s)", fontsize=12)
    ax.set_title("Tilt_q_centroid vs Phase Speed (STG/WTG)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.grid(alpha=0.2)
    ax.tick_params(direction="in", top=True, right=True)

    # --- Right: boxplot comparison ---
    ax2 = axes[1]
    data = [s_wtg.values, s_stg.values]
    bp = ax2.boxplot(data, tick_labels=[f"WTG\n(n={len(s_wtg)})", f"STG\n(n={len(s_stg)})"],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color="black", lw=2))
    bp["boxes"][0].set_facecolor("#3498DB")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#E74C3C")
    bp["boxes"][1].set_alpha(0.7)

    # jitter scatter
    rng = np.random.default_rng(42)
    for i, (d, c) in enumerate([(s_wtg.values, "#3498DB"), (s_stg.values, "#E74C3C")]):
        jitter = rng.uniform(-0.12, 0.12, len(d))
        ax2.scatter(np.full(len(d), i + 1) + jitter, d,
                    c=c, s=20, alpha=0.6, zorder=3, edgecolors="none")

    # significance bracket
    ymax = max(s_stg.max(), s_wtg.max())
    y_range = ax2.get_ylim()[1] - ax2.get_ylim()[0]
    h = ymax + y_range * 0.05
    ax2.plot([1, 1, 2, 2], [h, h + 0.1, h + 0.1, h], "k-", lw=1.2)

    if t_pval < 0.001:
        sig_txt = f"p<0.001***"
    elif t_pval < 0.01:
        sig_txt = f"p={t_pval:.3f}***"
    elif t_pval < 0.05:
        sig_txt = f"p={t_pval:.3f}**"
    elif t_pval < 0.1:
        sig_txt = f"p={t_pval:.2f}*"
    else:
        sig_txt = f"p={t_pval:.2f}"
    ax2.text(1.5, h + 0.15, sig_txt, ha="center", fontsize=11, fontweight="bold")

    # mean markers
    ax2.scatter([1], [s_wtg.mean()], marker="D", c="white", edgecolors="black",
                s=60, zorder=5, label=f"Mean: {s_wtg.mean():.2f}")
    ax2.scatter([2], [s_stg.mean()], marker="D", c="white", edgecolors="black",
                s=60, zorder=5, label=f"Mean: {s_stg.mean():.2f}")

    ax2.set_ylabel("Phase Speed (m/s)", fontsize=12)
    ax2.set_title("Phase Speed: WTG vs STG", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(direction="in", top=True, right=True)

    # adjust ylim to fit bracket
    ax2.set_ylim(ax2.get_ylim()[0], h + 0.6)

    plt.tight_layout()
    from pathlib import Path
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out = f"{OUT_DIR}/stg_wtg_phase_speed.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
