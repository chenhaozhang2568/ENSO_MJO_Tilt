# -*- coding: utf-8 -*-
"""
plot_mean_field_correlation.py
平均场上层西边界(q_max版)、下层q最大值与相速度的两两相关性散点图
使用平均场上直接计算的 up_west (非逐日均值)

三张图:
  1. field_up_west vs phase_speed
  2. field_q_max vs phase_speed
  3. field_up_west vs field_q_max
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

PHASE_SPEED_CSV = r"E:\Datas\Derived\phase_speed_q_events.csv"
FIELD_CSV = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_diagnose\event_mean_field_values.csv"
TILT_Q_NC = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
OUT_DIR = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\tilt_q_diagnose"

import xarray as xr

def compute_event_mean_q_max():
    """计算每个事件的平均 q_max_rel (从逐日数据)"""
    ds = xr.open_dataset(TILT_Q_NC)
    df_ev = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    times = pd.to_datetime(ds["time"].values)
    q_max = ds["q_max_rel"].values.astype(float)
    rows = []
    for _, r in df_ev.iterrows():
        ts = np.datetime64(r["start_date"])
        te = np.datetime64(r["end_date"])
        mask = (times >= ts) & (times <= te)
        v = q_max[mask]
        v = v[np.isfinite(v)]
        rows.append({"event_id": r["event_id"],
                      "mean_q_max": np.mean(v) if len(v) > 0 else np.nan})
    return pd.DataFrame(rows)


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
    df_qm = compute_event_mean_q_max()
    df = df_ps.merge(df_field, on="event_id").merge(df_qm, on="event_id")

    speed = df["phase_speed_m_s"].values
    up_west = df["field_up_west"].values
    q_max = df["mean_q_max"].values

    print("[1] Field ω west boundary vs Phase speed:")
    plot_scatter(speed, up_west,
                 "Speed (m/s)", "Upper ω West Boundary (°)",
                 "phase speed", "upper ω west boundary (mean-field)",
                 f"{OUT_DIR}/up_west_vs_phase_speed.png")

    print("[2] q max vs Phase speed:")
    plot_scatter(speed, q_max,
                 "Speed (m/s)", "Lower q Max Position (°)",
                 "phase speed", "lower q max position",
                 f"{OUT_DIR}/q_max_vs_phase_speed.png")

    print("[3] Field ω west boundary vs q max:")
    plot_scatter(up_west, q_max,
                 "Upper ω West Boundary (°)", "Lower q Max Position (°)",
                 "upper ω west boundary (mean-field)", "lower q max position",
                 f"{OUT_DIR}/up_west_vs_q_max.png")

    print("Done.")


if __name__ == "__main__":
    main()
