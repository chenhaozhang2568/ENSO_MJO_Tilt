# -*- coding: utf-8 -*-
"""
plot_tilt_q_vs_phase_speed.py
Tilt_q 与相速度散点图（风格参考 Hu & Li 2021 Fig.4）
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import xarray as xr

# === Paths ===
PHASE_SPEED_CSV = r"E:\Datas\Derived\phase_speed_q_events.csv"
TILT_Q_NC = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
OUT_PNG = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\phase_speed_q\tilt_q_vs_phase_speed.png"


def compute_event_mean_tilt():
    ds = xr.open_dataset(TILT_Q_NC)
    df_ev = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    times = pd.to_datetime(ds["time"].values)
    vals = ds["tilt_q"].values.astype(float)

    rows = []
    for _, r in df_ev.iterrows():
        ts = np.datetime64(r["start_date"])
        te = np.datetime64(r["end_date"])
        mask = (times >= ts) & (times <= te)
        v = vals[mask]
        v = v[np.isfinite(v)]
        rows.append({
            "event_id": r["event_id"],
            "mean_tilt_q": np.mean(v) if len(v) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    df_ps = pd.read_csv(PHASE_SPEED_CSV)
    df_tilt = compute_event_mean_tilt()
    df = df_ps.merge(df_tilt, on="event_id")

    x = df["phase_speed_m_s"].values
    y = df["mean_tilt_q"].values
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
    r = r_val

    # === Plot ===
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(x, y, c="black", s=25, zorder=3)

    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r-", linewidth=2, zorder=2)

    ax.set_xlabel("Speed (m/s)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Tilting Intensity Index", fontsize=14, fontweight="bold")

    # Cor in top-right
    ax.text(0.95, 0.98, f"Cor={r:.2f}",
            transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="top", ha="right")

    # Tick style (inward, all four sides)
    ax.tick_params(axis="both", which="major", labelsize=12,
                   direction="in", top=True, right=True, length=6)
    ax.tick_params(axis="both", which="minor",
                   direction="in", top=True, right=True, length=3)
    ax.minorticks_on()

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Caption below figure (multi-line)
    sig = "99%" if p_val < 0.01 else ("95%" if p_val < 0.05 else "not sig.")
    caption = (
        f"Scatter diagram of tilt_q (y) vs.\n"
        f"phase speed (x) for {len(x)} MJO events.\n"
        f"Red line: least squares fit.\n"
        f"Cor = {r:.2f} (p = {p_val:.4f}),\n"
        f"exceeding the {sig} confidence level."
    )
    fig.text(0.5, -0.10, caption, ha="center", fontsize=10, style="italic")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Cor={r:.2f}, p={p_val:.4f}, N={len(x)}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
