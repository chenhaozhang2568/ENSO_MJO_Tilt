# -*- coding: utf-8 -*-
"""
06k_sst_lhf_causal.py
SST-LHF 因果链分析：偏相关、中介效应、路径系数
输出到 sst_lhf_causal/ 文件夹
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

DERIVED_DIR = Path(r"E:\Datas\Derived")
SL_DIR = Path(r"E:\Datas\ERA5\raw\single_level\daily_mean")
SST_DIR = Path(r"E:\Datas\ERA5\raw\single_level\sst_daily")
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")
OUT_DIR = FIG_DIR / "sst_lhf_causal"

GROUP_SIGMA = 0.7
LON_WEST, LON_EAST = 60, 150  # warm pool


def _load_event_scalars(events):
    """Load event-mean warm-pool-averaged LHF, SST, q for each event."""
    n_ev = len(events)
    result = pd.DataFrame(index=range(n_ev),
                          columns=["bg_lhf", "bg_sst", "bg_precip"])

    # Load surface data
    sl_files = sorted(SL_DIR.glob("era5_sl_dailymean_*.nc"))
    sst_files = sorted(SST_DIR.glob("era5_sst_dailymean_*.nc"))

    # Build time-indexed arrays (lat-averaged)
    all_lhf, all_tp, all_t = [], [], []
    for f in sl_files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        lhf = -np.nanmean(ds['slhf'].values, axis=1)  # upward positive
        tp = np.nanmean(ds['tp'].values, axis=1)
        lons = ds['longitude'].values
        all_lhf.append(lhf)
        all_tp.append(tp)
        all_t.append(pd.to_datetime(ds[tdim].values))
        ds.close()

    lhf_arr = np.concatenate(all_lhf, axis=0)
    tp_arr = np.concatenate(all_tp, axis=0)
    sl_t = pd.DatetimeIndex(np.concatenate(all_t))

    all_sst, all_sst_t = [], []
    for f in sst_files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        sst_val = np.nanmean(ds['sst'].values, axis=1)
        all_sst.append(sst_val)
        all_sst_t.append(pd.to_datetime(ds[tdim].values))
        ds.close()

    sst_arr = np.concatenate(all_sst, axis=0)
    sst_t = pd.DatetimeIndex(np.concatenate(all_sst_t))

    lon_mask = (lons >= LON_WEST) & (lons <= LON_EAST)

    for i, (_, ev) in enumerate(events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        m = (sl_t >= ts) & (sl_t <= te)
        if m.sum() > 0:
            result.loc[i, "bg_lhf"] = np.nanmean(lhf_arr[m][:, lon_mask])
            result.loc[i, "bg_precip"] = np.nanmean(tp_arr[m][:, lon_mask])
        m2 = (sst_t >= ts) & (sst_t <= te)
        if m2.sum() > 0:
            result.loc[i, "bg_sst"] = np.nanmean(sst_arr[m2][:, lon_mask])

    # Also load column q from recon
    ds_q = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_q_norm_1979-2022.nc")
    if "pressure_level" in ds_q.dims:
        ds_q = ds_q.rename({"pressure_level": "level"})
    da_q = ds_q['q_mjo_recon_norm']
    time_q = pd.to_datetime(da_q['time'].values)
    levels = da_q['level'].values.astype(float)
    lons_q = da_q['lon'].values
    lon_mask_q = (lons_q >= LON_WEST) & (lons_q <= LON_EAST)

    sort_idx = np.argsort(levels)
    levels_Pa = levels[sort_idx] * 100.0

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_q >= pd.Timestamp(ev["start_date"])) & \
               (time_q <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        q_mean = np.nanmean(da_q.values[mask], axis=0)  # (level, lon)
        q_sorted = q_mean[sort_idx, :]
        col_q = np.abs(np.trapz(q_sorted, x=levels_Pa, axis=0)) / 9.81
        result.loc[i, "col_q"] = np.nanmean(col_q[lon_mask_q])

    ds_q.close()
    return result.astype(float)


def partial_corr(x, y, z):
    """Partial correlation between x and y, controlling for z."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 10:
        return np.nan, np.nan
    x, y, z = x[ok], y[ok], z[ok]
    # Residualize
    b_xz = np.polyfit(z, x, 1)
    b_yz = np.polyfit(z, y, 1)
    rx = x - np.polyval(b_xz, z)
    ry = y - np.polyval(b_yz, z)
    return stats.pearsonr(rx, ry)


def plot_partial_corr_table(scalars, phase_speed, out_path):
    """Plot partial correlation table."""
    sst = scalars["bg_sst"].values
    lhf = scalars["bg_lhf"].values
    colq = scalars["col_q"].values
    prec = scalars["bg_precip"].values
    ps = phase_speed

    pairs = [
        ("SST vs Speed", "control LHF", sst, ps, lhf),
        ("LHF vs Speed", "control SST", lhf, ps, sst),
        ("SST vs Speed", "control Col q", sst, ps, colq),
        ("LHF vs Speed", "control Col q", lhf, ps, colq),
        ("SST vs LHF", "control Speed", sst, lhf, ps),
        ("Col q vs Speed", "control SST", colq, ps, sst),
        ("Col q vs Speed", "control LHF", colq, ps, lhf),
    ]

    # Also compute zero-order correlations
    zero = [
        ("SST vs Speed", "", sst, ps),
        ("LHF vs Speed", "", lhf, ps),
        ("Col q vs Speed", "", colq, ps),
        ("Precip vs Speed", "", prec, ps),
        ("SST vs LHF", "", sst, lhf),
    ]

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    ax.axis('off')

    rows = []
    # Zero-order
    for name, _, x, y in zero:
        ok = np.isfinite(x) & np.isfinite(y)
        r, p = stats.pearsonr(x[ok], y[ok])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        rows.append([name, "—", f"{r:.3f}", f"{p:.4f}", sig])

    # Divider
    rows.append(["", "", "", "", ""])

    # Partial
    for name, ctrl, x, y, z in pairs:
        r, p = partial_corr(x, y, z)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        rows.append([name, ctrl, f"{r:.3f}", f"{p:.4f}", sig])

    col_labels = ["Correlation", "Controlled", "r", "p-value", "Sig"]
    table = ax.table(cellText=rows, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    # Style divider
    div_row = len(zero) + 1
    for j in range(len(col_labels)):
        table[div_row, j].set_facecolor('#D9E2F3')

    ax.set_title("Zero-Order & Partial Correlations: SST–LHF–q–Speed",
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_sst_vs_lhf_scatter(scalars, phase_speed, fast_mask, slow_mask, out_path):
    """SST vs LHF scatter, colored by Fast/Slow."""
    sst = scalars["bg_sst"].values
    lhf = scalars["bg_lhf"].values
    ok = np.isfinite(sst) & np.isfinite(lhf)
    r_all, p_all = stats.pearsonr(sst[ok], lhf[ok])

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    # Neutral
    neutral = ~fast_mask & ~slow_mask & ok
    ax.scatter(sst[neutral], lhf[neutral], c='gray', alpha=0.4, s=30, label='Neutral')
    ax.scatter(sst[fast_mask & ok], lhf[fast_mask & ok], c='tab:red', s=50,
               edgecolors='k', lw=0.5, label=f'Fast (N={fast_mask.sum()})')
    ax.scatter(sst[slow_mask & ok], lhf[slow_mask & ok], c='tab:blue', s=50,
               edgecolors='k', lw=0.5, label=f'Slow (N={slow_mask.sum()})')
    # Regression line
    z = np.polyfit(sst[ok], lhf[ok], 1)
    xx = np.linspace(np.nanmin(sst[ok]), np.nanmax(sst[ok]), 50)
    ax.plot(xx, np.polyval(z, xx), 'k--', lw=1.5, alpha=0.6)

    ax.set_xlabel("SST (K)", fontsize=12)
    ax.set_ylabel("LHF (W/m²)", fontsize=12)
    ax.set_title(f"SST vs LHF (warm pool avg)\nr={r_all:.3f}, p={p_all:.4f}",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_mediation(scalars, phase_speed, out_path):
    """Mediation analysis: SST → LHF → Speed."""
    sst = scalars["bg_sst"].values
    lhf = scalars["bg_lhf"].values
    ps = phase_speed
    ok = np.isfinite(sst) & np.isfinite(lhf) & np.isfinite(ps)

    # Standardize
    sst_s = (sst[ok] - np.mean(sst[ok])) / np.std(sst[ok])
    lhf_s = (lhf[ok] - np.mean(lhf[ok])) / np.std(lhf[ok])
    ps_s = (ps[ok] - np.mean(ps[ok])) / np.std(ps[ok])

    # Path a: SST → LHF
    a = np.polyfit(sst_s, lhf_s, 1)[0]
    # Path b: LHF → Speed (controlling SST)
    X = np.column_stack([sst_s, lhf_s])
    betas = np.linalg.lstsq(X, ps_s, rcond=None)[0]
    c_prime = betas[0]  # direct: SST → Speed
    b = betas[1]         # LHF → Speed (controlling SST)
    # Total: SST → Speed
    c = np.polyfit(sst_s, ps_s, 1)[0]
    indirect = a * b

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.axis('off')

    # Draw paths as text diagram
    txt = (
        f"SST  ──── a={a:.3f} ────→  LHF  ──── b={b:.3f} ────→  Speed\n"
        f"                                               ↑\n"
        f"SST  ──── c'={c_prime:.3f} ──────────────────→  Speed\n\n"
        f"Total effect (c) = {c:.3f}\n"
        f"Direct effect (c') = {c_prime:.3f}\n"
        f"Indirect effect (a×b) = {indirect:.3f}\n"
        f"Mediation ratio = {abs(indirect/c)*100:.1f}%"
    )
    ax.text(0.5, 0.5, txt, transform=ax.transAxes, fontsize=13,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.set_title("Mediation Analysis: SST → LHF → Phase Speed",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_causal_path(scalars, phase_speed, out_path):
    """Path diagram with all standardized coefficients."""
    sst = scalars["bg_sst"].values
    lhf = scalars["bg_lhf"].values
    colq = scalars["col_q"].values
    ps = phase_speed
    ok = np.isfinite(sst) & np.isfinite(lhf) & np.isfinite(colq) & np.isfinite(ps)

    # Standardize
    def std(x): return (x - np.mean(x)) / np.std(x)
    sst_s, lhf_s, q_s, ps_s = std(sst[ok]), std(lhf[ok]), std(colq[ok]), std(ps[ok])

    # Compute all path coefficients
    paths = {}
    paths["SST→LHF"] = np.polyfit(sst_s, lhf_s, 1)[0]
    paths["SST→q"] = np.polyfit(sst_s, q_s, 1)[0]
    paths["LHF→q"] = np.polyfit(lhf_s, q_s, 1)[0]

    # Multiple regression: Speed = f(SST, LHF, q)
    X = np.column_stack([sst_s, lhf_s, q_s])
    betas = np.linalg.lstsq(X, ps_s, rcond=None)[0]
    paths["SST→Speed"] = betas[0]
    paths["LHF→Speed"] = betas[1]
    paths["q→Speed"] = betas[2]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ax.axis('off')

    # Draw as text diagram
    lines = [
        "                 Causal Path Diagram (Standardized β)",
        "",
        f"  ┌─── SST ───────────────────────────────→ Speed",
        f"  │     │          β = {paths['SST→Speed']:.3f}           │",
        f"  │     │                                      │",
        f"  │     │ β={paths['SST→LHF']:.3f}                           │",
        f"  │     ↓                                      │",
        f"  │    LHF ─────────────────────────────────→ Speed",
        f"  │     │          β = {paths['LHF→Speed']:.3f}           │",
        f"  │     │                                      │",
        f"  │     │ β={paths['LHF→q']:.3f}                           │",
        f"  │     ↓                                      │",
        f"  │   Col q ────────────────────────────────→ Speed",
        f"  │              β = {paths['q→Speed']:.3f}                │",
        f"  │                                            │",
        f"  └─── SST → q   β = {paths['SST→q']:.3f}",
    ]
    txt = "\n".join(lines)
    ax.text(0.05, 0.5, txt, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#F0F8FF', alpha=0.9))
    ax.set_title("SST → LHF → q → Phase Speed: Path Coefficients",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def main():
    print("=" * 70)
    print("06k: SST-LHF Causal Chain Analysis")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)

    ps_valid = phase_speed[np.isfinite(phase_speed)]
    mu, sigma = np.mean(ps_valid), np.std(ps_valid)
    fast_mask = phase_speed > mu + GROUP_SIGMA * sigma
    slow_mask = phase_speed < mu - GROUP_SIGMA * sigma
    print(f"  Events: {len(merged)}, Fast={fast_mask.sum()}, Slow={slow_mask.sum()}")

    print("\n  Loading event-level scalars ...")
    scalars = _load_event_scalars(merged)
    print(f"  Scalars loaded: {list(scalars.columns)}")

    print("\n  --- Partial Correlation Table ---")
    plot_partial_corr_table(scalars, phase_speed, OUT_DIR / "partial_corr_table.png")

    print("\n  --- SST vs LHF Scatter ---")
    plot_sst_vs_lhf_scatter(scalars, phase_speed, fast_mask, slow_mask,
                            OUT_DIR / "sst_vs_lhf_scatter.png")

    print("\n  --- Mediation Analysis ---")
    plot_mediation(scalars, phase_speed, OUT_DIR / "mediation_analysis.png")

    print("\n  --- Causal Path Diagram ---")
    plot_causal_path(scalars, phase_speed, OUT_DIR / "causal_path_diagram.png")

    print(f"\nAll done! Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
