# -*- coding: utf-8 -*-
"""
06h_real_background_eof_mse.py
三合一分析:
  1) 真实ERA5背景场(非MJO重构)做多元回归与MJO重构对比
  2) EOF降维替代区域平均
  3) MSE收支分解: -u·∂MSE/∂x, -w·∂MSE/∂p
输出: real_background/, eof_regression/, mse_budget/
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.interpolate import interp1d
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

DAILY_DIR = Path(r"E:\Datas\ERA5\raw\pressure_level\era5_pl_mean_quvwT")
DERIVED_DIR = Path(r"E:\Datas\Derived")
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")

# Surface data directories
SL_DIR = Path(r"E:\Datas\ERA5\raw\single_level\daily_mean")
SST_DIR = Path(r"E:\Datas\ERA5\raw\single_level\sst_daily")

CP = 1004.0; LV = 2.501e6; G = 9.81; R_EARTH = 6.371e6
GROUP_SIGMA = 0.7; FDR_ALPHA = 0.05
LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}


# ====== LOAD REAL ERA5 BACKGROUND ======
def load_era5_dailymean(start_date, end_date):
    """Load ERA5 daily mean data for a date range, lat-averaged → (n_days, 9levels, 144lons)."""
    sd, ed = pd.Timestamp(start_date), pd.Timestamp(end_date)
    months_needed = pd.date_range(sd.to_period('M').to_timestamp(),
                                  ed.to_period('M').to_timestamp(), freq='MS')
    all_u, all_q, all_t, all_w, all_time = [], [], [], [], []
    for m in months_needed:
        fname = DAILY_DIR / f"era5_pl_dailymean_quvwT_{m.strftime('%Y%m')}.nc"
        if not fname.exists():
            continue
        ds = xr.open_dataset(fname)
        time_var = 'time' if 'time' in ds.dims else 'valid_time'
        times = pd.to_datetime(ds[time_var].values)
        mask = (times >= sd) & (times <= ed)
        if mask.sum() == 0:
            ds.close()
            continue
        # Lat-average (all latitudes in file are equatorial band)
        u = ds['u'].values[mask].mean(axis=2)   # (days, levels, lons)
        q = ds['q'].values[mask].mean(axis=2)
        t = ds['t'].values[mask].mean(axis=2)
        w = ds['w'].values[mask].mean(axis=2)
        all_u.append(u); all_q.append(q); all_t.append(t); all_w.append(w)
        all_time.extend(times[mask].tolist())
        ds.close()
    if not all_u:
        return None, None, None, None, None
    return (np.concatenate(all_u), np.concatenate(all_q),
            np.concatenate(all_t), np.concatenate(all_w),
            pd.DatetimeIndex(all_time))


def compute_real_bg_predictors(events):
    """For each event, compute scalar predictors from REAL ERA5 background."""
    n_ev = len(events)
    result = pd.DataFrame(index=range(n_ev))

    # Get lons from first file
    ds0 = xr.open_dataset(DAILY_DIR / "era5_pl_dailymean_quvwT_197901.nc")
    lons = ds0['longitude'].values
    levels = ds0['pressure_level'].values
    ds0.close()
    lon_mask = (lons >= 60) & (lons <= 150)
    dlon = np.abs(lons[1] - lons[0])
    dx_m = dlon * np.pi / 180 * R_EARTH

    idx_200 = np.argmin(np.abs(levels - 200))
    idx_850 = np.argmin(np.abs(levels - 850))
    low_levels = (levels >= 500) & (levels <= 1000)

    for i, (_, ev) in enumerate(events.iterrows()):
        u, q, t, w, times = load_era5_dailymean(ev["start_date"], ev["end_date"])
        if u is None:
            continue
        # Event means
        u_mean = np.nanmean(u, axis=0)  # (9, 144)
        q_mean = np.nanmean(q, axis=0)
        t_mean = np.nanmean(t, axis=0)
        w_mean = np.nanmean(w, axis=0)

        result.loc[i, "real_u200"] = np.nanmean(u_mean[idx_200, lon_mask])
        result.loc[i, "real_u850"] = np.nanmean(u_mean[idx_850, lon_mask])

        # Real advection
        dq_dx = np.gradient(q_mean, dx_m, axis=-1)
        adv = -u_mean * dq_dx
        result.loc[i, "real_advection"] = np.nanmean(adv[low_levels][:, lon_mask])

        # Column q and MSE
        sort_idx = np.argsort(levels)
        levels_Pa = levels[sort_idx] * 100.0
        col_q = np.abs(np.trapz(q_mean[sort_idx], x=levels_Pa, axis=0)) / G
        mse = CP * t_mean + LV * q_mean
        col_mse = np.abs(np.trapz(mse[sort_idx], x=levels_Pa, axis=0)) / G
        result.loc[i, "real_column_q"] = np.nanmean(col_q[lon_mask])
        result.loc[i, "real_column_mse"] = np.nanmean(col_mse[lon_mask])

        # MSE budget terms (column-averaged over warm pool)
        # Horizontal: -u · ∂MSE/∂x
        dmse_dx = np.gradient(mse, dx_m, axis=-1)
        h_adv = -u_mean * dmse_dx
        result.loc[i, "mse_hadv"] = np.nanmean(h_adv[low_levels][:, lon_mask])

        # Vertical: -w · ∂MSE/∂p (finite diff in pressure)
        dp = np.gradient(levels * 100.0)  # Pa
        dmse_dp = np.gradient(mse, axis=0) / dp[:, None]
        v_adv = -w_mean * dmse_dp
        result.loc[i, "mse_vadv"] = np.nanmean(v_adv[low_levels][:, lon_mask])

        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_ev} events processed")

    # ===== Surface flux terms =====
    print("    Loading surface flux data ...")
    sl_files = sorted(SL_DIR.glob("era5_sl_dailymean_*.nc"))
    all_lhf, all_shf, all_qrad, all_sl_t = [], [], [], []
    for f in sl_files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        lhf = -np.nanmean(ds['slhf'].values, axis=1)  # upward positive
        shf = -np.nanmean(ds['sshf'].values, axis=1)
        qrad = np.nanmean(ds['ssr'].values + ds['str'].values, axis=1)
        lons_sl = ds['longitude'].values
        all_lhf.append(lhf); all_shf.append(shf); all_qrad.append(qrad)
        all_sl_t.append(pd.to_datetime(ds[tdim].values))
        ds.close()
    lhf_arr = np.concatenate(all_lhf, axis=0)
    shf_arr = np.concatenate(all_shf, axis=0)
    qrad_arr = np.concatenate(all_qrad, axis=0)
    sl_t = pd.DatetimeIndex(np.concatenate(all_sl_t))
    lon_mask_sl = (lons_sl >= 60) & (lons_sl <= 150)

    for i, (_, ev) in enumerate(events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        m = (sl_t >= ts) & (sl_t <= te)
        if m.sum() > 0:
            result.loc[i, "sfc_lhf"] = np.nanmean(lhf_arr[m][:, lon_mask_sl])
            result.loc[i, "sfc_shf"] = np.nanmean(shf_arr[m][:, lon_mask_sl])
            result.loc[i, "sfc_qrad"] = np.nanmean(qrad_arr[m][:, lon_mask_sl])

    return result


# ====== EOF on 2D correlation field ======
def compute_eof_predictors(events, n_eof=3):
    """EOF decomposition of event-mean u field in MJO-significant region."""
    # Load recon u
    ds = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_u_norm_1979-2022.nc")
    if "pressure_level" in ds.dims:
        ds = ds.rename({"pressure_level": "level"})
    da = ds["u_mjo_recon_norm"]
    time_all = pd.to_datetime(da["time"].values)
    levels = da["level"].values
    lons = da["lon"].values
    data = da.values
    ds.close()

    # Focus on warm pool region (40-160°E) and all levels
    lon_mask = (lons >= 40) & (lons <= 160)
    lon_idx = np.where(lon_mask)[0]
    nL, nX = len(levels), len(lon_idx)

    # Compute event means
    n_ev = len(events)
    ev_means = np.full((n_ev, nL * nX), np.nan, dtype=np.float32)

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_all >= pd.Timestamp(ev["start_date"])) & \
               (time_all <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        field = np.nanmean(data[mask], axis=0)[:, lon_idx]  # (nL, nX_subset)
        ev_means[i] = field.flatten()

    # Remove events with NaN
    valid = np.all(np.isfinite(ev_means), axis=1)
    X = ev_means[valid]

    # Standardize columns
    col_mean = np.mean(X, axis=0)
    col_std = np.std(X, axis=0)
    col_std[col_std == 0] = 1
    X_z = (X - col_mean) / col_std

    # SVD-based EOF
    U, S, Vt = np.linalg.svd(X_z, full_matrices=False)
    explained = (S ** 2) / np.sum(S ** 2) * 100

    # PC scores for ALL events (project back)
    all_pcs = np.full((n_ev, n_eof), np.nan)
    for i in range(n_ev):
        if valid[i]:
            x_z = (ev_means[i] - col_mean) / col_std
            for k in range(n_eof):
                all_pcs[i, k] = np.dot(x_z, Vt[k])

    print(f"  EOF explained variance: " + ", ".join(f"PC{k+1}={explained[k]:.1f}%" for k in range(n_eof)))
    return all_pcs, explained[:n_eof]


# ====== PLOTTING ======
def plot_regression_comparison(recon_df, real_df, phase_speed, out_dir):
    """Compare R² and individual r between recon and real background."""
    out_dir.mkdir(parents=True, exist_ok=True)

    recon_vars = ["bg_u200", "bg_u850", "bg_advection", "column_q", "column_mse"]
    real_vars = ["real_u200", "real_u850", "real_advection", "real_column_q", "real_column_mse"]
    labels = ["u₂₀₀", "u₈₅₀", "\u2212u\u00b7\u2202q/\u2202x", "Col q", "Col MSE"]

    # Individual r comparison
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    recon_r, real_r = [], []
    for rv, rr in zip(recon_vars, real_vars):
        x_rc = recon_df[rv].values.astype(float) if rv in recon_df.columns else np.full(len(phase_speed), np.nan)
        x_rl = real_df[rr].values.astype(float) if rr in real_df.columns else np.full(len(phase_speed), np.nan)
        ok_rc = np.isfinite(x_rc) & np.isfinite(phase_speed)
        ok_rl = np.isfinite(x_rl) & np.isfinite(phase_speed)
        r_rc = stats.pearsonr(x_rc[ok_rc], phase_speed[ok_rc])[0] if ok_rc.sum() >= 10 else 0
        r_rl = stats.pearsonr(x_rl[ok_rl], phase_speed[ok_rl])[0] if ok_rl.sum() >= 10 else 0
        recon_r.append(r_rc)
        real_r.append(r_rl)

    x_pos = np.arange(len(labels))
    ax.bar(x_pos - 0.18, recon_r, 0.35, label="MJO Recon", color='#2196F3', alpha=0.85)
    ax.bar(x_pos + 0.18, real_r, 0.35, label="Real ERA5", color='#FF5722', alpha=0.85)
    for i, (rc, rl) in enumerate(zip(recon_r, real_r)):
        ax.text(i - 0.18, rc + 0.01 * np.sign(rc), f"{rc:.3f}", ha='center', fontsize=8)
        ax.text(i + 0.18, rl + 0.01 * np.sign(rl), f"{rl:.3f}", ha='center', fontsize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.axhline(0, color='gray', lw=0.8)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title("Individual r: MJO Recon vs Real ERA5 Background", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "compare_individual_r.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: compare_individual_r.png")

    # Full regression comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for src, df, vars_list, color, label in [
        ("recon", recon_df, recon_vars, '#2196F3', "MJO Recon"),
        ("real", real_df, real_vars, '#FF5722', "Real ERA5"),
    ]:
        X = df[vars_list].values.astype(float)
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(phase_speed)
        if ok.sum() < 10:
            continue
        X_v, y_v = X[ok], phase_speed[ok]
        X_z = (X_v - X_v.mean(0)) / (X_v.std(0) + 1e-12)
        X_d = np.column_stack([np.ones(len(X_z)), X_z])
        beta = np.linalg.lstsq(X_d, y_v, rcond=None)[0]
        y_pred = X_d @ beta
        ss_res = np.sum((y_v - y_pred) ** 2)
        ss_tot = np.sum((y_v - y_v.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        ax.scatter(y_v, y_pred, s=25, alpha=0.5, label=f"{label} R²={r2:.3f}", color=color)

    mn = min(phase_speed[np.isfinite(phase_speed)].min(), 2)
    mx = max(phase_speed[np.isfinite(phase_speed)].max(), 8)
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel("Actual Phase Speed (m/s)", fontsize=11)
    ax.set_ylabel("Predicted Phase Speed (m/s)", fontsize=11)
    ax.set_title("Multiple Regression: Recon vs Real ERA5", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "compare_regression_r2.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: compare_regression_r2.png")


def plot_eof_regression(pcs, explained, phase_speed, out_dir):
    """EOF-based regression and scatter plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_eof = pcs.shape[1]

    # Individual scatter for each PC
    fig, axes = plt.subplots(1, n_eof, figsize=(5 * n_eof, 4.5), dpi=150)
    if n_eof == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        ok = np.isfinite(pcs[:, k]) & np.isfinite(phase_speed)
        if ok.sum() < 10:
            continue
        x, y = pcs[ok, k], phase_speed[ok]
        ax.scatter(x, y, s=25, alpha=0.6, edgecolors='k', linewidths=0.3)
        r, p = stats.pearsonr(x, y)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        xline = np.linspace(x.min(), x.max(), 50)
        ax.plot(xline, slope * xline + intercept, 'r-', lw=2)
        sig = "*" if p < 0.05 else ""
        ax.set_title(f"PC{k+1} ({explained[k]:.1f}%)\nr={r:.3f}{sig}, p={p:.3f}", fontsize=10)
        ax.set_xlabel(f"PC{k+1} Score", fontsize=10)
        ax.set_ylabel("Phase Speed (m/s)" if k == 0 else "", fontsize=10)
        ax.grid(alpha=0.3)
    fig.suptitle("EOF(u) PCs vs Phase Speed", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "eof_pc_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: eof_pc_scatter.png")

    # EOF regression R²
    ok = np.all(np.isfinite(pcs), axis=1) & np.isfinite(phase_speed)
    X = pcs[ok]
    y = phase_speed[ok]
    X_d = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(X_d, y, rcond=None)[0]
    y_pred = X_d @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n_p = X.shape[1]
    r2_adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_p - 1)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(y, y_pred, s=25, alpha=0.6, edgecolors='k', linewidths=0.3)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
    ax.set_xlabel("Actual Phase Speed (m/s)", fontsize=11)
    ax.set_ylabel("Predicted Phase Speed (m/s)", fontsize=11)
    ax.set_title(f"EOF Regression (PC1-{n_eof})\nR²={r2:.3f}, Adj R²={r2_adj:.3f}, N={len(y)}",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(out_dir / "eof_regression_r2.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: eof_regression_r2.png (R²={r2:.3f})")


def plot_mse_budget(real_df, phase_speed, out_dir):
    """MSE budget terms correlation with phase speed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    terms = ["mse_hadv", "mse_vadv", "real_advection", "sfc_lhf", "sfc_shf", "sfc_qrad"]
    labels = ["\u2212u\u00b7\u2202MSE/\u2202x\n(Horizontal)", "\u2212w\u00b7\u2202MSE/\u2202p\n(Vertical)",
              "\u2212u\u00b7\u2202q/\u2202x\n(Moisture Adv)",
              "LHF\n(Source)", "SHF\n(Source)", "Q_rad\n(Source)"]
    colors_pos = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4']

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    r_vals, p_vals = [], []
    for term in terms:
        x = real_df[term].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(phase_speed)
        if ok.sum() >= 10:
            r, p = stats.pearsonr(x[ok], phase_speed[ok])
        else:
            r, p = 0, 1
        r_vals.append(r)
        p_vals.append(p)

    bars = ax.bar(range(len(terms)), r_vals, color=colors_pos, alpha=0.85,
                  edgecolor='k', linewidth=0.5)
    for i, (r, p) in enumerate(zip(r_vals, p_vals)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(i, r + 0.01 * np.sign(r), f"r={r:.3f}{sig}", ha='center', fontsize=10)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Pearson r with Phase Speed", fontsize=11)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_title("MSE Budget Terms vs Phase Speed (Real ERA5)", fontsize=12, fontweight="bold")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mse_budget_corr.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: mse_budget_corr.png")

    # Scatter for each term
    fig, axes = plt.subplots(1, len(terms), figsize=(5 * len(terms), 4.5), dpi=150, sharey=True)
    for ax, term, label, color in zip(axes, terms, labels, colors_pos):
        x = real_df[term].values.astype(float)
        ok = np.isfinite(x) & np.isfinite(phase_speed)
        if ok.sum() < 5:
            continue
        ax.scatter(x[ok], phase_speed[ok], s=25, alpha=0.6, color=color,
                   edgecolors='k', linewidths=0.3)
        r, p = stats.pearsonr(x[ok], phase_speed[ok])
        slope, intercept, _, _, _ = stats.linregress(x[ok], phase_speed[ok])
        xline = np.linspace(x[ok].min(), x[ok].max(), 50)
        ax.plot(xline, slope * xline + intercept, 'r-', lw=2)
        sig = "*" if p < 0.05 else ""
        ax.set_title(f"{label.replace(chr(10),' ')}\nr={r:.3f}{sig}", fontsize=10)
        ax.set_xlabel("Term value", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Phase Speed (m/s)", fontsize=11)
    fig.suptitle("MSE Budget Scatter (Real ERA5)", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "mse_budget_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: mse_budget_scatter.png")


# ====== MAIN ======
def main():
    print("=" * 70)
    print("06h: Real Background + EOF + MSE Budget")
    print("=" * 70)

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)
    print(f"  Events: {len(merged)}")

    # ===== Part 1: Real ERA5 background =====
    print("\n--- Part 1: Real ERA5 Background ---")
    real_df = compute_real_bg_predictors(merged)
    print(f"  Real predictors computed")

    # Load recon predictors from 06g for comparison
    # (recompute quickly)
    from importlib import import_module
    import sys
    sys.path.insert(0, str(Path(r"e:\Projects\ENSO_MJO_Tilt\src")))
    try:
        mod = import_module("06g_regional_regression")
        ds3 = xr.open_dataset(STEP3_NC)
        center_lon_all = ds3["center_lon_track"].values.astype(float)
        amp_all = ds3["amp"].values.astype(float)
        time_step3 = pd.to_datetime(ds3["time"].values)
        ds3.close()
        recon_df = mod.compute_event_predictors(merged, center_lon_all, time_step3, amp_all)
    except Exception as e:
        print(f"  Warning: Could not load recon predictors: {e}")
        recon_df = pd.DataFrame()

    out_real = FIG_DIR / "real_background"
    plot_regression_comparison(recon_df, real_df, phase_speed, out_real)

    # ===== Part 2: EOF regression =====
    print("\n--- Part 2: EOF Regression ---")
    pcs, explained = compute_eof_predictors(merged, n_eof=3)
    out_eof = FIG_DIR / "eof_regression"
    plot_eof_regression(pcs, explained, phase_speed, out_eof)

    # ===== Part 3: MSE Budget =====
    print("\n--- Part 3: MSE Budget ---")
    out_mse = FIG_DIR / "mse_budget"
    plot_mse_budget(real_df, phase_speed, out_mse)

    # ===== Summary =====
    print("\n--- Summary ---")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    methods = ["Recon\n(scalar avg)", "Real ERA5\n(scalar avg)", "EOF\n(PC1-3)"]
    r2_vals = []

    # Recon R²
    if not recon_df.empty:
        rv = ["bg_u200", "bg_u850", "bg_advection", "column_q", "column_mse"]
        X = recon_df[rv].values.astype(float)
        ok = np.all(np.isfinite(X), axis=1) & np.isfinite(phase_speed)
        if ok.sum() > 10:
            X_v, y_v = X[ok], phase_speed[ok]
            X_z = (X_v - X_v.mean(0)) / (X_v.std(0) + 1e-12)
            X_d = np.column_stack([np.ones(len(X_z)), X_z])
            b = np.linalg.lstsq(X_d, y_v, rcond=None)[0]
            r2_vals.append(1 - np.sum((y_v - X_d @ b) ** 2) / np.sum((y_v - y_v.mean()) ** 2))
        else:
            r2_vals.append(0)
    else:
        r2_vals.append(0)

    # Real ERA5 R²
    rv2 = ["real_u200", "real_u850", "real_advection", "real_column_q", "real_column_mse"]
    X2 = real_df[rv2].values.astype(float)
    ok2 = np.all(np.isfinite(X2), axis=1) & np.isfinite(phase_speed)
    if ok2.sum() > 10:
        X_v2, y_v2 = X2[ok2], phase_speed[ok2]
        X_z2 = (X_v2 - X_v2.mean(0)) / (X_v2.std(0) + 1e-12)
        X_d2 = np.column_stack([np.ones(len(X_z2)), X_z2])
        b2 = np.linalg.lstsq(X_d2, y_v2, rcond=None)[0]
        r2_vals.append(1 - np.sum((y_v2 - X_d2 @ b2) ** 2) / np.sum((y_v2 - y_v2.mean()) ** 2))
    else:
        r2_vals.append(0)

    # EOF R²
    ok3 = np.all(np.isfinite(pcs), axis=1) & np.isfinite(phase_speed)
    if ok3.sum() > 10:
        X3, y3 = pcs[ok3], phase_speed[ok3]
        X_d3 = np.column_stack([np.ones(len(X3)), X3])
        b3 = np.linalg.lstsq(X_d3, y3, rcond=None)[0]
        r2_vals.append(1 - np.sum((y3 - X_d3 @ b3) ** 2) / np.sum((y3 - y3.mean()) ** 2))
    else:
        r2_vals.append(0)

    colors = ['#2196F3', '#FF5722', '#4CAF50']
    ax.bar(range(len(methods)), r2_vals, color=colors, alpha=0.85, edgecolor='k', linewidth=0.5)
    for i, v in enumerate(r2_vals):
        ax.text(i, v + 0.005, f"R²={v:.3f}", ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Regression R² Comparison: Methods for Predicting Phase Speed",
                 fontsize=13, fontweight="bold")
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(r2_vals) * 1.3 + 0.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "real_background" / "summary_r2_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: summary_r2_comparison.png")

    print(f"\nAll done!")


if __name__ == "__main__":
    main()
