# -*- coding: utf-8 -*-
"""
06g_regional_regression.py
1) 按MJO中心经度分洋盆分析关键变量与相速度的关系
2) 多元线性回归量化各因子的独立贡献
输出: longitude_bins/ 和 multivariate_regression/
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
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")
BIN_DIR = FIG_DIR / "longitude_bins"
REG_DIR = FIG_DIR / "multivariate_regression"

# Surface data directories
SL_DIR = Path(r"E:\Datas\ERA5\raw\single_level\daily_mean")
SST_DIR = Path(r"E:\Datas\ERA5\raw\single_level\sst_daily")

AMP_THRESHOLD = 0.5
R_EARTH = 6.371e6

# Longitude bins (only two regions with data)
BINS = {
    "IO":  (40, 100,  "Indian Ocean"),
    "MC":  (100, 150, "Maritime Continent"),
}

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}


# ====== HELPERS ======
def _rename_level(da):
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    return da

def _column_integrate(data_3d, levels_hPa):
    sort_idx = np.argsort(levels_hPa)
    levels_Pa = levels_hPa[sort_idx] * 100.0
    data_sorted = data_3d[:, sort_idx, :]
    return np.abs(np.trapz(data_sorted, x=levels_Pa, axis=1)) / 9.81


# ====== COMPUTE EVENT-LEVEL PREDICTORS ======
def compute_event_predictors(events, center_lon_all, time_step3, amp_all):
    """
    For each event compute scalar predictors:
    - mean_center_lon: mean center longitude
    - bg_u200: event-mean u at 200hPa, averaged over 60-150°E
    - bg_u850: event-mean u at 850hPa, averaged over 60-150°E
    - bg_advection: event-mean -u·dq/dx averaged over 60-150°E, 850-500hPa
    - column_q: event-mean column q, averaged over 60-150°E
    - column_mse: event-mean column MSE, averaged over 60-150°E
    - olr_center: event-mean OLR at center longitude
    """
    n_ev = len(events)
    result = pd.DataFrame(index=range(n_ev))

    # --- Center longitude ---
    for i, (_, ev) in enumerate(events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        mask = (time_step3 >= ts) & (time_step3 <= te) & np.isfinite(center_lon_all)
        if mask.sum() > 0:
            clons = center_lon_all[mask]
            # Handle wrapping
            clons_rad = np.deg2rad(clons)
            result.loc[i, "mean_center_lon"] = np.rad2deg(
                np.arctan2(np.mean(np.sin(clons_rad)), np.mean(np.cos(clons_rad)))) % 360
        else:
            result.loc[i, "mean_center_lon"] = np.nan

    # --- u at specific levels ---
    ds_u = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_u_norm_1979-2022.nc")
    da_u = _rename_level(ds_u["u_mjo_recon_norm"])
    time_u = pd.to_datetime(da_u["time"].values)
    levels = da_u["level"].values
    lons = da_u["lon"].values
    data_u = da_u.values
    ds_u.close()

    lon_mask = (lons >= 60) & (lons <= 150)  # warm pool
    idx_200 = np.argmin(np.abs(levels - 200))
    idx_850 = np.argmin(np.abs(levels - 850))

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_u >= pd.Timestamp(ev["start_date"])) & \
               (time_u <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        u_block = data_u[mask]
        result.loc[i, "bg_u200"] = np.nanmean(u_block[:, idx_200, :][:, lon_mask])
        result.loc[i, "bg_u850"] = np.nanmean(u_block[:, idx_850, :][:, lon_mask])

    # --- q and T for column integration ---
    ds_q = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_q_norm_1979-2022.nc")
    ds_t = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_t_norm_1979-2022.nc")
    da_q = _rename_level(ds_q["q_mjo_recon_norm"])
    da_t = _rename_level(ds_t["t_mjo_recon_norm"])
    data_q = da_q.values
    data_t = da_t.values
    levels_f = da_q["level"].values.astype(float)
    ds_q.close(); ds_t.close()

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_u >= pd.Timestamp(ev["start_date"])) & \
               (time_u <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        q_mean = np.nanmean(data_q[mask], axis=0)  # (level, lon)
        t_mean = np.nanmean(data_t[mask], axis=0)
        col_q = _column_integrate(q_mean[None, :, :], levels_f)[0]  # (lon,)
        mse = 1004.0 * t_mean + 2.501e6 * q_mean
        col_mse = _column_integrate(mse[None, :, :], levels_f)[0]
        result.loc[i, "column_q"] = np.nanmean(col_q[lon_mask])
        result.loc[i, "column_mse"] = np.nanmean(col_mse[lon_mask])

    # --- Advection ---
    dlon = np.abs(lons[1] - lons[0])
    dx_m = dlon * np.pi / 180 * R_EARTH
    low_levels = (levels >= 500) & (levels <= 1000)

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_u >= pd.Timestamp(ev["start_date"])) & \
               (time_u <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        u_b = data_u[mask]
        q_b = data_q[mask]
        dq_dx = np.gradient(q_b, dx_m, axis=-1)
        adv = -u_b * dq_dx
        adv_mean = np.nanmean(adv, axis=0)  # (level, lon)
        result.loc[i, "bg_advection"] = np.nanmean(adv_mean[low_levels][:, lon_mask])

    # --- OLR ---
    ds3 = xr.open_dataset(STEP3_NC)
    olr = ds3['olr_recon'].values.astype(np.float64)
    amp = ds3['amp'].values.astype(np.float64)
    lons_olr = ds3['lon'].values
    time_olr = pd.to_datetime(ds3['time'].values)
    ds3.close()
    amp[amp < AMP_THRESHOLD] = np.nan
    olr_norm = olr / amp[:, None]

    for i, (_, ev) in enumerate(events.iterrows()):
        mask = (time_olr >= pd.Timestamp(ev["start_date"])) & \
               (time_olr <= pd.Timestamp(ev["end_date"]))
        if mask.sum() == 0:
            continue
        olr_mean = np.nanmean(olr_norm[mask], axis=0)
        clon = result.loc[i, "mean_center_lon"]
        if np.isfinite(clon):
            k = np.argmin(np.abs(lons_olr - clon))
            result.loc[i, "olr_center"] = olr_mean[k]
        result.loc[i, "olr_warmpool"] = np.nanmean(
            olr_mean[(lons_olr >= 60) & (lons_olr <= 150)])

    return result


def compute_surface_predictors(events, pred_df):
    """Compute event-level surface predictors from daily_mean and sst_daily."""
    n_ev = len(events)

    # Load all surface data into memory (lat-averaged)
    sl_files = sorted(SL_DIR.glob("era5_sl_dailymean_*.nc"))
    sst_files = sorted(SST_DIR.glob("era5_sst_dailymean_*.nc"))

    # Build time-indexed arrays
    sl_data, sl_times = {}, None
    all_sl, all_sl_t = [], []
    for f in sl_files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        t = pd.to_datetime(ds[tdim].values)
        lhf = -np.nanmean(ds['slhf'].values, axis=1)  # flip sign: upward positive
        shf = -np.nanmean(ds['sshf'].values, axis=1)
        qrad = np.nanmean(ds['ssr'].values + ds['str'].values, axis=1)  # net rad
        tp = np.nanmean(ds['tp'].values, axis=1)
        lons_sl = ds['longitude'].values
        all_sl.append(np.stack([lhf, qrad, tp], axis=1))  # (days, 3, lon)
        all_sl_t.append(t)
        ds.close()

    sl_arr = np.concatenate(all_sl, axis=0)  # (N_days, 3, 144)
    sl_t = pd.DatetimeIndex(np.concatenate(all_sl_t))

    all_sst, all_sst_t = [], []
    for f in sst_files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        sst_val = np.nanmean(ds['sst'].values, axis=1)  # (days, lon)
        all_sst.append(sst_val)
        all_sst_t.append(pd.to_datetime(ds[tdim].values))
        ds.close()

    sst_arr = np.concatenate(all_sst, axis=0)
    sst_t = pd.DatetimeIndex(np.concatenate(all_sst_t))

    lon_mask = (lons_sl >= 60) & (lons_sl <= 150)

    for i, (_, ev) in enumerate(events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        # Surface fluxes
        m = (sl_t >= ts) & (sl_t <= te)
        if m.sum() > 0:
            block = sl_arr[m]  # (days, 3, lon)
            pred_df.loc[i, "bg_lhf"] = np.nanmean(block[:, 0, :][:, lon_mask])
            pred_df.loc[i, "bg_qrad"] = np.nanmean(block[:, 1, :][:, lon_mask])
            pred_df.loc[i, "bg_precip"] = np.nanmean(block[:, 2, :][:, lon_mask])
        # SST
        m2 = (sst_t >= ts) & (sst_t <= te)
        if m2.sum() > 0:
            pred_df.loc[i, "bg_sst"] = np.nanmean(sst_arr[m2][:, lon_mask])

    return pred_df


# ====== PART 1: LONGITUDE BINS ======
def plot_longitude_bins(pred_df, phase_speed):
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    vars_to_plot = ["bg_u200", "bg_u850", "bg_advection", "column_q", "column_mse",
                     "bg_lhf", "bg_sst", "bg_precip"]
    var_labels = {
        "bg_u200": "u₂₀₀ (warm pool avg)",
        "bg_u850": "u₈₅₀ (warm pool avg)",
        "bg_advection": "\u2212u\u00b7\u2202q/\u2202x (low-level, warm pool)",
        "column_q": "Column q (warm pool)",
        "column_mse": "Column MSE (warm pool)",
        "bg_lhf": "LHF (warm pool avg, W/m²)",
        "bg_sst": "SST (warm pool avg, K)",
        "bg_precip": "Precip (warm pool avg, mm/day)",
    }

    # --- Scatter per bin ---
    for vname in vars_to_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, sharey=True)
        for ax, (bk, (lo, hi, bname)) in zip(axes, BINS.items()):
            clon = pred_df["mean_center_lon"].values
            bmask = (clon >= lo) & (clon < hi) & np.isfinite(phase_speed) & \
                    np.isfinite(pred_df[vname].values.astype(float))
            x = pred_df.loc[bmask, vname].values.astype(float)
            y = phase_speed[bmask]
            ax.scatter(x, y, s=30, alpha=0.7, edgecolors='k', linewidths=0.5)
            if len(x) >= 5:
                slope, intercept, r, p, _ = stats.linregress(x, y)
                xline = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                ax.plot(xline, slope * xline + intercept, 'r-', lw=2)
                sig_mark = "*" if p < 0.05 else ""
                ax.set_title(f"{bname} (N={bmask.sum()})\nr={r:.3f}{sig_mark}, p={p:.3f}",
                             fontsize=10)
            else:
                ax.set_title(f"{bname} (N={bmask.sum()})", fontsize=10)
            ax.set_xlabel(var_labels[vname], fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("Phase Speed (m/s)", fontsize=11)
        fig.suptitle(f"Longitude-Bin: {var_labels[vname]} vs Phase Speed",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(BIN_DIR / f"scatter_{vname}.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"    Saved: scatter_{vname}.png")

    # --- Summary bar chart: per-bin correlations ---
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    n_vars = len(vars_to_plot)
    n_bins = len(BINS)
    x_pos = np.arange(n_vars)
    width = 0.35
    colors = ['#2196F3', '#4CAF50']

    for bi, (bk, (lo, hi, bname)) in enumerate(BINS.items()):
        clon = pred_df["mean_center_lon"].values
        bmask = (clon >= lo) & (clon < hi) & np.isfinite(phase_speed)
        r_vals = []
        for vname in vars_to_plot:
            vm = bmask & np.isfinite(pred_df[vname].values.astype(float))
            if vm.sum() >= 10:
                r, _ = stats.pearsonr(pred_df.loc[vm, vname].values.astype(float),
                                      phase_speed[vm])
                r_vals.append(r)
            else:
                r_vals.append(0)
        ax.bar(x_pos + bi * width, r_vals, width, label=f"{bname} (N={bmask.sum()})",
               color=colors[bi], alpha=0.85)

    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels([var_labels[v].split("(")[0].strip() for v in vars_to_plot],
                       fontsize=9, rotation=15)
    ax.set_ylabel("Pearson r", fontsize=11)
    ax.set_title("Correlation with Phase Speed by Region", fontsize=12, fontweight="bold")
    ax.axhline(0, color='gray', lw=0.8)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(BIN_DIR / "summary_regional_corr.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: summary_regional_corr.png")


# ====== PART 2: MULTIVARIATE REGRESSION ======
def run_multivariate_regression(pred_df, phase_speed):
    REG_DIR.mkdir(parents=True, exist_ok=True)
    predictors = ["bg_u200", "bg_u850", "bg_advection", "column_q", "column_mse",
                   "bg_lhf", "bg_sst", "bg_qrad", "bg_precip"]
    pred_labels = {
        "bg_u200": "u₂₀₀",
        "bg_u850": "u₈₅₀",
        "bg_advection": "\u2212u\u00b7\u2202q/\u2202x",
        "column_q": "Col q",
        "column_mse": "Col MSE",
        "bg_lhf": "LHF",
        "bg_sst": "SST",
        "bg_qrad": "Q_rad",
        "bg_precip": "Precip",
    }

    # Build design matrix
    X = pred_df[predictors].values.astype(float)
    y = phase_speed.copy()
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_v, y_v = X[valid], y[valid]
    print(f"  Regression: {valid.sum()} valid events out of {len(y)}")

    # Standardize
    X_mean = np.mean(X_v, axis=0)
    X_std = np.std(X_v, axis=0)
    X_std[X_std == 0] = 1
    X_z = (X_v - X_mean) / X_std
    y_mean = np.mean(y_v)
    y_std = np.std(y_v)

    # --- Individual r values ---
    indiv_r = []
    indiv_p = []
    for j, pname in enumerate(predictors):
        r, p = stats.pearsonr(X_v[:, j], y_v)
        indiv_r.append(r)
        indiv_p.append(p)

    # --- Full multivariate regression ---
    X_design = np.column_stack([np.ones(len(X_z)), X_z])
    beta, residuals, rank, sv = np.linalg.lstsq(X_design, y_v, rcond=None)
    y_pred = X_design @ beta
    ss_res = np.sum((y_v - y_pred) ** 2)
    ss_tot = np.sum((y_v - y_mean) ** 2)
    R2_full = 1 - ss_res / ss_tot
    n, p_num = len(y_v), len(predictors)
    R2_adj = 1 - (1 - R2_full) * (n - 1) / (n - p_num - 1)
    print(f"  Full model R² = {R2_full:.4f}, Adjusted R² = {R2_adj:.4f}")

    # Standard errors and p-values for coefficients
    mse_resid = ss_res / (n - p_num - 1)
    var_beta = mse_resid * np.linalg.inv(X_design.T @ X_design).diagonal()
    se_beta = np.sqrt(np.abs(var_beta))
    t_stat = beta / se_beta
    p_vals_coef = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - p_num - 1))

    # Standardized coefficients (skip intercept)
    std_coefs = beta[1:]
    std_p = p_vals_coef[1:]

    # --- Stepwise R² (cumulative) ---
    # Order predictors by |individual r|
    order = np.argsort(np.abs(indiv_r))[::-1]
    cumul_R2 = []
    cumul_labels = []
    for k in range(1, len(predictors) + 1):
        sel = order[:k]
        X_sub = np.column_stack([np.ones(len(X_z)), X_z[:, sel]])
        b_sub = np.linalg.lstsq(X_sub, y_v, rcond=None)[0]
        y_sub = X_sub @ b_sub
        ss_r = np.sum((y_v - y_sub) ** 2)
        cumul_R2.append(1 - ss_r / ss_tot)
        cumul_labels.append(pred_labels[predictors[order[k - 1]]])

    # ====== PLOTS ======
    # 1. Standardized coefficients
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    labels = [pred_labels[p] for p in predictors]
    colors = ['#2196F3' if c > 0 else '#FF5722' for c in std_coefs]
    bars = ax.barh(range(len(std_coefs)), std_coefs, color=colors, alpha=0.85,
                   edgecolor='k', linewidth=0.5)
    for i, (v, p) in enumerate(zip(std_coefs, std_p)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(v + 0.01 * np.sign(v), i, f"{v:.3f}{sig}", va='center', fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlabel("Standardized Coefficient", fontsize=11)
    ax.set_title(f"Multiple Regression: Phase Speed\nR\u00b2={R2_full:.3f}, "
                 f"Adj R\u00b2={R2_adj:.3f}, N={n}", fontsize=12, fontweight="bold")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(REG_DIR / "coefficients.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: coefficients.png")

    # 2. Individual r vs full model
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.barh(range(len(indiv_r)), indiv_r, color='#607D8B', alpha=0.85,
            edgecolor='k', linewidth=0.5)
    for i, (r, p) in enumerate(zip(indiv_r, indiv_p)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(r + 0.01 * np.sign(r), i, f"r={r:.3f}{sig}", va='center', fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlabel("Pearson r (individual)", fontsize=11)
    ax.set_title(f"Individual Correlations with Phase Speed (N={n})",
                 fontsize=12, fontweight="bold")
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(REG_DIR / "individual_r.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: individual_r.png")

    # 3. Cumulative R²
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(range(len(cumul_R2)), cumul_R2, color='#009688', alpha=0.85,
           edgecolor='k', linewidth=0.5)
    for i, v in enumerate(cumul_R2):
        ax.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)
    ax.set_xticks(range(len(cumul_labels)))
    cum_xlabels = [f"+{cumul_labels[i]}" if i > 0 else cumul_labels[i]
                   for i in range(len(cumul_labels))]
    ax.set_xticklabels(cum_xlabels, fontsize=10, rotation=15)
    ax.set_ylabel("Cumulative R\u00b2", fontsize=11)
    ax.set_title("Stepwise R\u00b2 (adding predictors by |r|)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(REG_DIR / "cumulative_r2.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: cumulative_r2.png")

    # 4. Predicted vs actual scatter
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(y_v, y_pred, s=30, alpha=0.7, edgecolors='k', linewidths=0.5)
    mn, mx = min(y_v.min(), y_pred.min()), max(y_v.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='1:1')
    ax.set_xlabel("Actual Phase Speed (m/s)", fontsize=11)
    ax.set_ylabel("Predicted Phase Speed (m/s)", fontsize=11)
    ax.set_title(f"Predicted vs Actual (R\u00b2={R2_full:.3f})",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(REG_DIR / "predicted_vs_actual.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: predicted_vs_actual.png")


# ====== MAIN ======
def main():
    print("=" * 70)
    print("06g: Regional Analysis & Multivariate Regression")
    print("=" * 70)

    ds3 = xr.open_dataset(STEP3_NC)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    amp_all = ds3["amp"].values.astype(float)
    time_step3 = pd.to_datetime(ds3["time"].values)
    ds3.close()

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)
    print(f"  Events: {len(merged)}")

    print("\n--- Computing event-level predictors ---")
    pred_df = compute_event_predictors(merged, center_lon_all, time_step3, amp_all)
    print(f"  Atmospheric predictors computed for {len(pred_df)} events")

    print("\n--- Computing surface predictors (LHF, SST, Q_rad, Precip) ---")
    pred_df = compute_surface_predictors(merged, pred_df)
    print(f"  Surface predictors added")

    # Check bin distribution
    clon = pred_df["mean_center_lon"].values
    for bk, (lo, hi, bname) in BINS.items():
        n = np.sum((clon >= lo) & (clon < hi) & np.isfinite(phase_speed))
        print(f"    {bname}: N={n}")

    print("\n--- Part 1: Longitude Bins ---")
    plot_longitude_bins(pred_df, phase_speed)

    print("\n--- Part 2: Multivariate Regression ---")
    run_multivariate_regression(pred_df, phase_speed)

    print(f"\nAll done!")


if __name__ == "__main__":
    main()
