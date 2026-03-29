# -*- coding: utf-8 -*-
"""
06l_asymmetry_index.py
MJO 前方-后方不对称性指数：前方(+15°~+45°) vs 后方(-45°~-15°)
评估 SST/LHF/降水/柱q 的前后差异是否预测相速度
输出到 asymmetry/ 文件夹
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
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\field_phase_speed_correlation")
OUT_DIR = FIG_DIR / "asymmetry"

AMP_THRESHOLD = 0.5
GROUP_SIGMA = 0.7
FRONT_RANGE = (15, 45)    # degrees ahead of MJO center
REAR_RANGE = (-45, -15)   # degrees behind MJO center


def _get_segment_mean(data_day, lons, center_lon, lo_rel, hi_rel):
    """Average data in [center+lo_rel, center+hi_rel] longitude range."""
    lon_360 = np.mod(lons, 360)
    c360 = np.mod(center_lon, 360)
    vals = []
    for rl in np.arange(lo_rel, hi_rel + 0.1, np.abs(lons[1] - lons[0])):
        tlon = np.mod(c360 + rl, 360)
        k = np.argmin(np.abs(lon_360 - tlon))
        if np.abs(lon_360[k] - tlon) < 5.0 and np.isfinite(data_day[k]):
            vals.append(data_day[k])
    return np.mean(vals) if vals else np.nan


def compute_asymmetry_index(events, center_lon_all, time_step3, amp_all):
    """For each event, compute front-rear asymmetry of SST, LHF, precip, col_q."""
    n_ev = len(events)

    # Load surface data (lat-averaged)
    sl_files = sorted(SL_DIR.glob("era5_sl_dailymean_*.nc"))
    sst_files = sorted(SST_DIR.glob("era5_sst_dailymean_*.nc"))

    all_lhf, all_tp, all_t = [], [], []
    for f in sl_files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        lhf = -np.nanmean(ds['slhf'].values, axis=1)
        tp = np.nanmean(ds['tp'].values, axis=1)
        lons = ds['longitude'].values
        all_lhf.append(lhf); all_tp.append(tp)
        all_t.append(pd.to_datetime(ds[tdim].values))
        ds.close()
    lhf_arr = np.concatenate(all_lhf, axis=0)
    tp_arr = np.concatenate(all_tp, axis=0)
    sl_t = pd.DatetimeIndex(np.concatenate(all_t))

    all_sst, all_sst_t = [], []
    for f in sst_files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        all_sst.append(np.nanmean(ds['sst'].values, axis=1))
        all_sst_t.append(pd.to_datetime(ds[tdim].values))
        ds.close()
    sst_arr = np.concatenate(all_sst, axis=0)
    sst_t = pd.DatetimeIndex(np.concatenate(all_sst_t))

    # Load column q from recon
    ds_q = xr.open_dataset(DERIVED_DIR / "era5_mjo_recon_q_norm_1979-2022.nc")
    if "pressure_level" in ds_q.dims:
        ds_q = ds_q.rename({"pressure_level": "level"})
    da_q = ds_q['q_mjo_recon_norm']
    time_q = pd.to_datetime(da_q['time'].values)
    levels = da_q['level'].values.astype(float)
    lons_q = da_q['lon'].values
    sort_idx = np.argsort(levels)
    levels_Pa = levels[sort_idx] * 100.0
    # Column integrate entire timeseries
    q_data = da_q.values  # (time, level, lon)
    colq_ts = np.abs(np.trapz(q_data[:, sort_idx, :], x=levels_Pa, axis=1)) / 9.81
    ds_q.close()

    # Compute asymmetry for each event
    result = pd.DataFrame(index=range(n_ev),
                          columns=["asym_sst", "asym_lhf", "asym_precip", "asym_colq"])

    for i, (_, ev) in enumerate(events.iterrows()):
        ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
        si_mask = (time_step3 >= ts) & (time_step3 <= te)
        si_idx = np.where(si_mask)[0]
        if len(si_idx) == 0:
            continue

        asym_sst, asym_lhf, asym_tp, asym_q = [], [], [], []

        for si in si_idx:
            c, a = center_lon_all[si], amp_all[si]
            if not np.isfinite(c) or not np.isfinite(a) or a < AMP_THRESHOLD:
                continue
            t_val = time_step3[si]

            # Surface data
            for arr, t_idx, out_list in [
                (sst_arr, sst_t, asym_sst),
                (lhf_arr, sl_t, asym_lhf),
                (tp_arr, sl_t, asym_tp),
            ]:
                k = np.argmin(np.abs((t_idx - t_val).total_seconds()))
                if abs((t_idx[k] - t_val).total_seconds()) < 43200:
                    front = _get_segment_mean(arr[k], lons, c, *FRONT_RANGE)
                    rear = _get_segment_mean(arr[k], lons, c, *REAR_RANGE)
                    if np.isfinite(front) and np.isfinite(rear):
                        out_list.append(front - rear)

            # Column q
            k_q = np.argmin(np.abs((time_q - t_val).total_seconds()))
            if abs((time_q[k_q] - t_val).total_seconds()) < 43200:
                front_q = _get_segment_mean(colq_ts[k_q], lons_q, c, *FRONT_RANGE)
                rear_q = _get_segment_mean(colq_ts[k_q], lons_q, c, *REAR_RANGE)
                if np.isfinite(front_q) and np.isfinite(rear_q):
                    asym_q.append(front_q - rear_q)

        if asym_sst: result.loc[i, "asym_sst"] = np.mean(asym_sst)
        if asym_lhf: result.loc[i, "asym_lhf"] = np.mean(asym_lhf)
        if asym_tp: result.loc[i, "asym_precip"] = np.mean(asym_tp)
        if asym_q: result.loc[i, "asym_colq"] = np.mean(asym_q)

    return result.astype(float)


def plot_scatter(x, y, xlabel, ylabel, title, r, p, fast_mask, slow_mask, out_path):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    ok = np.isfinite(x) & np.isfinite(y)
    neutral = ~fast_mask & ~slow_mask & ok
    ax.scatter(x[neutral], y[neutral], c='gray', alpha=0.4, s=25)
    ax.scatter(x[fast_mask & ok], y[fast_mask & ok], c='tab:red', s=40,
               edgecolors='k', lw=0.5, label=f'Fast')
    ax.scatter(x[slow_mask & ok], y[slow_mask & ok], c='tab:blue', s=40,
               edgecolors='k', lw=0.5, label=f'Slow')
    # Regression line
    z = np.polyfit(x[ok], y[ok], 1)
    xx = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 50)
    ax.plot(xx, np.polyval(z, xx), 'k--', lw=1.5, alpha=0.6)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{title}\nr={r:.3f}, p={p:.4f} {sig}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_summary_bar(results, out_path):
    """Bar chart comparing r-values of all asymmetry indices."""
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    names = list(results.keys())
    r_vals = [results[k][0] for k in names]
    p_vals = [results[k][1] for k in names]
    colors = ['tab:red' if p < 0.05 else 'gray' for p in p_vals]
    bars = ax.bar(names, r_vals, color=colors, edgecolor='black', lw=0.8)
    for bar, r, p in zip(bars, r_vals, p_vals):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{r:.3f}{sig}", ha='center', fontsize=10)
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title("Front−Rear Asymmetry vs Phase Speed\n(+15°~+45° minus −45°~−15°)",
                 fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, ls='--', axis='y')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def main():
    print("=" * 70)
    print("06l: Front-Rear Asymmetry Index Analysis")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds3 = xr.open_dataset(STEP3_NC)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    amp_all = ds3["amp"].values.astype(float)
    time_step3 = pd.to_datetime(ds3["time"].values)
    ds3.close()

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)

    ps_valid = phase_speed[np.isfinite(phase_speed)]
    mu, sigma = np.mean(ps_valid), np.std(ps_valid)
    fast_mask = phase_speed > mu + GROUP_SIGMA * sigma
    slow_mask = phase_speed < mu - GROUP_SIGMA * sigma
    print(f"  Events: {len(merged)}, Fast={fast_mask.sum()}, Slow={slow_mask.sum()}")

    print("\n  Computing asymmetry indices ...")
    asym_df = compute_asymmetry_index(merged, center_lon_all, time_step3, amp_all)

    var_configs = {
        "SST": ("asym_sst", "ΔSST (Front−Rear, K)"),
        "LHF": ("asym_lhf", "ΔLHF (Front−Rear, W/m²)"),
        "Precip": ("asym_precip", "ΔPrecip (Front−Rear, mm/day)"),
        "Col q": ("asym_colq", "ΔCol q (Front−Rear, kg/m²)"),
    }

    results = {}
    for label, (col, xlabel) in var_configs.items():
        x = asym_df[col].values
        ok = np.isfinite(x) & np.isfinite(phase_speed)
        if ok.sum() < 10:
            print(f"  {label}: insufficient data ({ok.sum()})")
            continue
        r, p = stats.pearsonr(x[ok], phase_speed[ok])
        results[label] = (r, p)
        print(f"  {label}: r={r:.3f}, p={p:.4f}")
        plot_scatter(x, phase_speed, xlabel, "Phase Speed (m/s)",
                     f"Asymmetry: {label} vs Phase Speed",
                     r, p, fast_mask, slow_mask,
                     OUT_DIR / f"asymmetry_scatter_{col.replace('asym_', '')}.png")

    print("\n  --- Summary ---")
    plot_summary_bar(results, OUT_DIR / "asymmetry_summary.png")

    print(f"\nAll done! Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
