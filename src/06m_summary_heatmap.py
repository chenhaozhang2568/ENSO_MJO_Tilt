# -*- coding: utf-8 -*-
"""
06m_summary_heatmap.py
全变量汇总：热力图 + 重要性排名柱状图
输出到 field_phase_speed_correlation/ 根目录
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

AMP_THRESHOLD = 0.5
GROUP_SIGMA = 0.7
R_EARTH = 6.371e6

# ====== All variables and their data sources ======
# Format: (label, data_type, data_key)
# data_type: "recon_2d", "recon_1d_col", "sl_1d", "sst_1d", "advection_2d", "olr_1d"
VARIABLE_REGISTRY = [
    # 2D pressure-level (take max |r| across all grid points)
    ("u", "recon_2d", "u"),
    ("v", "recon_2d", "v"),
    ("w (omega)", "recon_2d", "w"),
    ("q (humidity)", "recon_2d", "q"),
    ("T", "recon_2d", "t"),
    # 1D column-integrated
    ("Column q", "recon_1d_col", "q"),
    ("Column MSE", "recon_1d_col", "mse"),
    # 1D surface
    ("OLR", "olr_1d", "olr"),
    ("LHF", "sl_1d", "slhf"),
    ("SST", "sst_1d", "sst"),
    ("SHF", "sl_1d", "sshf"),
    ("Q_rad (net)", "sl_1d", "qrad"),
    ("Precip", "sl_1d", "tp"),
    # 2D derived
    ("-u·∂q/∂x", "advection_2d", "full"),
    ("TermA (-ū·∂q'/∂x)", "advection_2d", "termA"),
    ("TermB (-u'·∂q̄/∂x)", "advection_2d", "termB"),
]


def _rename_level(da):
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    return da


def compute_max_r_recon_2d(events, phase_speed, var_name, coord="bg"):
    """Compute max |r| for a 2D recon field in bg or mjo coordinate."""
    fname = f"era5_mjo_recon_{var_name}_norm_1979-2022.nc"
    ds = xr.open_dataset(DERIVED_DIR / fname)
    vn = [k for k in ds.data_vars if var_name in k.lower()][0]
    da = _rename_level(ds[vn])
    time_all = pd.to_datetime(da["time"].values)
    levels = da["level"].values
    lons = da["lon"].values
    data = da.values
    ds.close()

    n_ev = len(events)
    nL, nX = len(levels), len(lons)

    if coord == "bg":
        ev_means = np.full((n_ev, nL, nX), np.nan, dtype=np.float32)
        for i, (_, ev) in enumerate(events.iterrows()):
            mask = (time_all >= pd.Timestamp(ev["start_date"])) & \
                   (time_all <= pd.Timestamp(ev["end_date"]))
            if mask.sum() > 0:
                ev_means[i] = np.nanmean(data[mask], axis=0)
    else:
        # MJO aligned
        ds3 = xr.open_dataset(STEP3_NC)
        center_lon_all = ds3["center_lon_track"].values.astype(float)
        amp_all = ds3["amp"].values.astype(float)
        time_step3 = pd.to_datetime(ds3["time"].values)
        ds3.close()
        dlon = np.abs(lons[1] - lons[0])
        n_rel = int(360 / dlon) + 1
        rel_lons = np.linspace(-180, 180, n_rel)
        ev_means = np.full((n_ev, nL, n_rel), np.nan, dtype=np.float32)
        for i, (_, ev) in enumerate(events.iterrows()):
            ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
            fi_mask = (time_all >= ts) & (time_all <= te)
            si_mask = (time_step3 >= ts) & (time_step3 <= te)
            fi_idx = np.where(fi_mask)[0]
            si_idx = np.where(si_mask)[0]
            if len(fi_idx) == 0: continue
            samples = []
            for fi in fi_idx:
                t_val = time_all[fi]
                si_match = [si for si in si_idx
                            if abs((time_step3[si] - t_val).total_seconds()) < 43200]
                if not si_match: continue
                si = si_match[0]
                c, a = center_lon_all[si], amp_all[si]
                if not np.isfinite(c) or not np.isfinite(a) or a < AMP_THRESHOLD: continue
                lon_360 = np.mod(lons, 360)
                c360 = np.mod(c, 360)
                sample = np.full((nL, n_rel), np.nan, dtype=np.float32)
                for j, rl in enumerate(rel_lons):
                    tlon = np.mod(c360 + rl, 360)
                    k = np.argmin(np.abs(lon_360 - tlon))
                    if np.abs(lon_360[k] - tlon) < dlon:
                        sample[:, j] = data[fi, :, k]
                samples.append(sample)
            if samples:
                ev_means[i] = np.nanmean(np.array(samples), axis=0)

    # Compute r at each grid point, return max |r|
    max_r, best_p = 0.0, 1.0
    n_ev_dim, nL_dim, nX_dim = ev_means.shape
    for k in range(nL_dim):
        for j in range(nX_dim):
            v = ev_means[:, k, j]
            ok = np.isfinite(v) & np.isfinite(phase_speed)
            if ok.sum() < 10: continue
            r, p = stats.pearsonr(v[ok], phase_speed[ok])
            if abs(r) > abs(max_r):
                max_r, best_p = r, p
    return max_r, best_p


def compute_max_r_1d(events, phase_speed, data_arr, time_arr, lons, coord="bg",
                     center_lon_all=None, time_step3=None, amp_all=None):
    """Compute max |r| for a 1D field."""
    n_ev = len(events)

    if coord == "bg":
        nX = len(lons)
        ev_means = np.full((n_ev, nX), np.nan, dtype=np.float32)
        for i, (_, ev) in enumerate(events.iterrows()):
            mask = (time_arr >= pd.Timestamp(ev["start_date"])) & \
                   (time_arr <= pd.Timestamp(ev["end_date"]))
            if mask.sum() > 0:
                ev_means[i] = np.nanmean(data_arr[mask], axis=0)
    else:
        dlon = np.abs(lons[1] - lons[0])
        n_rel = int(360 / dlon) + 1
        rel_lons = np.linspace(-180, 180, n_rel)
        ev_means = np.full((n_ev, n_rel), np.nan, dtype=np.float32)
        for i, (_, ev) in enumerate(events.iterrows()):
            ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
            fi_mask = (time_arr >= ts) & (time_arr <= te)
            si_mask = (time_step3 >= ts) & (time_step3 <= te)
            fi_idx = np.where(fi_mask)[0]
            si_idx = np.where(si_mask)[0]
            if len(fi_idx) == 0: continue
            samples = []
            for fi in fi_idx:
                t_val = time_arr[fi]
                si_match = [si for si in si_idx
                            if abs((time_step3[si] - t_val).total_seconds()) < 43200]
                if not si_match: continue
                si = si_match[0]
                c, a = center_lon_all[si], amp_all[si]
                if not np.isfinite(c) or not np.isfinite(a) or a < AMP_THRESHOLD: continue
                lon_360 = np.mod(lons, 360)
                c360 = np.mod(c, 360)
                s = np.full(n_rel, np.nan, dtype=np.float32)
                for j, rl in enumerate(rel_lons):
                    tlon = np.mod(c360 + rl, 360)
                    k = np.argmin(np.abs(lon_360 - tlon))
                    if np.abs(lon_360[k] - tlon) < dlon:
                        s[j] = data_arr[fi, k]
                samples.append(s)
            if samples:
                ev_means[i] = np.nanmean(np.array(samples), axis=0)

    max_r, best_p = 0.0, 1.0
    nX_dim = ev_means.shape[1]
    for j in range(nX_dim):
        v = ev_means[:, j]
        ok = np.isfinite(v) & np.isfinite(phase_speed)
        if ok.sum() < 10: continue
        r, p = stats.pearsonr(v[ok], phase_speed[ok])
        if abs(r) > abs(max_r):
            max_r, best_p = r, p
    return max_r, best_p


def main():
    print("=" * 70)
    print("06m: Summary Heatmap & Variable Importance Ranking")
    print("=" * 70)

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ps_df = pd.read_csv(PHASE_SPEED_CSV)
    merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
    phase_speed = merged["phase_speed_m_s"].values.astype(float)
    print(f"  Events: {len(merged)}")

    # Load step3
    ds3 = xr.open_dataset(STEP3_NC)
    center_lon_all = ds3["center_lon_track"].values.astype(float)
    amp_all = ds3["amp"].values.astype(float)
    time_step3 = pd.to_datetime(ds3["time"].values)
    ds3.close()

    # Load surface data once
    sl_files = sorted(SL_DIR.glob("era5_sl_dailymean_*.nc"))
    all_sl_data = {}
    all_sl_t = []
    lons_sl = None
    for f in sl_files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        t = pd.to_datetime(ds[tdim].values)
        for vn in ['slhf', 'sshf', 'tp']:
            vals = np.nanmean(ds[vn].values, axis=1)
            if vn in ['slhf', 'sshf']:
                vals = -vals  # flip sign
            all_sl_data.setdefault(vn, []).append(vals)
        # Net radiation
        qrad = np.nanmean(ds['ssr'].values + ds['str'].values, axis=1)
        all_sl_data.setdefault('qrad', []).append(qrad)
        all_sl_t.append(t)
        if lons_sl is None:
            lons_sl = ds['longitude'].values
        ds.close()

    sl_arrays = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in all_sl_data.items()}
    sl_t = pd.DatetimeIndex(np.concatenate(all_sl_t))

    # SST
    sst_files = sorted(SST_DIR.glob("era5_sst_dailymean_*.nc"))
    all_sst, all_sst_t = [], []
    for f in sst_files:
        ds = xr.open_dataset(f)
        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        all_sst.append(np.nanmean(ds['sst'].values, axis=1))
        all_sst_t.append(pd.to_datetime(ds[tdim].values))
        ds.close()
    sst_arr = np.concatenate(all_sst, axis=0).astype(np.float32)
    sst_t = pd.DatetimeIndex(np.concatenate(all_sst_t))

    # OLR
    ds3 = xr.open_dataset(STEP3_NC)
    olr = ds3['olr_recon'].values.astype(np.float64)
    amp = ds3['amp'].values.astype(np.float64)
    lons_olr = ds3['lon'].values
    time_olr = pd.to_datetime(ds3['time'].values)
    ds3.close()
    amp[amp < AMP_THRESHOLD] = np.nan
    olr_norm = (olr / amp[:, None]).astype(np.float32)

    # ====== Compute max |r| for all variables, bg and mjo ======
    results = {}  # {label: {"bg_r": r, "bg_p": p, "mjo_r": r, "mjo_p": p}}

    var_list = [
        ("u", "recon_2d", "u"),
        ("v", "recon_2d", "v"),
        ("ω (omega)", "recon_2d", "w"),
        ("q (humidity)", "recon_2d", "q"),
        ("T", "recon_2d", "t"),
    ]

    for label, dtype, key in var_list:
        print(f"  {label} (bg)...", end="")
        r_bg, p_bg = compute_max_r_recon_2d(merged, phase_speed, key, "bg")
        print(f" r={r_bg:.3f}", end="")
        print(f"  (mjo)...", end="")
        r_mjo, p_mjo = compute_max_r_recon_2d(merged, phase_speed, key, "mjo")
        print(f" r={r_mjo:.3f}")
        results[label] = {"bg_r": r_bg, "bg_p": p_bg, "mjo_r": r_mjo, "mjo_p": p_mjo}

    # 1D surface variables
    sl_vars = [
        ("LHF", sl_arrays['slhf'], sl_t, lons_sl),
        ("SHF", sl_arrays['sshf'], sl_t, lons_sl),
        ("Q_rad", sl_arrays['qrad'], sl_t, lons_sl),
        ("Precip", sl_arrays['tp'], sl_t, lons_sl),
        ("SST", sst_arr, sst_t, lons_sl),
    ]
    for label, arr, t_arr, lons in sl_vars:
        print(f"  {label} (bg)...", end="")
        r_bg, p_bg = compute_max_r_1d(merged, phase_speed, arr, t_arr, lons, "bg")
        print(f" r={r_bg:.3f}", end="")
        print(f"  (mjo)...", end="")
        r_mjo, p_mjo = compute_max_r_1d(merged, phase_speed, arr, t_arr, lons, "mjo",
                                         center_lon_all, time_step3, amp_all)
        print(f" r={r_mjo:.3f}")
        results[label] = {"bg_r": r_bg, "bg_p": p_bg, "mjo_r": r_mjo, "mjo_p": p_mjo}

    # OLR
    print(f"  OLR (bg)...", end="")
    r_bg, p_bg = compute_max_r_1d(merged, phase_speed, olr_norm, time_olr, lons_olr, "bg")
    print(f" r={r_bg:.3f}", end="")
    print(f"  (mjo)...", end="")
    r_mjo, p_mjo = compute_max_r_1d(merged, phase_speed, olr_norm, time_olr, lons_olr, "mjo",
                                     center_lon_all, time_step3, amp_all)
    print(f" r={r_mjo:.3f}")
    results["OLR"] = {"bg_r": r_bg, "bg_p": p_bg, "mjo_r": r_mjo, "mjo_p": p_mjo}

    # ====== PLOT 1: Heatmap ======
    print("\n  --- Heatmap ---")
    labels = list(results.keys())
    n = len(labels)
    heatmap = np.zeros((n, 2))
    for i, lab in enumerate(labels):
        heatmap[i, 0] = results[lab]["bg_r"]
        heatmap[i, 1] = results[lab]["mjo_r"]

    fig, ax = plt.subplots(figsize=(6, max(8, n * 0.55)), dpi=150)
    vmax = np.nanmax(np.abs(heatmap))
    im = ax.imshow(heatmap, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    for i in range(n):
        for j in range(2):
            r = heatmap[i, j]
            p = results[labels[i]]["bg_p" if j == 0 else "mjo_p"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            color = 'white' if abs(r) > vmax * 0.6 else 'black'
            ax.text(j, i, f"{r:.3f}{sig}", ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Background", "Perturbation\n(MJO-aligned)"], fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title("Max |r| with Phase Speed: All Variables",
                 fontsize=13, fontweight='bold', pad=15)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "summary_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: summary_heatmap.png")

    # ====== PLOT 2: Importance Ranking ======
    print("\n  --- Importance Ranking ---")
    # Take max(|bg_r|, |mjo_r|) for each variable
    best = []
    for lab in labels:
        bg_r = results[lab]["bg_r"]
        mjo_r = results[lab]["mjo_r"]
        if abs(bg_r) >= abs(mjo_r):
            best.append((lab, bg_r, results[lab]["bg_p"], "BG"))
        else:
            best.append((lab, mjo_r, results[lab]["mjo_p"], "MJO"))

    # Sort by |r|
    best.sort(key=lambda x: abs(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.5)), dpi=150)
    y_pos = range(len(best))
    r_vals = [b[1] for b in best]
    names = [f"{b[0]} ({b[3]})" for b in best]
    colors = ['tab:red' if b[2] < 0.05 else 'lightgray' for b in best]

    bars = ax.barh(y_pos, r_vals, color=colors, edgecolor='black', lw=0.6, height=0.7)
    for i, (name, r, p, src) in enumerate(best):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if r >= 0:
            ax.text(r + 0.01, i, f"r={r:.3f}{sig}", va='center', ha='left', fontsize=9)
        else:
            # Place label inside the bar to avoid overlapping with y-axis labels
            ax.text(r + 0.01, i, f"r={r:.3f}{sig}", va='center', ha='left', fontsize=9,
                    color='white', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color='gray', lw=0.8)
    ax.set_xlabel("Max Pearson r with Phase Speed", fontsize=12)
    ax.set_title("Variable Importance Ranking (Best r across BG/MJO coords)",
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.subplots_adjust(left=0.22)
    plt.savefig(FIG_DIR / "summary_ranking.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: summary_ranking.png")

    print(f"\nAll done! Output: {FIG_DIR}")


if __name__ == "__main__":
    main()
