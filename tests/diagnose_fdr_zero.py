# -*- coding: utf-8 -*-
"""
Diagnose why bg_q, mjo_q, mjo_w show 0.0% FDR significance.
Check: raw p-value distribution, sample sizes, and FDR threshold.
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.interpolate import interp1d
from pathlib import Path

DERIVED_DIR = Path(r"E:\Datas\Derived")
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = DERIVED_DIR / "mjo_events_step3_1979-2022.csv"
PHASE_SPEED_CSV = DERIVED_DIR / "phase_speed_q_events.csv"

AMP_THRESHOLD = 0.5
GROUP_THRESHOLD_STD = 0.7
FDR_ALPHA = 0.05
REL_LON_WEST, REL_LON_EAST = -180.0, 180.0

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}
TARGET_HEIGHTS = np.linspace(0.5, 12, 24)


def _rename_level(da):
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    return da


def fdr_correction(p_values, alpha=0.05):
    shape = p_values.shape
    pf = p_values.flatten()
    valid = np.isfinite(pf)
    sig = np.zeros(pf.size, dtype=bool)
    if valid.sum() == 0:
        return sig.reshape(shape)
    pv = pf[valid]
    si = np.argsort(pv)
    ranks = np.empty_like(si)
    ranks[si] = np.arange(1, len(pv) + 1)
    m = len(pv)
    sv = pv <= (ranks / m * alpha)
    if sv.any():
        sv = ranks <= np.max(ranks[sv])
    sig[valid] = sv
    return sig.reshape(shape)


def _align_day(data_day, lons, center_lon, rel_lons, dlon):
    lon_360 = np.mod(lons, 360)
    c360 = np.mod(center_lon, 360)
    nL = data_day.shape[0]
    sample = np.full((nL, len(rel_lons)), np.nan, dtype=np.float32)
    for j, rl in enumerate(rel_lons):
        tlon = np.mod(c360 + rl, 360)
        k = np.argmin(np.abs(lon_360 - tlon))
        if np.abs(lon_360[k] - tlon) < dlon:
            sample[:, j] = data_day[:, k]
    return sample


def diagnose_var(var, events, hi_mask, lo_mask, time_step3, center_lon_all, amp_all, mode="bg"):
    print(f"\n{'='*60}")
    print(f"Variable: {var}, Mode: {mode}")
    print(f"{'='*60}")

    ds = xr.open_dataset(DERIVED_DIR / f"era5_mjo_recon_{var}_norm_1979-2022.nc")
    vn = [k for k in ds.data_vars if var in k.lower()][0]
    da = _rename_level(ds[vn])
    time_field = pd.to_datetime(da["time"].values)
    levels = da["level"].values
    lons = da["lon"].values
    data = da.values
    dlon = np.abs(lons[1] - lons[0])

    if mode == "mjo":
        n_x = int((REL_LON_EAST - REL_LON_WEST) / dlon) + 1
        x_axis = np.linspace(REL_LON_WEST, REL_LON_EAST, n_x)
    else:
        n_x = len(lons)
        x_axis = lons

    nL = len(levels)

    for grp_name, grp_mask in [("hi", hi_mask), ("lo", lo_mask)]:
        group_events = events[grp_mask]
        n_ev = len(group_events)
        em = np.full((n_ev, nL, n_x), np.nan, dtype=np.float32)

        for i, (_, ev) in enumerate(group_events.iterrows()):
            ts, te = pd.Timestamp(ev["start_date"]), pd.Timestamp(ev["end_date"])
            fi_mask_t = (time_field >= ts) & (time_field <= te)
            fi_idx = np.where(fi_mask_t)[0]
            if len(fi_idx) == 0:
                continue

            if mode == "bg":
                block = data[fi_idx, :, :]
                em[i] = np.nanmean(block, axis=0)
            else:
                si_mask_t = (time_step3 >= ts) & (time_step3 <= te)
                si_idx = np.where(si_mask_t)[0]
                samples = []
                for fi in fi_idx:
                    t_val = time_field[fi]
                    si_match = [si for si in si_idx
                                if abs((time_step3[si] - t_val).total_seconds()) < 43200]
                    if not si_match:
                        continue
                    si = si_match[0]
                    c, a = center_lon_all[si], amp_all[si]
                    if not np.isfinite(c) or not np.isfinite(a) or a < AMP_THRESHOLD:
                        continue
                    s = _align_day(data[fi], lons, c, x_axis, dlon)
                    samples.append(s)
                if samples:
                    em[i] = np.nanmean(np.array(samples), axis=0)

        if grp_name == "hi":
            em_hi = em
        else:
            em_lo = em

    # Check finite counts in event_means
    hi_finite_per_event = np.array([np.isfinite(em_hi[i]).sum() for i in range(em_hi.shape[0])])
    lo_finite_per_event = np.array([np.isfinite(em_lo[i]).sum() for i in range(em_lo.shape[0])])
    print(f"\n  hi events with data: {(hi_finite_per_event > 0).sum()} / {em_hi.shape[0]}")
    print(f"  lo events with data: {(lo_finite_per_event > 0).sum()} / {em_lo.shape[0]}")

    # --- Compute raw t-test p-values ---
    p_map = np.full((nL, n_x), np.nan, dtype=np.float32)
    sample_sizes_hi = np.full((nL, n_x), 0, dtype=int)
    sample_sizes_lo = np.full((nL, n_x), 0, dtype=int)
    for k in range(nL):
        for j in range(n_x):
            h, l = em_hi[:, k, j], em_lo[:, k, j]
            vh, vl = np.isfinite(h), np.isfinite(l)
            sample_sizes_hi[k, j] = vh.sum()
            sample_sizes_lo[k, j] = vl.sum()
            if vh.sum() >= 3 and vl.sum() >= 3:
                _, p_map[k, j] = stats.ttest_ind(h[vh], l[vl], equal_var=False)

    valid_p = p_map[np.isfinite(p_map)]
    print(f"\n  Total grid points: {nL * n_x}")
    print(f"  Valid p-values: {len(valid_p)}")
    print(f"  Avg hi sample size: {sample_sizes_hi[sample_sizes_hi > 0].mean():.1f}")
    print(f"  Avg lo sample size: {sample_sizes_lo[sample_sizes_lo > 0].mean():.1f}")
    print(f"\n  p < 0.05 (uncorrected): {(valid_p < 0.05).sum()} ({(valid_p < 0.05).sum()/len(valid_p)*100:.1f}%)")
    print(f"  p < 0.10 (uncorrected): {(valid_p < 0.10).sum()} ({(valid_p < 0.10).sum()/len(valid_p)*100:.1f}%)")
    print(f"  Min p-value: {np.min(valid_p):.8f}")
    print(f"  Median p-value: {np.median(valid_p):.4f}")
    print(f"  1st percentile: {np.percentile(valid_p, 1):.6f}")
    print(f"  5th percentile: {np.percentile(valid_p, 5):.6f}")

    # FDR threshold
    m = len(valid_p)
    sorted_p = np.sort(valid_p)
    fdr_thresholds = np.arange(1, m+1) / m * FDR_ALPHA
    print(f"\n  FDR threshold at rank 1: {fdr_thresholds[0]:.10f}")
    print(f"  Smallest p-value:       {sorted_p[0]:.10f}")
    if sorted_p[0] <= fdr_thresholds[0]:
        print(f"  => PASSES FDR at rank 1")
    else:
        print(f"  => FAILS FDR: smallest p ({sorted_p[0]:.8f}) > threshold ({fdr_thresholds[0]:.8f})")
        # Find smallest k where p(k) <= k/m * alpha
        passes = sorted_p <= fdr_thresholds
        if passes.any():
            max_k = np.max(np.where(passes)[0]) + 1
            print(f"  But passes at higher rank? max_k = {max_k}")
        else:
            print(f"  => NO rank passes FDR. This is genuinely 0.0% FDR.")

    sig_fdr = fdr_correction(p_map, FDR_ALPHA)
    n_sig = sig_fdr.sum()
    print(f"\n  FDR significant: {n_sig} / {len(valid_p)} ({n_sig/len(valid_p)*100:.1f}%)")

    ds.close()


# --- MAIN ---
print("Loading metadata...")
ds3 = xr.open_dataset(STEP3_NC)
center_lon_all = ds3["center_lon_track"].values.astype(float)
amp_all = ds3["amp"].values.astype(float)
time_step3 = pd.to_datetime(ds3["time"].values)
ds3.close()

events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
ps_df = pd.read_csv(PHASE_SPEED_CSV)
merged = events.merge(ps_df[["event_id", "phase_speed_m_s"]], on="event_id")
speed = merged["phase_speed_m_s"].values.astype(float)
sp_mean, sp_std = np.nanmean(speed), np.nanstd(speed)
hi_thr = sp_mean + GROUP_THRESHOLD_STD * sp_std
lo_thr = sp_mean - GROUP_THRESHOLD_STD * sp_std
hi_mask = speed > hi_thr
lo_mask = speed < lo_thr
print(f"Fast (>{hi_thr:.2f}): N={hi_mask.sum()}")
print(f"Slow (<{lo_thr:.2f}): N={lo_mask.sum()}")

# Diagnose the 3 problematic cases
diagnose_var("q", merged, hi_mask, lo_mask, time_step3, center_lon_all, amp_all, mode="bg")
diagnose_var("q", merged, hi_mask, lo_mask, time_step3, center_lon_all, amp_all, mode="mjo")
diagnose_var("w", merged, hi_mask, lo_mask, time_step3, center_lon_all, amp_all, mode="mjo")

# Also diagnose one "working" case for comparison (bg_t has 27%)
diagnose_var("t", merged, hi_mask, lo_mask, time_step3, center_lon_all, amp_all, mode="bg")

print("\n\nDone!")
