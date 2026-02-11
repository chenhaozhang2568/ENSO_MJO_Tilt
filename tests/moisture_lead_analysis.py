# -*- coding: utf-8 -*-
"""
moisture_lead_analysis.py — 边界层水汽领先性分析

功能：
    分析边界层水汽 q_925hPa 对 MJO 对流的领先作用，
    检验不同 ENSO 相位下水汽-对流相位关系的差异。
输入：
    era5_mjo_recon_q_norm_1979-2022.nc, mjo_mvEOF_step3_1979-2022.nc,
    mjo_events_step3_1979-2022.csv, tilt_event_stats_with_enso_1979-2022.csv
输出：
    figures/moisture_lead/moisture_lead_lag_correlation.png
用法：
    python tests/moisture_lead_analysis.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ========================
# PATHS
# ========================
Q_RECON_NC = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ENSO_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\moisture")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# SETTINGS
# ========================
MAX_LAG = 15  # days
ENSO_ORDER = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}


def load_data():
    """Load q_925 and OLR data"""
    print("Loading data...")
    
    # Load q at 925 hPa
    ds_q = xr.open_dataset(Q_RECON_NC)
    q_recon = ds_q['q_mjo_recon_norm']
    if "pressure_level" in q_recon.dims:
        q_recon = q_recon.rename({"pressure_level": "level"})
    q_925 = q_recon.sel(level=925)  # (time, lon)
    
    # Load OLR recon
    ds3 = xr.open_dataset(STEP3_NC)
    olr_recon = ds3['olr_recon']  # (time, lon)
    center_lon = ds3['center_lon_track'].values
    
    # Load events and ENSO
    events = pd.read_csv(EVENTS_CSV, parse_dates=['start_date', 'end_date'])
    enso_stats = pd.read_csv(ENSO_STATS_CSV)
    enso_map = dict(zip(enso_stats['event_id'], enso_stats['enso_phase']))
    
    # Build event days dataframe
    time = pd.to_datetime(q_925.time.values)
    df = pd.DataFrame({'time': time, 'center_lon': center_lon})
    df['event_id'] = np.nan
    df['enso_phase'] = ''
    
    for _, ev in events.iterrows():
        mask = (df['time'] >= ev['start_date']) & (df['time'] <= ev['end_date'])
        df.loc[mask, 'event_id'] = ev['event_id']
        df.loc[mask, 'enso_phase'] = enso_map.get(ev['event_id'], 'Unknown')
    
    print(f"  q_925 shape: {q_925.shape}")
    print(f"  OLR shape: {olr_recon.shape}")
    
    return q_925, olr_recon, df


def compute_lead_lag_correlation(q_925, olr, df, max_lag=15):
    """
    Compute lead-lag correlation between q_925 and OLR at MJO center longitude.
    Positive lag = q leads (moisture precedes convection)
    """
    print("\nComputing lead-lag correlations...")
    
    lon = q_925.lon.values
    lon_360 = np.mod(lon, 360)
    time = pd.to_datetime(q_925.time.values)
    
    lags = np.arange(-max_lag, max_lag + 1)
    
    results = {}
    
    for phase in ENSO_ORDER:
        phase_df = df[df['enso_phase'] == phase].copy()
        event_ids = phase_df['event_id'].dropna().unique()
        
        # Collect all event correlations
        event_corrs = []
        
        for eid in event_ids:
            ev_df = phase_df[phase_df['event_id'] == eid]
            if len(ev_df) < max_lag * 2:
                continue
            
            # Get indices for this event
            ev_times = ev_df['time'].values
            ev_indices = [np.where(time == t)[0][0] for t in ev_times if t in time.values]
            
            if len(ev_indices) < max_lag * 2:
                continue
            
            # Sample q and OLR at center longitude for each day
            q_series = []
            olr_series = []
            
            for idx in ev_indices:
                clon = df.iloc[idx]['center_lon']
                if not np.isfinite(clon):
                    q_series.append(np.nan)
                    olr_series.append(np.nan)
                    continue
                
                clon_360 = np.mod(clon, 360)
                lon_idx = np.argmin(np.abs(lon_360 - clon_360))
                
                q_series.append(float(q_925.isel(time=idx, lon=lon_idx).values))
                olr_series.append(float(olr.isel(time=idx, lon=lon_idx).values))
            
            q_series = np.array(q_series)
            olr_series = np.array(olr_series)
            
            # Compute lead-lag correlation for this event
            corrs = []
            for lag in lags:
                if lag >= 0:
                    q_lagged = q_series[:len(q_series)-lag] if lag > 0 else q_series
                    olr_target = olr_series[lag:]
                else:
                    q_lagged = q_series[-lag:]
                    olr_target = olr_series[:len(olr_series)+lag]
                
                valid = np.isfinite(q_lagged) & np.isfinite(olr_target)
                if valid.sum() < 5:
                    corrs.append(np.nan)
                else:
                    r, _ = stats.pearsonr(q_lagged[valid], olr_target[valid])
                    corrs.append(r)
            
            event_corrs.append(corrs)
        
        # Average across events
        if len(event_corrs) > 0:
            event_corrs = np.array(event_corrs)
            mean_corr = np.nanmean(event_corrs, axis=0)
            std_corr = np.nanstd(event_corrs, axis=0)
            n_events = (~np.isnan(event_corrs[:, max_lag])).sum()
        else:
            mean_corr = np.full(len(lags), np.nan)
            std_corr = np.full(len(lags), np.nan)
            n_events = 0
        
        results[phase] = {
            'lags': lags,
            'mean_corr': mean_corr,
            'std_corr': std_corr,
            'n_events': n_events
        }
        
        print(f"  {phase}: {n_events} events")
    
    return results


def find_peak_lag(lags, corr):
    """Find the lag with minimum correlation (q leads convection -> negative OLR)
    
    限制在 0-10 天范围搜索，避免找到边界值
    """
    # Since high q should precede low OLR (strong convection), we look for most negative correlation
    # 限制搜索范围在合理的领先天数 (0 到 10 天)
    search_mask = (lags >= 0) & (lags <= 10) & np.isfinite(corr)
    if not search_mask.any():
        return np.nan
    
    # 在限制范围内找最小值
    search_corr = np.where(search_mask, corr, np.inf)
    min_idx = np.argmin(search_corr)
    return lags[min_idx]


def analyze_moisture_lead(results):
    """Analyze and print results"""
    print("\n" + "="*70)
    print("Hypothesis 5: Boundary Layer Moisture Leading Analysis")
    print("="*70)
    
    print("\n[Lead-Lag Analysis] q_925 vs OLR at MJO center:")
    print("-"*50)
    print("  Positive lag = q leads (moisture precedes convection)")
    print("  More negative correlation = stronger q-OLR relationship")
    print()
    
    for phase in ENSO_ORDER:
        r = results[phase]
        peak_lag = find_peak_lag(r['lags'], r['mean_corr'])
        peak_corr = np.nanmin(r['mean_corr'])
        lag0_corr = r['mean_corr'][len(r['lags'])//2]  # lag=0
        
        print(f"  {phase:10s}: N_events={r['n_events']:3d}")
        print(f"              Peak correlation at lag={peak_lag:+.0f} days (r={peak_corr:.3f})")
        print(f"              Lag=0 correlation: r={lag0_corr:.3f}")
        print()


def plot_lead_lag_correlation(results):
    """Plot lead-lag correlation curves"""
    print("Plotting lead-lag correlation...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # =====================
    # Panel 1: Lead-lag correlation curves
    # =====================
    ax1 = axes[0]
    
    for phase in ENSO_ORDER:
        r = results[phase]
        lags = r['lags']
        mean_corr = r['mean_corr']
        std_corr = r['std_corr']
        sem = std_corr / np.sqrt(r['n_events'])
        
        ax1.plot(lags, mean_corr, color=ENSO_COLORS[phase], linewidth=2, 
                label=f"{phase} (N={r['n_events']})")
        ax1.fill_between(lags, mean_corr - sem, mean_corr + sem, 
                        color=ENSO_COLORS[phase], alpha=0.2)
    
    ax1.axvline(0, color='black', linestyle='--', linewidth=1.5)
    ax1.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_xlabel("Lag (days, positive = q leads)")
    ax1.set_ylabel("Correlation (q_925 vs OLR)")
    ax1.set_title("(a) Lead-Lag Correlation: q_925hPa vs OLR", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.set_xlim(-MAX_LAG, MAX_LAG)
    
    # Add annotation
    ax1.annotate("q leads\n(recharge)", xy=(8, ax1.get_ylim()[0] + 0.05), 
                fontsize=9, ha='center', style='italic')
    ax1.annotate("OLR leads\n(discharge)", xy=(-8, ax1.get_ylim()[0] + 0.05), 
                fontsize=9, ha='center', style='italic')
    
    # =====================
    # Panel 2: Peak lag comparison
    # =====================
    ax2 = axes[1]
    
    peak_lags = []
    peak_corrs = []
    for phase in ENSO_ORDER:
        r = results[phase]
        peak_lag = find_peak_lag(r['lags'], r['mean_corr'])
        peak_corr = np.nanmin(r['mean_corr'])
        peak_lags.append(peak_lag)
        peak_corrs.append(peak_corr)
    
    x = np.arange(len(ENSO_ORDER))
    bars = ax2.bar(x, peak_lags, 0.6, color=[ENSO_COLORS[p] for p in ENSO_ORDER], edgecolor='black')
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ENSO_ORDER)
    ax2.set_ylabel("Peak Correlation Lag (days)")
    ax2.set_title("(b) Lag at Peak q-OLR Correlation", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar, lag, corr in zip(bars, peak_lags, peak_corrs):
        if np.isfinite(lag):
            va = 'bottom' if lag >= 0 else 'top'
            offset = 0.3 if lag >= 0 else -0.3
            ax2.text(bar.get_x() + bar.get_width()/2, lag + offset, 
                    f'{lag:+.0f}d\n(r={corr:.2f})', ha='center', va=va, fontsize=10)
    
    plt.suptitle("Hypothesis 5: Boundary Layer Moisture Leading Analysis", 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = FIG_DIR / "moisture_lead_lag_correlation.png"
    plt.savefig(out_path, bbox_inches='tight')
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    q_925, olr, df = load_data()
    results = compute_lead_lag_correlation(q_925, olr, df, max_lag=MAX_LAG)
    analyze_moisture_lead(results)
    plot_lead_lag_correlation(results)
    
    print("\n" + "="*70)
    print("Moisture lead analysis completed!")
    print(f"Figures saved to: {FIG_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
