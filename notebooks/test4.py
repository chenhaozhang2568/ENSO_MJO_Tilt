# -*- coding: utf-8 -*-
"""
test4.py: Generate check plots for MJO Tilt analysis

Usage:
  cd notebooks
  python test4.py
"""

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths (using absolute paths based on existing config)
SRC_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\src")
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Input Files
TILT_NC = r"E:\Datas\ClimateIndex\processed\tilt_daily_step4_layermean_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\ClimateIndex\processed\mjo_events_step3_1979-2022.csv"

# Output Directory
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\checks")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Settings
sns.set_theme(style="whitegrid")

def load_data():
    print(f"Loading Tilt NC: {TILT_NC}")
    if not Path(TILT_NC).exists():
        raise FileNotFoundError(f"Missing {TILT_NC}")
    ds = xr.open_dataset(TILT_NC)
    
    print(f"Loading Events CSV: {EVENTS_CSV}")
    if not Path(EVENTS_CSV).exists():
        raise FileNotFoundError(f"Missing {EVENTS_CSV}")
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    
    return ds, events

def plot_event_trajectory(ds, events, n_examples=3):
    """
    Plot the trajectory (Long vs Time) for a few random events, colored by tilt or active status.
    """
    print("Generating Event Trajectories...")
    
    # Pick random events
    if len(events) == 0:
        print("No events to plot.")
        return
        
    sample_events = events.sample(min(n_examples, len(events)), random_state=42)
    
    for _, ev in sample_events.iterrows():
        eid = ev["event_id"]
        start = ev["start_date"]
        end = ev["end_date"]
        
        # Select data for this event
        sub = ds.sel(time=slice(start, end))
        if sub.sizes["time"] == 0:
            continue
            
        times = sub["time"].values
        # low_center_rel is relative to MJO center (which is 0). 
        # But we want the absolute longitude track? 
        # Wait, Step4 NC doesn't have absolute lon track, only relative geometries.
        # But Step3 NC has it. Since we only read Step4 NC here, we can plot the TILT evolution.
        
        # Let's plot Tilt Evolution instead + West/East boundaries relative to center
        fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
        
        t_nums = np.arange(len(times))
        
        # Plot Upper and Lower boxes (relative longitude)
        # Lower: low_west_rel to low_east_rel
        # Upper: up_west_rel to up_east_rel
        
        # To make it readable, we plot bars for each day
        width = 0.3
        
        # Lower Box (Blue)
        ax.bar(t_nums - width/2, 
               sub["low_east_rel"] - sub["low_west_rel"], 
               bottom=sub["low_west_rel"], 
               width=width, label="Lower Box (1000-700hPa)", color='blue', alpha=0.6)
               
        # Upper Box (Red)
        ax.bar(t_nums + width/2, 
               sub["up_east_rel"] - sub["up_west_rel"], 
               bottom=sub["up_west_rel"], 
               width=width, label="Upper Box (300-200hPa)", color='red', alpha=0.6)
        
        # Zero line (Convective Center)
        ax.axhline(0, color='black', linestyle='--', linewidth=1, label="Convective Center")
        
        # Tilt value
        ax2 = ax.twinx()
        ax2.plot(t_nums, sub["tilt"], color='green', marker='o', linewidth=2, label="Tilt Index")
        ax2.set_ylabel("Tilt Index (deg)")
        ax2.grid(False)
        
        ax.set_title(f"Event {eid}: {start.date()} to {end.date()} (Duration: {len(times)} days)")
        ax.set_ylabel("Relative Longitude (deg)")
        ax.set_xlabel("Day of Event")
        ax.set_xticks(t_nums)
        ax.set_xticklabels([str(d)[5:10] for d in pd.to_datetime(times)], rotation=45)
        
        # Legends - OUTSIDE the plot to avoid blocking data
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        
        plt.tight_layout()
        out_path = FIG_DIR / f"check_trajectory_event_{eid}.png"
        plt.savefig(out_path)
        print(f"  Saved {out_path}")
        plt.close()

def plot_tilt_geometry_check(ds):
    """
    Scatter plot of Lower West vs Upper West extent.
    Check if they are correlated or if one varies more than the other.
    """
    print("Generating Tilt Geometry Check...")
    
    # Filter only valid tilt days
    valid = ds.where(np.isfinite(ds["tilt"]), drop=True)
    
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    
    sns.scatterplot(x=valid["up_west_rel"], y=valid["low_west_rel"], 
                    alpha=0.3, s=10, ax=ax)
    
    # y=x line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label="1:1 Line")
    
    ax.set_aspect('equal')
    ax.set_xlabel("Upper West Extent (rel deg)")
    ax.set_ylabel("Lower West Extent (rel deg)")
    ax.set_title("Lower vs Upper West Boundaries")
    
    # Color regions
    # Tilt = Lower - Upper
    # If Lower > Upper (above diagonal) -> Positive Tilt (Westward with height)
    ax.fill_between(lims, lims, [lims[1], lims[1]], color='green', alpha=0.1, label="Positive Tilt (Westward)")
    ax.fill_between(lims, [lims[0], lims[0]], lims, color='red', alpha=0.1, label="Negative Tilt (Eastward)")
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "check_tilt_geometry_scatter.png")
    plt.close()

def plot_seasonal_distribution(ds):
    """
    Number of valid tilt days per month.
    """
    print("Generating Seasonal Distribution...")
    
    valid = ds["tilt"].dropna(dim="time")
    months = valid["time"].dt.month
    
    # Custom Order: 11, 12, 1, 2, 3, 4
    target_order = [11, 12, 1, 2, 3, 4]
    counts_series = months.to_series().value_counts()
    
    # Reindex to force specific order (fill missing with 0)
    counts = counts_series.reindex(target_order, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis", order=target_order)
    
    ax.set_xlabel("Month")
    ax.set_ylabel("Count of Valid Tilt Days")
    ax.set_title("Seasonal Distribution of Valid Tilt Samples (1979-2022)")
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "check_seasonal_counts.png")
    plt.close()

def plot_tilt_histogram_all(ds):
    """
    Histogram of ALL daily tilt values (not just event means).
    """
    print("Generating Daily Tilt Histogram...")
    
    tilt = ds["tilt"].values.flatten()
    tilt = tilt[np.isfinite(tilt)]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    sns.histplot(tilt, bins=30, kde=True, color="purple", ax=ax)
    
    ax.axvline(0, color='k', linestyle='--')
    ax.axvline(np.mean(tilt), color='r', linestyle='-', label=f"Mean: {np.mean(tilt):.2f}")
    
    ax.set_xlabel("Daily Tilt Index (deg)")
    ax.set_title(f"Distribution of Daily Tilt Values (N={len(tilt)})")
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "check_daily_tilt_hist.png")
    plt.close()

def main():
    ds, events = load_data()
    
    plot_event_trajectory(ds, events)
    plot_tilt_geometry_check(ds)
    plot_seasonal_distribution(ds)
    plot_tilt_histogram_all(ds)
    
    print("\nAll check plots generated in:", FIG_DIR)

if __name__ == "__main__":
    main()
