# -*- coding: utf-8 -*-
"""
Calculate tilt statistics separated by ENSO phase (El Nino, La Nina, Neutral).

Inputs:
- Tilt Event Stats: E:\\Datas\\ClimateIndex\\processed\\tilt_event_stats_1979-2022.csv
  (Output from 04_tilt_statistics.py)
- ONI Index: E:\\Datas\\ClimateIndex\\raw\\oni\\oni.ascii.txt

Outputs:
- ENSO classified stats CSV: E:\\Datas\\ClimateIndex\\processed\\tilt_event_stats_with_enso.csv
- Boxplot Figure: E:\\Projects\\ENSO_MJO_Tilt\\outputs\\figures\\tilt_boxplot_by_enso.png
- Distribution Figure: E:\\Projects\\ENSO_MJO_Tilt\\outputs\\figures\\tilt_distribution_by_enso.png
- Combined Stats Figure: E:\\Projects\\ENSO_MJO_Tilt\\outputs\\figures\\tilt_combined_stats_by_enso.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# ======================
# PATHS
# ======================
EVENT_STATS_CSV = r"E:\Datas\ClimateIndex\processed\tilt_event_stats_1979-2022.csv"
ONI_TXT = r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt"

OUT_CSV = r"E:\Datas\ClimateIndex\processed\tilt_event_stats_with_enso.csv"
FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG_BOX = FIG_DIR / "tilt_boxplot_by_enso.png"
OUT_FIG_DIST = FIG_DIR / "tilt_distribution_by_enso.png"
OUT_FIG_COMBINED = FIG_DIR / "tilt_combined_stats_by_enso.png"

# ======================
# SETTINGS
# ======================
# NOAA Operational Definitions
ONI_ELNINO_THRESH = 0.5
ONI_LANINA_THRESH = -0.5

# ENSO phase order and colors
ENSO_ORDER = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {"El Nino": "#E74C3C", "Neutral": "#95A5A6", "La Nina": "#3498DB"}

def parse_oni(path):
    """
    Parse NOAA ONI ascii file.
    Format: SEAS YR TOTAL ANOM
    We only need YR, SEAS(to month), ANOM
    """
    # Mapping seasonal strings to approximate center month
    seas_map = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
    }
    
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        seas_str = parts[0]
        yr = int(parts[1])
        anom = float(parts[3])
        
        # Construct a datetime (using center month)
        mon = seas_map.get(seas_str, None)
        if mon is None:
            continue
            
        # For DJF, the year in the file is usually the Jan/Feb year? 
        # Checking file content: line 2 is DJF 1950. Usually DJF 1950 in NOAA file means Dec 1949 - Feb 1950 centered on Jan 1950.
        # So YR is the year of the central month.
        dt = pd.Timestamp(year=yr, month=mon, day=1)
        data.append({"time": dt, "oni": anom})
        
    df = pd.DataFrame(data)
    df = df.set_index("time").sort_index()
    return df

def classify_event(start_date, end_date, oni_df):
    """
    Classify an event based on the average ONI during its duration.
    """
    # Get ONI values within the event range (rounded to nearest month)
    # Since events are short (<1-2 months), usually 1 or 2 monthly values cover it.
    
    # Resample event range to monthly to match ONI
    # Simple approach: get all ONI records that fall within [start_month, end_month]
    s = pd.Timestamp(start_date).replace(day=1)
    e = pd.Timestamp(end_date).replace(day=1) + pd.offsets.MonthEnd(0)
    
    # Slice ONI
    # Using slice on DatetimeIndex
    # Need to handle exact indexing
    # Let's find index where time >= s and time <= e
    mask = (oni_df.index >= s) & (oni_df.index <= e)
    sub = oni_df.loc[mask]
    
    if sub.empty:
        # Fallback: find nearest single month
        idx = oni_df.index.get_indexer([pd.Timestamp(start_date)], method="nearest")
        if idx[0] == -1:
            return np.nan, "Unknown"
        val = oni_df.iloc[idx[0]]["oni"]
    else:
        val = sub["oni"].mean()
        
    # Classify
    if val >= ONI_ELNINO_THRESH:
        cat = "El Nino"
    elif val <= ONI_LANINA_THRESH:
        cat = "La Nina"
    else:
        cat = "Neutral"
        
    return val, cat

def main():
    # 1. Load Inputs
    print(f"Loading ONI: {ONI_TXT}")
    if not Path(ONI_TXT).exists():
        raise FileNotFoundError(f"Missing ONI file: {ONI_TXT}")
    oni_df = parse_oni(ONI_TXT)
    print(f"  ONI range: {oni_df.index[0].date()} to {oni_df.index[-1].date()}")
    
    print(f"Loading event stats: {EVENT_STATS_CSV}")
    if not Path(EVENT_STATS_CSV).exists():
        # Fallback: if not generated yet, remind user
         raise FileNotFoundError(f"Missing event stats CSV: {EVENT_STATS_CSV}. Run 05_tilt_statistics.py first.")
    
    events = pd.read_csv(EVENT_STATS_CSV, parse_dates=["start_date", "end_date"])
    
    # Filter only valid events (with valid mean_tilt)
    events = events.dropna(subset=["mean_tilt"])
    print(f"  Valid events: {len(events)}")
    
    # 2. Classify Events
    oni_vals = []
    enso_cats = []
    
    for _, row in events.iterrows():
        val, cat = classify_event(row["start_date"], row["end_date"], oni_df)
        oni_vals.append(val)
        enso_cats.append(cat)
        
    events["oni_avg"] = oni_vals
    events["enso_phase"] = enso_cats
    
    # 3. Save Classified Data
    events.to_csv(OUT_CSV, index=False)
    print(f"Saved classified events to: {OUT_CSV}")
    
    # 4. Statistics by Group
    groups = events.groupby("enso_phase")["mean_tilt"]
    print("\n" + "="*40)
    print("TILT STATISTICS BY ENSO PHASE")
    print("="*40)
    print(groups.describe())
    
    # Extract arrays for testing
    el_nino = events[events["enso_phase"] == "El Nino"]["mean_tilt"]
    la_nina = events[events["enso_phase"] == "La Nina"]["mean_tilt"]
    neutral = events[events["enso_phase"] == "Neutral"]["mean_tilt"]
    
    print("\n--- Significance Tests (T-test) ---")
    
    if len(el_nino) > 1 and len(la_nina) > 1:
        t, p = stats.ttest_ind(el_nino, la_nina, equal_var=False)
        print(f"El Nino vs La Nina: t={t:.3f}, p={p:.4f} {'*' if p<0.05 else ''}")
    else:
        print("El Nino vs La Nina: Not enough samples")
        
    if len(el_nino) > 1 and len(neutral) > 1:
        t, p = stats.ttest_ind(el_nino, neutral, equal_var=False)
        print(f"El Nino vs Neutral: t={t:.3f}, p={p:.4f} {'*' if p<0.05 else ''}")
        
    if len(la_nina) > 1 and len(neutral) > 1:
        t, p = stats.ttest_ind(la_nina, neutral, equal_var=False)
        print(f"La Nina vs Neutral: t={t:.3f}, p={p:.4f} {'*' if p<0.05 else ''}")

    # 5. Plotting
    # =====================
    # Figure 1: Boxplot
    # =====================
    plt.figure(figsize=(8, 6), dpi=150)
    
    # Order: El Nino, Neutral, La Nina
    order = ["El Nino", "Neutral", "La Nina"]
    palette = {"El Nino": "#FF6B6B", "Neutral": "#E0E0E0", "La Nina": "#4ECDC4"}
    
    # Use seaborn for nice boxplots
    # boxplot with swarmplot overlay to see points
    sns.boxplot(x="enso_phase", y="mean_tilt", data=events, order=order,
                hue="enso_phase", palette=palette, width=0.5, showfliers=False, legend=False)
                
    sns.swarmplot(x="enso_phase", y="mean_tilt", data=events, order=order,
                  color=".2", alpha=0.7, warn_thresh=0)
    
    plt.title("MJO Vertical Tilt Distribution by ENSO Phase")
    plt.ylabel("Mean Tilt (deg)")
    plt.xlabel("ENSO Phase")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add counts to labels
    counts = events["enso_phase"].value_counts()
    new_labels = [f"{lbl}\n(N={counts.get(lbl, 0)})" for lbl in order]
    plt.xticks(range(3), new_labels)

    plt.tight_layout()
    plt.savefig(OUT_FIG_BOX)
    print(f"\nSaved boxplot to: {OUT_FIG_BOX}")
    plt.close()

    # =====================
    # Figure 2: Distribution (KDE + Histogram)
    # =====================
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    for phase in order:
        subset = events[events["enso_phase"] == phase]["mean_tilt"]
        if len(subset) > 2:
            sns.kdeplot(subset, label=f"{phase} (N={len(subset)})", 
                       color=palette[phase], linewidth=2, ax=ax)
    
    # Add vertical lines for means
    for phase in order:
        subset = events[events["enso_phase"] == phase]["mean_tilt"]
        if len(subset) > 0:
            mean_val = subset.mean()
            ax.axvline(mean_val, color=palette[phase], linestyle='--', 
                      alpha=0.7, linewidth=1.5)
    
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Mean Tilt (deg)")
    ax.set_ylabel("Density")
    ax.set_title("Tilt Distribution by ENSO Phase (KDE)")
    ax.legend(loc='upper right')
    ax.grid(axis='both', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_FIG_DIST)
    print(f"Saved distribution figure to: {OUT_FIG_DIST}")
    plt.close()

    # =====================
    # Figure 3: Combined Statistics Summary
    # =====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    
    # Panel 1: Violin Plot
    ax1 = axes[0]
    sns.violinplot(x="enso_phase", y="mean_tilt", data=events, order=order,
                   hue="enso_phase", palette=palette, ax=ax1, inner="quartile", legend=False)
    ax1.set_title("Violin Plot")
    ax1.set_xlabel("ENSO Phase")
    ax1.set_ylabel("Mean Tilt (deg)")
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Panel 2: Bar Chart with Error Bars
    ax2 = axes[1]
    means = []
    stds = []
    sems = []
    ns = []
    for phase in order:
        subset = events[events["enso_phase"] == phase]["mean_tilt"]
        means.append(subset.mean() if len(subset) > 0 else 0)
        stds.append(subset.std() if len(subset) > 1 else 0)
        sems.append(subset.sem() if len(subset) > 1 else 0)
        ns.append(len(subset))
    
    x_pos = np.arange(len(order))
    bars = ax2.bar(x_pos, means, yerr=sems, capsize=5, 
                   color=[palette[p] for p in order], edgecolor='black', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{o}\n(N={n})" for o, n in zip(order, ns)])
    ax2.set_ylabel("Mean Tilt (deg)")
    ax2.set_title("Mean Tilt ± SEM")
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels on bars
    for bar, m, s in zip(bars, means, sems):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                f'{m:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Panel 3: Scatter (ONI vs Tilt)
    ax3 = axes[2]
    for phase in order:
        subset = events[events["enso_phase"] == phase]
        ax3.scatter(subset["oni_avg"], subset["mean_tilt"], 
                   c=palette[phase], label=phase, alpha=0.7, s=40, edgecolors='k', linewidths=0.5)
    
    # Add regression line for all data
    valid = events.dropna(subset=["oni_avg", "mean_tilt"])
    if len(valid) > 2:
        slope, intercept, r_val, p_val, _ = stats.linregress(valid["oni_avg"], valid["mean_tilt"])
        x_line = np.linspace(valid["oni_avg"].min(), valid["oni_avg"].max(), 100)
        y_line = slope * x_line + intercept
        ax3.plot(x_line, y_line, 'k--', linewidth=1.5, 
                label=f"r={r_val:.2f}, p={p_val:.3f}")
    
    ax3.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax3.axvline(0, color='gray', linestyle='-', alpha=0.3)
    ax3.axvline(ONI_ELNINO_THRESH, color='red', linestyle=':', alpha=0.5)
    ax3.axvline(ONI_LANINA_THRESH, color='blue', linestyle=':', alpha=0.5)
    ax3.set_xlabel("ONI (°C)")
    ax3.set_ylabel("Mean Tilt (deg)")
    ax3.set_title("ONI vs Tilt Scatter")
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(linestyle='--', alpha=0.3)
    
    plt.suptitle("MJO Tilt Statistics by ENSO Phase", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_FIG_COMBINED)
    print(f"Saved combined figure to: {OUT_FIG_COMBINED}")
    plt.close()
    
    print("\n" + "="*40)
    print("ALL PROCESSING COMPLETE")
    print("="*40)

if __name__ == "__main__":
    main()
