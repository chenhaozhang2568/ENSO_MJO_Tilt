
# -*- coding: utf-8 -*-
"""
plot_olr_hovmoller.py: OLR Hovmöller 时-经度图绘制脚本

================================================================================
功能描述：
    本脚本生成逐年的 OLR 异常场 Hovmöller 图（时间-经度图），用于可视化 MJO 的东传过程。

时间范围：
    每年冬季（11月-次年4月），覆盖 1979-2022 年

主要特征：
    1. OLR 异常填色图（蓝色=对流增强，红色=对流抑制）
    2. -15 W/m² 等值线（标记活跃对流区边界）
    3. MJO 事件传播趋势线（基于对流中心轨迹线性拟合）
    4. 每年输出一张独立图片

输出目录：
    outputs/figures/hovmoller/
"""

import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import pandas as pd
from pathlib import Path

# Adjust path to import config if needed
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Input/Output paths
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
FAILED_EVENTS_CSV = r"E:\Datas\Derived\mjo_failed_events_step3_1979-2022.csv"
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\hovmoller_yearly_olr_recon_contour-15_1979-2022")

# Settings
START_YEAR = 1979
END_YEAR = 2022
LON_RANGE = (20, 220)  # 20E to 140W (140W = 360-140 = 220)
# Ticks: 20E, 60E, 100E, 140E, 180, 140W (220)
XTICK_LOCS = [20, 60, 100, 140, 180, 220]
XTICK_LABELS = ["20E", "60E", "100E", "140E", "180", "140W"]

OLR_LEVELS = np.arange(-75, 70, 5) # -75 to 65
CONTOUR_LEVEL = -15.0

def setup_colormap():
    """
    Create a discrete colormap: Blue (neg) -> White (-5 to 5) -> Red (pos)
    Bins every 10 units.
    """
    # Define boundaries: every 10 units, with white zone from -5 to 5
    # Bins: [-75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65]
    boundaries = [-75, -65, -55, -45, -35, -25, -15, -5, 5, 15, 25, 35, 45, 55, 65]
    
    # Number of colors = len(boundaries) - 1 = 14
    n_colors = len(boundaries) - 1
    
    # Use RdBu_r as base (Blue for negative, Red for positive)
    base_cmap = plt.cm.RdBu_r
    
    # Sample colors from the colormap
    # We want: strong blue for -75, gradually lighter, white around 0, gradually to strong red at 65
    # Map boundary midpoints to [0, 1] range for sampling
    colors = []
    for i in range(n_colors):
        mid = (boundaries[i] + boundaries[i+1]) / 2.0
        
        # Special handling for white zone [-5, 5]
        if -5 <= mid <= 5:
            colors.append('white')
        else:
            # Map to [0, 1] for colormap sampling
            # -75 -> 0 (strong blue), 0 -> 0.5 (white), 65 -> 1 (strong red)
            norm_val = (mid + 75) / (65 + 75)  # Normalize to [0, 1]
            colors.append(base_cmap(norm_val))
    
    # Create discrete colormap
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    
    return cmap, norm

def preprocess_data(ds):
    """
    Ensure 0-360 longitude and handle data ranges.
    """
    # Convert -180..180 to 0..360 if needed
    if ds.lon.min() < 0:
        ds.coords["lon"] = (ds.coords["lon"] + 360) % 360
        ds = ds.sortby("lon")
    
    # We need to cover 20 to 220.
    # If data is 0..360, we can directly slice if it's contiguous.
    # 20..220 is contiguous in 0..360.
    return ds

def get_winter_window(year):
    """
    Return (start_date, end_date) for the winter season ending in `year`.
    e.g. year=1980 -> Nov 1, 1979 to Apr 30, 1980.
    """
    start_date = f"{year-1}-11-01"
    end_date = f"{year}-04-30"
    return start_date, end_date

def fit_trend_line(times, lons):
    """
    Fit a linear trend to longitude vs time.
    times: numeric (e.g. days since start)
    lons: degrees
    Returns: x_pred, y_pred (points for plotting the line)
    """
    if len(times) < 2:
        return None, None
    
    # Simple linear regression
    # Handle longitude roughly (assuming no dateline crossing issues within short segments for now, 
    # or assuming inputs are already unwrapped if needed. 
    # For global MJO, crossing 360 is common.
    # If data jumps 350 -> 10, unwrapping is needed.)
    
    lons_unwrapped = np.unwrap(lons, period=360)
    
    A = np.vstack([times, np.ones(len(times))]).T
    m, c = np.linalg.lstsq(A, lons_unwrapped, rcond=None)[0]
    
    # Generate line points
    y_pred = m * times + c
    
    return times, y_pred

def plot_event_trend_lines(ax, df_events, track_full, win_start, win_end,
                           color, linewidth=2.0, lon_fit_range=(60, 180)):
    """
    Plot trend lines for events, but only fit+draw within lon_fit_range.
    lon_fit_range: (lon_min, lon_max) in 0-360 degE.
    """
    lon_min, lon_max = lon_fit_range

    # Filter events overlapping with this window
    mask = (df_events['start_date'] <= win_end) & (df_events['end_date'] >= win_start)
    events_in_window = df_events[mask]

    for _, event in events_in_window.iterrows():
        t0 = max(event['start_date'], win_start)
        t1 = min(event['end_date'], win_end)
        if t0 >= t1:
            continue

        track_slice = track_full.sel(time=slice(t0, t1))
        if track_slice.sizes["time"] < 2:
            continue

        t_vals = mdates.date2num(track_slice.time.values)
        l_vals = (track_slice.values + 360) % 360  # to 0-360

        # valid finite
        valid = np.isfinite(l_vals)
        if np.sum(valid) < 2:
            continue

        t_fit = t_vals[valid]
        l_fit = l_vals[valid]

        # -------- NEW: only keep points within [60,180] for fitting --------
        in_range = (l_fit >= lon_min) & (l_fit <= lon_max)
        if np.sum(in_range) < 2:
            continue

        t_fit = t_fit[in_range]
        l_fit = l_fit[in_range]

        # unwrap for linear fit (safe here since range doesn't cross dateline)
        l_rad = np.deg2rad(l_fit)
        l_unwrapped = np.rad2deg(np.unwrap(l_rad))

        # Linear Fit
        coeffs = np.polyfit(t_fit, l_unwrapped, 1)
        poly = np.poly1d(coeffs)

        # -------- NEW: evaluate on a dense time grid so line is smooth --------
        t_pred = np.linspace(t_fit.min(), t_fit.max(), 80)
        y_pred_unwrapped = poly(t_pred)
        y_pred_wrapped = y_pred_unwrapped % 360

        # -------- NEW: clip to [60,180] for plotting --------
        mask_plot = (y_pred_wrapped >= lon_min) & (y_pred_wrapped <= lon_max)

        plot_t = t_pred.copy()                 # NO NaN in time
        plot_l = y_pred_wrapped.copy()
        plot_l[~mask_plot] = np.nan            # NaN only in lon to break segments

        y_dt = mdates.num2date(plot_t)         # safe now
        if np.all(~np.isfinite(plot_l)):
            continue
        ax.plot(plot_l, y_dt, color=color, linewidth=linewidth)


def plot_hovmoller(ds_sub, year, output_path):
    """
    ds_sub: Dataset sliced to the time window and longitude range.
            Should contain 'olr_recon' and 'center_lon_track'.
    """
    # Prepare data for plotting
    data = ds_sub["olr_recon"]
    
    # Data selection (Lon Range)
    data = data.sel(lon=slice(LON_RANGE[0], LON_RANGE[1]))
    lon = data.lon.values
    time = data.time.values
    
    # Create Figure
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # --- 1. Shading ---
    cmap, norm = setup_colormap()  # Get discrete colormap and norm
    
    # Use contourf for smooth edges instead of blocky pcolormesh
    # Extract boundaries from norm for contourf levels
    levels = norm.boundaries
    cf = ax.contourf(lon, time, data.values, levels=levels, cmap=cmap, norm=norm, extend='both')
    
    # --- 2. Contours (-15 W/m2) ---
    cs = ax.contour(lon, time, data.values, levels=[CONTOUR_LEVEL], colors=['blue'], linewidths=1.5)
    
    # --- 3. Trend Lines (Event-based) ---
    # Current window boundaries
    win_start = pd.to_datetime(time[0])
    win_end = pd.to_datetime(time[-1])
    
    # Track data (full window)
    track_full = ds_sub["center_lon_track"]
    
    # 3a. Plot FAILED events (RED lines) - draw first so success events appear on top
    if Path(FAILED_EVENTS_CSV).exists():
        df_failed = pd.read_csv(FAILED_EVENTS_CSV)
        df_failed['start_date'] = pd.to_datetime(df_failed['start_date'])
        df_failed['end_date'] = pd.to_datetime(df_failed['end_date'])
        
        plot_event_trend_lines(ax, df_failed, track_full, win_start, win_end,
                       color='red', linewidth=2.0, lon_fit_range=(60, 180))
    
    # 3b. Plot SUCCESS events (BLACK lines) - draw on top
    if Path(EVENTS_CSV).exists():
        df_events = pd.read_csv(EVENTS_CSV)
        df_events['start_date'] = pd.to_datetime(df_events['start_date'])
        df_events['end_date'] = pd.to_datetime(df_events['end_date'])
        
        plot_event_trend_lines(ax, df_events, track_full, win_start, win_end,
                       color='black', linewidth=2.0, lon_fit_range=(60, 180))


    # ------------------------------------

    # --- 4. Formatting ---
    
    # Y-Axis Interval (Months)
    ax.yaxis.set_major_locator(mdates.MonthLocator(bymonth=[11, 12, 1, 2, 3, 4], bymonthday=1))
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%b%d'))
    
    # Custom Y-Ticks: NOV1, DEC1, JAN1, FEB1, MAR1, APR1, APR30
    # Manually setting might be easier to strictly match format "MMM+D" (upper)
    # Let's generate the specific tick dates for this window
    
    y_year_start = int(year) - 1
    y_year_end = int(year)
    
    ticks_dates = [
        pd.Timestamp(f"{y_year_start}-11-01"),
        pd.Timestamp(f"{y_year_start}-12-01"),
        pd.Timestamp(f"{y_year_end}-01-01"),
        pd.Timestamp(f"{y_year_end}-02-01"),
        pd.Timestamp(f"{y_year_end}-03-01"),
        pd.Timestamp(f"{y_year_end}-04-01"),
        pd.Timestamp(f"{y_year_end}-04-30"),
    ]
    ax.set_yticks(ticks_dates)
    ax.set_yticklabels([d.strftime("%b%d").upper() for d in ticks_dates])
    
    # X-Axis
    ax.set_xticks(XTICK_LOCS)
    ax.set_xticklabels(XTICK_LABELS)
    ax.set_xlabel("Longitude (degE)") # Optional based on provided image, usually just labels
    ax.set_ylabel("Time")
    
    # Limits
    ax.set_xlim(LON_RANGE)
    ax.set_ylim(ticks_dates[0], ticks_dates[-1])
    
    # Vertical Lines
    ax.axvline(x=60, color='black', linewidth=1.0)
    ax.axvline(x=180, color='black', linewidth=1.0)
    
    # Titles/Labels
    # Top Left: (a)
    ax.text(0.02, 1.02, "(a)", transform=ax.transAxes, fontsize=14, fontweight='bold', va='bottom')
    # Center: OLR
    ax.text(0.5, 1.02, "OLR", transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center', va='bottom')
    # Right: Year
    ax.text(0.98, 1.02, str(year), transform=ax.transAxes, fontsize=14, fontweight='bold', ha='right', va='bottom')
    
    # Colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    # cbar.set_ticks(np.arange(-60, 61, 20)) # Match user scaling roughly?
    cbar.set_label("OLR anomalies (W m$^{-2}$)")
    
    # Save
    plt.tight_layout()
    out_file = output_path / f"hovmoller_olr_recon_{year}_contour-10.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Generated: {out_file}")

def main():
    # 1. Load Data
    if not Path(STEP3_NC).exists():
        print(f"File not found: {STEP3_NC}")
        return

    ds = xr.open_dataset(STEP3_NC)
    
    # Preprocess
    ds = preprocess_data(ds)
    
    # Check output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Loop Years
    for year in range(START_YEAR, END_YEAR + 1):
        # Define window
        s_date, e_date = get_winter_window(year)
        
        # Slice
        try:
            ds_win = ds.sel(time=slice(s_date, e_date))
            if ds_win.sizes["time"] < 10:
                print(f"Skipping {year}: Insufficient data ({ds_win.sizes['time']} days)")
                continue
            
            # Plot
            plot_hovmoller(ds_win, year, OUT_DIR)
            
        except KeyError:
            print(f"Skipping {year}: Data range not found.")
            continue
        except Exception as e:
            print(f"Error plotting {year}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
