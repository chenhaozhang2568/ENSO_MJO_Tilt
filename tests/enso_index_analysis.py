# -*- coding: utf-8 -*-
"""
enso_index_analysis.py — ENSO 指数综合分析

功能：
    对比 ONI 与 Niño3.4 SSTA 时间序列并可视化逐事件 ENSO 分类，
    检验两种指数的一致性。
输入：
    mjo_events_step3_1979-2022.csv, oni.ascii.txt,
    era5_sst_monthly_nino34_1979_2024.nc
输出：
    figures/ 下的 ONI-SSTA 时间序列图和逐事件图
用法：
    python tests/enso_index_analysis.py           # 全部
    python tests/enso_index_analysis.py timeseries # 时间序列
    python tests/enso_index_analysis.py events     # 逐事件
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# PATHS
# ======================
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
ONI_TXT    = r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt"
SST_NC     = r"E:\Datas\ERA5\raw\single_level\sst\era5_sst_monthly_nino34_1979_2024.nc"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\enso_index")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ======================
# SETTINGS
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"
ELNINO_THR = 0.5
LANINA_THR = -0.5
CLIM_START = "1991-01-01"
CLIM_END   = "2020-12-31"
SST_VAR    = None
SST_IN_KELVIN = True


# ======================
# SHARED HELPERS
# ======================
def parse_oni(path: str) -> pd.Series:
    """Parse ONI ascii → monthly Series."""
    seas_map = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
        "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12
    }
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        mon = seas_map.get(parts[0])
        if mon is None:
            continue
        data.append((pd.Timestamp(year=int(parts[1]), month=mon, day=1), float(parts[3])))
    s = pd.Series(dict(data)).sort_index()
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp(how="start")
    s.name = "ONI"
    return s


def standardize_latlon(ds):
    ren = {}
    if "latitude" in ds.coords: ren["latitude"] = "lat"
    if "longitude" in ds.coords: ren["longitude"] = "lon"
    return ds.rename(ren) if ren else ds


def to_lon_360(ds):
    if "lon" in ds.coords and np.nanmin(ds["lon"].values) < 0:
        ds = ds.assign_coords(lon=(ds["lon"].values % 360)).sortby("lon")
    return ds


def pick_var(ds, prefer=None):
    if prefer and prefer in ds.data_vars:
        return prefer
    for c in ["sea_surface_temperature", "sst", "SST", "tos", "ts"]:
        if c in ds.data_vars:
            return c
    return list(ds.data_vars)[0]


def monthly_climatology(ts, clim_start, clim_end):
    base = ts.loc[pd.Timestamp(clim_start):pd.Timestamp(clim_end)]
    clim12 = base.groupby(base.index.month).mean()
    out = ts.index.to_series().apply(lambda d: clim12.loc[d.month]).astype(float)
    out.index = ts.index
    return out


def load_sst_ssta():
    """加载 SST → SSTA（3个月滑动均值）"""
    ds = xr.open_dataset(SST_NC)
    ds = standardize_latlon(ds)
    ds = to_lon_360(ds)
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.dims:
        ds = ds.rename_dims({"valid_time": "time"}).rename_vars({"valid_time": "time"})
    vname = pick_var(ds, SST_VAR)
    sst = ds[vname].sel(time=slice(START_DATE, END_DATE))
    # average if has lat/lon
    if "lat" in sst.dims and "lon" in sst.dims:
        w = np.cos(np.deg2rad(sst["lat"]))
        sst = sst.weighted(w).mean(("lat", "lon"), skipna=True)
    elif "lat" in sst.dims:
        w = np.cos(np.deg2rad(sst["lat"]))
        sst = sst.weighted(w).mean(("lat",), skipna=True)
    elif "lon" in sst.dims:
        sst = sst.mean(("lon",), skipna=True)
    sst_pd = sst.to_series()
    sst_pd.index = pd.to_datetime(sst_pd.index).to_period("M").to_timestamp(how="start")
    sst_pd = sst_pd.sort_index()
    if SST_IN_KELVIN:
        sst_pd = sst_pd - 273.15
    clim = monthly_climatology(sst_pd, CLIM_START, CLIM_END)
    ssta = sst_pd - clim
    ssta_3 = ssta.rolling(3, center=True, min_periods=2).mean()
    ssta_3.name = "Niño3.4 SSTA 3mo"
    return ssta_3


def shade_runs(ax, times, mask, facecolor, alpha, label):
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]
    first = True
    for s, e in zip(starts, ends):
        t0 = times[s]
        t1_ext = (times[e - 1].to_period("M") + 1).to_timestamp(how="start")
        ax.axvspan(t0, t1_ext, facecolor=facecolor, alpha=alpha, linewidth=0,
                   label=(label if first else None))
        first = False


# ============================================================
# 1. TIME SERIES (from oni_ssta.py)
# ============================================================
def run_timeseries():
    """ONI vs Niño3.4 SSTA 时间序列"""
    print("\n[1] ONI vs SSTA time series...")
    oni = parse_oni(ONI_TXT).loc[pd.Timestamp(START_DATE):pd.Timestamp(END_DATE)]
    ssta_3 = load_sst_ssta()
    df = pd.concat([oni, ssta_3], axis=1).dropna()
    en_mask = df["ONI"].values >= ELNINO_THR
    ln_mask = df["ONI"].values <= LANINA_THR
    corr = df["ONI"].corr(df[ssta_3.name])

    fig, ax = plt.subplots(figsize=(16, 5.5), dpi=160)
    shade_runs(ax, df.index, en_mask, "red", 0.08, "El Niño")
    shade_runs(ax, df.index, ln_mask, "blue", 0.08, "La Niña")
    ax.plot(df.index, df["ONI"], lw=1.4, label="ONI (NOAA CPC)")
    ax.plot(df.index, df[ssta_3.name], lw=1.2, label="Niño3.4 SSTA (ERA5)")
    ax.axhline(0, lw=0.8)
    ax.axhline(ELNINO_THR, ls="--", lw=0.9)
    ax.axhline(LANINA_THR, ls="--", lw=0.9)
    ax.set_title(f"ONI vs Niño3.4 SSTA (1979–2022), corr={corr:.2f}")
    ax.set_ylabel("°C")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=3, fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "oni_vs_nino34_ssta_1979_2022.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")
    print(f"  corr = {corr:.3f}")


# ============================================================
# 2. EVENT PLOT (from event_oni_ssta_plot.py)
# ============================================================
def run_events():
    """逐事件 ENSO 指数可视化"""
    print("\n[2] Event ONI/SSTA plot...")
    ev = pd.read_csv(EVENTS_CSV)
    ev["start_date"] = pd.to_datetime(ev["start_date"])
    ev["end_date"] = pd.to_datetime(ev["end_date"])
    ev = ev.sort_values("event_id").reset_index(drop=True)

    oni = parse_oni(ONI_TXT)
    ssta_3 = load_sst_ssta()

    rows = []
    for _, r in ev.iterrows():
        s, e = pd.Timestamp(r["start_date"]), pd.Timestamp(r["end_date"])
        per = pd.period_range(s.to_period("M"), e.to_period("M"), freq="M")
        months = per.to_timestamp(how="start")
        oni_mean = oni.reindex(months).mean()
        ssta_mean = ssta_3.reindex(months).mean()
        cls = "ElNino" if (np.isfinite(oni_mean) and oni_mean >= ELNINO_THR) else \
              "LaNina" if (np.isfinite(oni_mean) and oni_mean <= LANINA_THR) else "Neutral"
        rows.append({"event_id": int(r["event_id"]), "oni_mean": oni_mean,
                      "ssta_mean": ssta_mean, "enso_class": cls})
    df = pd.DataFrame(rows).sort_values("event_id")

    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=160)
    x = df["event_id"].values
    ax.plot(x, df["oni_mean"], marker="o", lw=1.4, ms=3.8, label="ONI")
    ax.plot(x, df["ssta_mean"], marker="s", lw=1.2, ms=3.6, label="SSTA")
    ax.axhline(0, lw=0.8)
    ax.axhline(ELNINO_THR, ls="--", lw=0.9)
    ax.axhline(LANINA_THR, ls="--", lw=0.9)
    for xi, cls in zip(x, df["enso_class"]):
        if cls in ("ElNino", "LaNina"):
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.08)
    ax.set_xlabel("Event ID")
    ax.set_ylabel("°C")
    ax.set_title("Event-mean ONI vs Niño3.4 SSTA")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=9)
    fig.tight_layout()
    out = FIG_DIR / "event_oni_ssta.png"
    fig.savefig(out)
    plt.close(fig)

    csv_out = out.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    print(f"  Saved: {out}, {csv_out}")


# ============================================================
# MAIN
# ============================================================
ANALYSES = {"timeseries": run_timeseries, "events": run_events}


def main():
    print("=" * 70)
    print("ENSO Index Analysis")
    print("=" * 70)
    if len(sys.argv) > 1:
        name = sys.argv[1].lower()
        if name in ANALYSES:
            ANALYSES[name]()
        else:
            print(f"Unknown: {name}. Available: {list(ANALYSES.keys())}")
    else:
        for func in ANALYSES.values():
            func()
    print("\nDone!")


if __name__ == "__main__":
    main()
