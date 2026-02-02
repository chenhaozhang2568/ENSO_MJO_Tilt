# -*- coding: utf-8 -*-
"""
event_oni_ssta_plot.py: 逐事件 ENSO 指数可视化

================================================================================
功能描述：
    本脚本绘制每个 MJO 事件期间对应的 ONI 和 Niño3.4 SSTA 指数，用于验证事件的
    ENSO 相位归属。

主要输出：
    1. 逐事件 ONI 和 SSTA 柱状/折线图
    2. El Niño / Neutral / La Niña 分类标记
    3. x 轴为事件 ID，y 轴为平均指数值

用途：
    检查 MJO 事件 ENSO 分类是否与 ONI 标准一致，排除分类错误。

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# PATHS (EDIT THESE)
# =========================
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"  # <- 你的事件表
ONI_TXT    = r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt"                   # <- 你的 ONI ascii
SST_NC     = r"E:\Datas\ERA5\raw\single_level\sst\era5_sst_monthly_nino34_1979_2024.nc"  # <- 你下载的月平均 SST

OUT_FIG    = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\event_oni_ssta.png"


# =========================
# SETTINGS
# =========================
# ENSO threshold (standard ONI)
ELNINO_THR = 0.5
LANINA_THR = -0.5

# SSTA climatology base period (you can change)
CLIM_START = "1991-01-01"
CLIM_END   = "2020-12-31"

# If your SST variable name is not detected, set it explicitly:
SST_VAR = None  # e.g. "sst" or "sea_surface_temperature"


# =========================
# HELPERS
# =========================
def parse_oni(path: str) -> pd.Series:
    """
    Parse ONI ascii like:
    SEAS  YR  TOTAL  ANOM ...
    DJF  1979  ...   ...
    Map season to center month (DJF->Jan, JFM->Feb, ..., NDJ->Dec).
    Return monthly series indexed by Month Start.
    """
    seas_map = {
        "DJF": 1,  "JFM": 2,  "FMA": 3,  "MAM": 4,  "AMJ": 5,  "MJJ": 6,
        "JJA": 7,  "JAS": 8,  "ASO": 9,  "SON": 10, "OND": 11, "NDJ": 12
    }
    data = []
    with open(path, "r") as f:
        lines = f.readlines()
    # skip header line
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        seas = parts[0]
        yr = int(parts[1])
        anom = float(parts[3])
        mon = seas_map.get(seas)
        if mon is None:
            continue
        dt = pd.Timestamp(year=yr, month=mon, day=1)
        data.append((dt, anom))

    s = pd.Series(dict(data)).sort_index()
    # ensure Month Start index
    s.index = pd.to_datetime(s.index).to_period("M").to_timestamp(how="start")
    s.name = "ONI"
    return s


def standardize_latlon(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "latitude" in ds.coords: ren["latitude"] = "lat"
    if "longitude" in ds.coords: ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    return ds


def to_lon_360(ds: xr.Dataset) -> xr.Dataset:
    if "lon" not in ds.coords:
        return ds
    lon = ds["lon"].values
    if np.nanmin(lon) < 0:
        ds = ds.assign_coords(lon=(lon % 360)).sortby("lon")
    return ds


def pick_var(ds: xr.Dataset, prefer: str | None) -> str:
    if prefer is not None:
        if prefer in ds.data_vars:
            return prefer
        raise KeyError(f"SST_VAR={prefer} not found. Available: {list(ds.data_vars)}")

    for cand in ["sea_surface_temperature", "sst", "SST", "tos", "ts"]:
        if cand in ds.data_vars:
            return cand
    return list(ds.data_vars)[0]


def monthly_climatology(ts: pd.Series, clim_start: str, clim_end: str) -> pd.Series:
    """Monthly climatology (12-month) computed on a base period."""
    base = ts.loc[pd.Timestamp(clim_start):pd.Timestamp(clim_end)]
    if base.dropna().empty:
        raise ValueError("Climatology base period has no data. Adjust CLIM_START/CLIM_END.")
    clim = base.groupby(base.index.month).mean()
    # expand to full time
    return ts.index.to_series().apply(lambda d: clim.loc[d.month]).astype(float).set_axis(ts.index)


def event_months(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return month-starts covered by [start, end]. Compatible with older pandas."""
    p0 = start.to_period("M")
    p1 = end.to_period("M")
    # period_range 是按月的 PeriodIndex，不需要 "MS"
    per = pd.period_range(p0, p1, freq="M")
    # 转成每月月初
    return per.to_timestamp(how="start")


def classify_enso(x: float) -> str:
    if np.isfinite(x) and x >= ELNINO_THR:
        return "ElNino"
    if np.isfinite(x) and x <= LANINA_THR:
        return "LaNina"
    return "Neutral"


# =========================
# MAIN
# =========================
def main():
    # --- load events
    ev = pd.read_csv(EVENTS_CSV)
    need_cols = {"event_id", "start_date", "end_date"}
    miss = need_cols - set(ev.columns)
    if miss:
        raise KeyError(f"EVENTS_CSV missing columns: {miss}. Found: {list(ev.columns)}")

    ev["start_date"] = pd.to_datetime(ev["start_date"])
    ev["end_date"] = pd.to_datetime(ev["end_date"])
    ev = ev.sort_values("event_id").reset_index(drop=True)

    # --- load ONI
    oni = parse_oni(ONI_TXT)

    # --- load SST monthly (Niño3.4 already cropped in your download)
    ds = xr.open_dataset(SST_NC)
    ds = standardize_latlon(ds)
    ds = to_lon_360(ds)

    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.dims:
        ds = ds.rename_dims({"valid_time": "time"}).rename_vars({"valid_time": "time"})
    vname = pick_var(ds, SST_VAR)

    sst = ds[vname]
    if "time" not in sst.dims:
        raise ValueError(f"SST variable {vname} has no 'time' dim. dims={sst.dims}")

    # If still has lat/lon dims (not fully averaged), average them (cos-lat weighted)
    if ("lat" in sst.dims) and ("lon" in sst.dims):
        w = np.cos(np.deg2rad(sst["lat"]))
        sst = sst.weighted(w).mean(("lat", "lon"), skipna=True)
    elif ("lat" in sst.dims) and ("lon" not in sst.dims):
        w = np.cos(np.deg2rad(sst["lat"]))
        sst = sst.weighted(w).mean(("lat",), skipna=True)
    elif ("lon" in sst.dims) and ("lat" not in sst.dims):
        sst = sst.mean(("lon",), skipna=True)

    # to pandas monthly
    sst_pd = sst.to_series()
    sst_pd.index = pd.to_datetime(sst_pd.index).to_period("M").to_timestamp(how="start")
    sst_pd = sst_pd.sort_index()

    # ERA5 SST is Kelvin -> Celsius
    # (If your SST is already Celsius, comment this out)
    sst_pd_c = sst_pd - 273.15
    sst_pd_c.name = "Niño3.4 SST (C)"

    # compute SSTA using monthly climatology base period
    clim = monthly_climatology(sst_pd_c, CLIM_START, CLIM_END)
    ssta = sst_pd_c - clim
    ssta.name = "Niño3.4 SSTA (C)"

    # optional: 3-month running mean to align with ONI definition style
    ssta_3 = ssta.rolling(3, center=True, min_periods=2).mean()
    ssta_3.name = "Niño3.4 SSTA 3mo (C)"

    # --- compute event mean ONI and SSTA
    rows = []
    for _, r in ev.iterrows():
        eid = int(r["event_id"])
        s = pd.Timestamp(r["start_date"])
        e = pd.Timestamp(r["end_date"])

        months = event_months(s, e)

        oni_evt = oni.reindex(months).mean()
        ssta_evt = ssta_3.reindex(months).mean()

        cls = classify_enso(oni_evt)

        rows.append(
            dict(
                event_id=eid,
                start_date=s.date().isoformat(),
                end_date=e.date().isoformat(),
                oni_mean=float(oni_evt) if np.isfinite(oni_evt) else np.nan,
                ssta_mean=float(ssta_evt) if np.isfinite(ssta_evt) else np.nan,
                enso_class=cls,
            )
        )

    df = pd.DataFrame(rows).sort_values("event_id")

    # --- plotting
    x = df["event_id"].values
    y1 = df["oni_mean"].values
    y2 = df["ssta_mean"].values

    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=160)

    ax.plot(x, y1, marker="o", linewidth=1.4, markersize=3.8, label="Event-mean ONI")
    ax.plot(x, y2, marker="s", linewidth=1.2, markersize=3.6, label="Event-mean Niño3.4 SSTA (3-mo rm)")

    # thresholds
    ax.axhline(0.0, linewidth=0.8)
    ax.axhline(ELNINO_THR, linestyle="--", linewidth=0.9)
    ax.axhline(LANINA_THR, linestyle="--", linewidth=0.9)

    # mark ENSO events
    en = df["enso_class"].values == "ElNino"
    ln = df["enso_class"].values == "LaNina"

    # emphasize points
    ax.scatter(x[en], y1[en], s=38, marker="o", edgecolors="black", linewidths=0.6, label="El Niño events (by ONI)")
    ax.scatter(x[ln], y1[ln], s=38, marker="o", edgecolors="black", linewidths=0.6, label="La Niña events (by ONI)")

    # background shading by event index (vertical bands)
    # This makes the ENSO-classified events visually obvious even with two lines.
    for xi, cls in zip(x, df["enso_class"].values):
        if cls == "ElNino":
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.08)
        elif cls == "LaNina":
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.08)

    ax.set_xlabel("Event ID")
    ax.set_ylabel("ONI / SSTA (°C)")
    ax.set_title("Event-mean ONI vs Niño3.4 SSTA (ERA5 monthly SST)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=2, fontsize=9)

    fig.tight_layout()
    Path(OUT_FIG).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)

    # save a small table too (useful for reporting)
    out_csv = Path(OUT_FIG).with_suffix(".csv")
    df.to_csv(out_csv, index=False)

    print("Figure saved:", OUT_FIG)
    print("Table saved :", str(out_csv))
    print(df.head(10))


if __name__ == "__main__":
    main()
