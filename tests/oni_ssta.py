# -*- coding: utf-8 -*-
"""
oni_ssta.py: ONI 与 Niño3.4 SSTA 时间序列对比图

================================================================================
功能描述：
    本脚本绘制 1979-2022 年 ONI（Oceanic Niño Index）与 ERA5 计算的 Niño3.4 SSTA
    时间序列对比图，并标注 El Niño 和 La Niña 事件期。

数据来源：
    - ONI: NOAA CPC 官方发布的三个月滑动平均指数
    - SSTA: 基于 ERA5 月平均 SST 计算，减去 1991-2020 气候态

主要输出：
    1. 双线时间序列图（ONI vs SSTA_3mo）
    2. El Niño（红色）/La Niña（蓝色）期间阴影标注
    3. ONI 与 SSTA 的相关系数

ENSO 临界值：
    - El Niño: ONI ≥ +0.5°C
    - La Niña: ONI ≤ -0.5°C

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# PATHS (EDIT THESE)
# =========================
ONI_TXT = r"E:\Datas\ClimateIndex\raw\oni\oni.ascii.txt"
SST_NC  = r"E:\Datas\ERA5\raw\single_level\sst\era5_sst_monthly_nino34_1979_2024.nc"
OUT_FIG = r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\oni_vs_nino34_ssta_1979_2022.png"


# =========================
# SETTINGS
# =========================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"

ELNINO_THR = 0.5
LANINA_THR = -0.5

# climatology base period for SSTA (change if you want)
CLIM_START = "1991-01-01"
CLIM_END   = "2020-12-31"

# If auto-detect fails, set explicitly, e.g. "sst" or "sea_surface_temperature"
SST_VAR = None

# If your SST is already in Celsius, set this False
SST_IN_KELVIN = True


# =========================
# HELPERS
# =========================
def parse_oni(path: str) -> pd.Series:
    """
    ONI ascii: season code + year + ... + anomaly
    Map season to center month: DJF->Jan, JFM->Feb, ..., NDJ->Dec.
    Return monthly series at Month Start (MS-like), compatible with older pandas.
    """
    seas_map = {
        "DJF": 1,  "JFM": 2,  "FMA": 3,  "MAM": 4,  "AMJ": 5,  "MJJ": 6,
        "JJA": 7,  "JAS": 8,  "ASO": 9,  "SON": 10, "OND": 11, "NDJ": 12
    }
    data = []
    with open(path, "r") as f:
        lines = f.readlines()

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
        data.append((pd.Timestamp(year=yr, month=mon, day=1), anom))

    s = pd.Series(dict(data)).sort_index()
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
    """
    Monthly climatology (12 values) computed on base period, then expanded to full index.
    """
    base = ts.loc[pd.Timestamp(clim_start):pd.Timestamp(clim_end)]
    if base.dropna().empty:
        raise ValueError("Climatology base period has no data. Adjust CLIM_START/CLIM_END.")
    clim12 = base.groupby(base.index.month).mean()
    out = ts.index.to_series().apply(lambda d: clim12.loc[d.month]).astype(float)
    out.index = ts.index
    return out


def shade_runs(ax, times: pd.DatetimeIndex, mask: np.ndarray, facecolor: str, alpha: float, label: str):
    """
    Shade contiguous True runs on a time axis.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return

    # find run boundaries
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends   = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size]

    first = True
    for s, e in zip(starts, ends):
        t0 = times[s]
        t1 = times[e - 1]
        # extend to next month start for full-span look
        t1_ext = (t1.to_period("M") + 1).to_timestamp(how="start")
        ax.axvspan(t0, t1_ext, facecolor=facecolor, alpha=alpha, linewidth=0,
                   label=(label if first else None))
        first = False


# =========================
# MAIN
# =========================
def main():
    # --- ONI
    oni = parse_oni(ONI_TXT).loc[pd.Timestamp(START_DATE):pd.Timestamp(END_DATE)]

    # --- SST -> Niño3.4 SSTA (3-mo running mean)
    ds = xr.open_dataset(SST_NC)
    ds = standardize_latlon(ds)
    ds = to_lon_360(ds)

    # handle ERA5 monthly time name
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    elif "valid_time" in ds.dims:
        ds = ds.rename_dims({"valid_time": "time"}).rename_vars({"valid_time": "time"})

    vname = pick_var(ds, SST_VAR)
    sst = ds[vname]

    if "time" not in sst.dims:
        raise ValueError(f"SST variable {vname} has no 'time' dim. dims={sst.dims}")

    sst = sst.sel(time=slice(START_DATE, END_DATE))

    # if still has lat/lon dims, area-average with cos(lat) weights
    if ("lat" in sst.dims) and ("lon" in sst.dims):
        w = np.cos(np.deg2rad(sst["lat"]))
        sst = sst.weighted(w).mean(("lat", "lon"), skipna=True)
    elif ("lat" in sst.dims):
        w = np.cos(np.deg2rad(sst["lat"]))
        sst = sst.weighted(w).mean(("lat",), skipna=True)
    elif ("lon" in sst.dims):
        sst = sst.mean(("lon",), skipna=True)

    sst_pd = sst.to_series()
    sst_pd.index = pd.to_datetime(sst_pd.index).to_period("M").to_timestamp(how="start")
    sst_pd = sst_pd.sort_index()

    if SST_IN_KELVIN:
        sst_pd = sst_pd - 273.15

    sst_pd.name = "Niño3.4 SST (C)"

    clim = monthly_climatology(sst_pd, CLIM_START, CLIM_END)
    ssta = (sst_pd - clim).rename("Niño3.4 SSTA (C)")
    ssta_3 = ssta.rolling(3, center=True, min_periods=2).mean().rename("Niño3.4 SSTA 3-mo rm (C)")

    # align monthly index
    df = pd.concat([oni, ssta_3], axis=1).dropna()

    # ENSO masks from ONI
    en_mask = df["ONI"].values >= ELNINO_THR
    ln_mask = df["ONI"].values <= LANINA_THR

    # correlation (optional but useful)
    corr = df["ONI"].corr(df["Niño3.4 SSTA 3-mo rm (C)"])

    # --- plot
    fig, ax = plt.subplots(figsize=(16, 5.5), dpi=160)

    # shade ENSO periods (time-based)
    shade_runs(ax, df.index, en_mask, facecolor="red",  alpha=0.08, label="El Niño (ONI ≥ 0.5)")
    shade_runs(ax, df.index, ln_mask, facecolor="blue", alpha=0.08, label="La Niña (ONI ≤ -0.5)")

    # lines
    ax.plot(df.index, df["ONI"], linewidth=1.4, label="ONI (NOAA CPC)")
    ax.plot(df.index, df["Niño3.4 SSTA 3-mo rm (C)"], linewidth=1.2, label="Niño3.4 SSTA (ERA5 SST, 3-mo rm)")

    # threshold lines
    ax.axhline(0.0, linewidth=0.8)
    ax.axhline(ELNINO_THR, linestyle="--", linewidth=0.9)
    ax.axhline(LANINA_THR, linestyle="--", linewidth=0.9)

    ax.set_title(f"ONI vs Niño3.4 SSTA (1979–2022), corr={corr:.2f}")
    ax.set_ylabel("°C")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.25)
    ax.legend(ncols=3, fontsize=9)

    fig.tight_layout()
    Path(OUT_FIG).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG)
    plt.close(fig)

    print("Saved:", OUT_FIG)
    print("corr(ONI, SSTA_3mo) =", corr)


if __name__ == "__main__":
    main()
