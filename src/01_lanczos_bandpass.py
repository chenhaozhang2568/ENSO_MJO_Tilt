# -*- coding: utf-8 -*-
"""
E:\Projects\ENSO_MJO_Tilt\src\01_lanczos_bandpass.py

STEP 2 (Hu & Li 2021): 20–100-day Lanczos bandpass filtering
- OLR (daily)
- ERA5 U850 and U200 (daily 00Z snapshots from monthly files)

Fixes included (based on your runtime errors):
1) Only open months within START_DATE..END_DATE (won't touch other years)
2) Rename latitude/longitude -> lat/lon when needed
3) Avoid netCDF coord write errors by sanitizing coord encodings
4) Fix apply_ufunc core-dim + multi-chunk error by rechunking time to a single chunk
5) Fix MergeError conflicting 'level' coord by DROPPING scalar 'level' coord after selecting 850/200

ADD (minimal change requested):
6) Add ERA5 omega (w, Pa/s) bandpass output (lat-mean over 15S–15N to keep file size practical for later Step6/7).
   - This only ADDS a new function and a new output file; existing steps are untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import xarray as xr

# ======================
# USER PATHS (KEEP SAME)
# ======================
OLR_PATH = r"E:\Datas\ClimateIndex\raw\olr\olr.day.mean.nc"
ERA5_DIR = r"E:\Datas\ERA5\raw\pressure_level\era5_1979-2022_quvwT_9_20 -180 20 180"

OUT_OLR_DIR = Path(r"E:\Datas\ClimateIndex\processed")
OUT_ERA5_PL_DIR = Path(r"E:\Datas\ERA5\processed\pressure_level")

OUT_OLR_DIR.mkdir(parents=True, exist_ok=True)
OUT_ERA5_PL_DIR.mkdir(parents=True, exist_ok=True)


# ======================
# TIME RANGE (EDIT HERE)
# ======================
START_DATE = "1979-01-01"
END_DATE = "2022-12-31"

# ======================
# REGION
# ======================
LAT_MIN, LAT_MAX = -20, 20
LON_MIN, LON_MAX = -180, 180

# ======================
# BANDPASS SETTINGS
# ======================
PERIOD_LOW = 20.0     # days
PERIOD_HIGH = 100.0   # days
BANDPASS_WINDOW = 121 # odd length (2M+1), default M=60

# ======================
# ADD: omega lat-mean band (paper uses 15S–15N)
# ======================
OMEGA_LATMEAN_MIN, OMEGA_LATMEAN_MAX = -15.0, 15.0


# ======================
# HELPERS
# ======================
def _guess_time_name(ds: xr.Dataset) -> str:
    for c in ["time", "valid_time", "date"]:
        if c in ds.coords:
            return c
    raise KeyError("Cannot find time coordinate among ['time','valid_time','date'].")

def _guess_level_name(ds: xr.Dataset) -> str:
    for c in ["level", "pressure_level", "plev", "isobaricInhPa"]:
        if c in ds.coords:
            return c
    raise KeyError("Cannot find pressure level coordinate among common names.")

def _rename_latlon_if_needed(ds: xr.Dataset) -> xr.Dataset:
    ren = {}
    if "latitude" in ds.coords:
        ren["latitude"] = "lat"
    if "longitude" in ds.coords:
        ren["longitude"] = "lon"
    if ren:
        ds = ds.rename(ren)
    return ds

def _to_lon_180(obj: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Ensure lon in [-180, 180) and sorted increasing."""
    if "lon" not in obj.coords:
        return obj
    lon = obj["lon"].values
    lon180 = ((lon + 180) % 360) - 180
    obj = obj.assign_coords(lon=lon180).sortby("lon")
    return obj

def _months_between(start_date: str, end_date: str) -> list[str]:
    """Inclusive months between start_date and end_date. Return ['YYYYMM', ...]."""
    y0, m0 = int(start_date[:4]), int(start_date[5:7])
    y1, m1 = int(end_date[:4]), int(end_date[5:7])
    out = []
    y, m = y0, m0
    while (y < y1) or (y == y1 and m <= m1):
        out.append(f"{y:04d}{m:02d}")
        m += 1
        if m == 13:
            y += 1
            m = 1
    return out

def lanczos_lowpass_weights(fc: float, window: int) -> np.ndarray:
    """
    Lanczos low-pass filter weights (daily sampling).
    fc: cutoff frequency in cycles/day (0 < fc < 0.5).
    window: odd length = 2M+1.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd.")
    if not (0.0 < fc < 0.5):
        raise ValueError("fc must be between 0 and 0.5 cycles/day for daily sampling.")
    M = (window - 1) // 2
    k = np.arange(-M, M + 1, dtype=float)
    h = 2.0 * fc * np.sinc(2.0 * fc * k)      # ideal lowpass
    sigma = np.sinc(k / (M + 1.0))            # Lanczos sigma factor
    w = h * sigma
    w = w / w.sum()
    return w

def lanczos_bandpass_weights(period_low: float, period_high: float, window: int) -> np.ndarray:
    """
    Bandpass = LP(1/period_low) - LP(1/period_high)
    Keep periods in [period_low, period_high] days.
    """
    fc_high = 1.0 / period_low
    fc_low = 1.0 / period_high
    w_high = lanczos_lowpass_weights(fc_high, window)
    w_low = lanczos_lowpass_weights(fc_low, window)
    return w_high - w_low

def apply_fir_convolution_along_time(da: xr.DataArray, weights: np.ndarray, time_dim: str) -> xr.DataArray:
    """
    Convolve along time dimension with FIR weights.
    IMPORTANT: time must be a single chunk if dask arrays are present.
    Edge effect: set first/last M samples to NaN.
    """
    M = (len(weights) - 1) // 2

    if hasattr(da.data, "chunks"):
        da = da.chunk({time_dim: -1})

    def _conv_1d(x: np.ndarray) -> np.ndarray:
        y = np.convolve(x, weights, mode="same")
        y[:M] = np.nan
        y[-M:] = np.nan
        return y

    out = xr.apply_ufunc(
        _conv_1d,
        da,
        input_core_dims=[[time_dim]],
        output_core_dims=[[time_dim]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float32],
        keep_attrs=True,
    )
    return out.astype(np.float32)

def _sanitize_coords_for_netcdf(ds: xr.Dataset) -> xr.Dataset:
    """Prevent netCDF4 'Invalid argument' errors on coordinates."""
    for c in list(ds.coords):
        ds[c].encoding = {}
        ds[c].encoding["_FillValue"] = None
    if "lat" in ds.coords:
        ds = ds.assign_coords(lat=ds["lat"].astype(np.float32))
        ds["lat"].encoding = {"_FillValue": None}
    if "lon" in ds.coords:
        ds = ds.assign_coords(lon=ds["lon"].astype(np.float32))
        ds["lon"].encoding = {"_FillValue": None}
    return ds

def _encoding_for_dataset(ds: xr.Dataset, complevel: int = 4) -> dict:
    enc: dict = {}
    for v in ds.data_vars:
        enc[v] = {"zlib": True, "complevel": complevel}
    for c in ds.coords:
        enc[c] = {"zlib": False, "_FillValue": None}
    return enc

def _drop_scalar_level_coord(da: xr.DataArray) -> xr.DataArray:
    """
    After selecting level=850/200, xarray keeps a scalar coord 'level' which
    conflicts when merging u850 and u200. Drop it.
    """
    if "level" in da.coords:
        da = da.reset_coords("level", drop=True)
    return da


# ======================
# STEP2: OLR
# ======================
def step2_filter_olr() -> Path:
    ds = xr.open_dataset(OLR_PATH, engine="netcdf4")
    ds = _rename_latlon_if_needed(ds)

    tname = _guess_time_name(ds)
    if tname != "time":
        ds = ds.rename({tname: "time"})

    var = "olr" if "olr" in ds.data_vars else list(ds.data_vars)[0]
    ds = _to_lon_180(ds)

    ds = ds.sel(time=slice(START_DATE, END_DATE))
    # --- robust lat slicing for increasing/decreasing coords ---
    lat_vals = ds["lat"].values
    if lat_vals.size >= 2 and (lat_vals[1] - lat_vals[0]) < 0:
        # decreasing latitude (e.g., 90 -> -90): slice must be reversed
        ds = ds.sel(lat=slice(LAT_MAX, LAT_MIN))
    else:
        ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX))

    ds = ds.sel(lon=slice(LON_MIN, LON_MAX))

    # ... (原有代码: ds = ds.sel(lon=slice(LON_MIN, LON_MAX)))

    # ================= NEW: 计算原始距平 (Raw Anomaly) =================
    # 目的：为了后续回归重建能恢复真实振幅，需计算去除季节循环后的异常值
    # 方法：减去多年日平均气候态 (Climatology)
    print("Calculating daily anomaly (removing seasonal cycle)...")
    climatology = ds[var].groupby("time.dayofyear").mean("time")
    olr_anom = ds[var].groupby("time.dayofyear") - climatology
    olr_anom = olr_anom.drop_vars("dayofyear").rename("olr_anom")
    # ===================================================================

    w_bp = lanczos_bandpass_weights(PERIOD_LOW, PERIOD_HIGH, BANDPASS_WINDOW)
    olr_bp = apply_fir_convolution_along_time(ds[var], w_bp, "time").rename("olr_bp")

    # 修改输出 Dataset，把 olr_anom 也加进去
    out = xr.Dataset({"olr_bp": olr_bp, "olr_anom": olr_anom}) # <--- 修改这里
    out["olr_bp"].attrs.update(ds[var].attrs)
    out["olr_bp"].attrs["filter"] = f"Lanczos bandpass {PERIOD_LOW:.0f}-{PERIOD_HIGH:.0f} day, window={BANDPASS_WINDOW}"
    # ... (原有代码: out["olr_bp"].attrs...)
    # 记得给新变量加属性，防止由计算产生的属性丢失
    out["olr_anom"].attrs = ds[var].attrs
    out["olr_anom"].attrs["note"] = "Raw daily anomaly (seasonal cycle removed), NO bandpass filter"
    out = _sanitize_coords_for_netcdf(out)
    out_path = OUT_OLR_DIR / f"olr_bp_{START_DATE[:4]}-{END_DATE[:4]}.nc"
    out.to_netcdf(out_path, engine="netcdf4", encoding=_encoding_for_dataset(out))
    return out_path


# ======================
# STEP2: ERA5 U850/U200
# ======================
def step2_filter_era5_u() -> Path:
    months = _months_between(START_DATE, END_DATE)

    files: list[str] = []
    missing_or_empty: list[str] = []
    for ym in months:
        f = os.path.join(ERA5_DIR, f"era5_pl_{ym}_00Z.nc")
        if os.path.exists(f) and os.path.getsize(f) > 0:
            files.append(f)
        else:
            missing_or_empty.append(f)

    if missing_or_empty:
        print("\n[WARNING] Missing or empty ERA5 monthly files in requested range (will cause gaps):")
        for m in missing_or_empty:
            print("  -", m)
        print()

    if not files:
        raise FileNotFoundError("No non-empty ERA5 monthly files found for the requested date range.")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        engine="netcdf4",
        parallel=False,
        chunks={"time": 31},  # ok; convolution will rechunk time to -1 internally
    )

    ds = _rename_latlon_if_needed(ds)

    tname = _guess_time_name(ds)
    lname = _guess_level_name(ds)
    if tname != "time":
        ds = ds.rename({tname: "time"})
    if lname != "level":
        ds = ds.rename({lname: "level"})

    ds = _to_lon_180(ds)

    ds = ds.sel(time=slice(START_DATE, END_DATE))

    # robust lat slicing for increasing/decreasing coords
    lat_vals = ds["lat"].values
    if lat_vals.size >= 2 and (lat_vals[1] - lat_vals[0]) < 0:
        ds = ds.sel(lat=slice(LAT_MAX, LAT_MIN))
    else:
        ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX))
    ds = ds.sel(lon=slice(LON_MIN, LON_MAX))

    if "u" in ds.data_vars:
        uvar = "u"
    elif "u_component_of_wind" in ds.data_vars:
        uvar = "u_component_of_wind"
    else:
        raise KeyError("Cannot find zonal wind variable 'u' (or 'u_component_of_wind') in ERA5 files.")

    # select levels and DROP scalar 'level' coord to avoid MergeError later
    u850 = _drop_scalar_level_coord(ds[uvar].sel(level=850)).rename("u850")
    u200 = _drop_scalar_level_coord(ds[uvar].sel(level=200)).rename("u200")

    w_bp = lanczos_bandpass_weights(PERIOD_LOW, PERIOD_HIGH, BANDPASS_WINDOW)
    u850_bp = apply_fir_convolution_along_time(u850, w_bp, "time").rename("u850_bp")
    u200_bp = apply_fir_convolution_along_time(u200, w_bp, "time").rename("u200_bp")

    # build dataset (no conflicting 'level' coord anymore)
    out = xr.Dataset({"u850_bp": u850_bp, "u200_bp": u200_bp})

    out["u850_bp"].attrs.update(u850.attrs)
    out["u200_bp"].attrs.update(u200.attrs)
    out["u850_bp"].attrs["filter"] = f"Lanczos bandpass {PERIOD_LOW:.0f}-{PERIOD_HIGH:.0f} day, window={BANDPASS_WINDOW}"
    out["u200_bp"].attrs["filter"] = f"Lanczos bandpass {PERIOD_LOW:.0f}-{PERIOD_HIGH:.0f} day, window={BANDPASS_WINDOW}"

    out = _sanitize_coords_for_netcdf(out)
    out_path = OUT_ERA5_PL_DIR / f"era5_u850_u200_bp_{START_DATE[:4]}-{END_DATE[:4]}.nc"
    out.to_netcdf(out_path, engine="netcdf4", encoding=_encoding_for_dataset(out))
    return out_path

# ======================
# ADD: STEP2: ERA5 omega (w) bandpass (lat-mean over 15S–15N)
# ======================
def step2_filter_era5_w() -> Path:
    """
    Add-on for Step2:
    - Read ERA5 'w' (Pa/s) from monthly files (daily 00Z)
    - Subset time and region same as other ERA5 steps
    - Meridional mean over 15S–15N (paper uses 15S–15N for profiles)
    - Apply the SAME Lanczos bandpass along time
    Output dims: (time, level, lon)  [lat removed by mean]
    """
    months = _months_between(START_DATE, END_DATE)

    files: list[str] = []
    missing_or_empty: list[str] = []
    for ym in months:
        f = os.path.join(ERA5_DIR, f"era5_pl_{ym}_00Z.nc")
        if os.path.exists(f) and os.path.getsize(f) > 0:
            files.append(f)
        else:
            missing_or_empty.append(f)

    if missing_or_empty:
        print("\n[WARNING] Missing or empty ERA5 monthly files in requested range (will cause gaps):")
        for m in missing_or_empty:
            print("  -", m)
        print()

    if not files:
        raise FileNotFoundError("No non-empty ERA5 monthly files found for the requested date range.")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        engine="netcdf4",
        parallel=False,
        chunks={"time": 31},  # ok; convolution will rechunk time to -1 internally
    )

    ds = _rename_latlon_if_needed(ds)

    tname = _guess_time_name(ds)
    lname = _guess_level_name(ds)
    if tname != "time":
        ds = ds.rename({tname: "time"})
    if lname != "level":
        ds = ds.rename({lname: "level"})

    ds = _to_lon_180(ds)

    ds = ds.sel(time=slice(START_DATE, END_DATE))

    # robust lat slicing for increasing/decreasing coords (use ERA5 region first)
    lat_vals = ds["lat"].values
    if lat_vals.size >= 2 and (lat_vals[1] - lat_vals[0]) < 0:
        ds = ds.sel(lat=slice(LAT_MAX, LAT_MIN))
    else:
        ds = ds.sel(lat=slice(LAT_MIN, LAT_MAX))
    ds = ds.sel(lon=slice(LON_MIN, LON_MAX))

    # omega variable in your files is exactly 'w'
    if "w" not in ds.data_vars:
        raise KeyError("Cannot find omega variable 'w' in ERA5 files (your sample showed data_vars include 'w').")

    # meridional mean over 15S–15N (paper)
    lat_vals2 = ds["lat"].values
    if lat_vals2.size >= 2 and (lat_vals2[1] - lat_vals2[0]) < 0:
        ds_w = ds["w"].sel(lat=slice(OMEGA_LATMEAN_MAX, OMEGA_LATMEAN_MIN))
    else:
        ds_w = ds["w"].sel(lat=slice(OMEGA_LATMEAN_MIN, OMEGA_LATMEAN_MAX))

    w_latmean = ds_w.mean("lat", skipna=True).rename("w_latmean")  # dims: time, level, lon

    w_bp = lanczos_bandpass_weights(PERIOD_LOW, PERIOD_HIGH, BANDPASS_WINDOW)
    w_latmean_bp = apply_fir_convolution_along_time(w_latmean, w_bp, "time").rename("w_bp")

    out = xr.Dataset({"w_bp": w_latmean_bp})
    out["w_bp"].attrs.update(ds["w"].attrs)
    out["w_bp"].attrs["filter"] = f"Lanczos bandpass {PERIOD_LOW:.0f}-{PERIOD_HIGH:.0f} day, window={BANDPASS_WINDOW}"
    out["w_bp"].attrs["lat_mean_band"] = f"{OMEGA_LATMEAN_MIN}..{OMEGA_LATMEAN_MAX}"
    out["w_bp"].attrs["note"] = "Meridional mean applied before filtering (linear, commutes with filtering); output is (time, level, lon)."

    out = _sanitize_coords_for_netcdf(out)
    out_path = OUT_ERA5_PL_DIR / f"era5_w_bp_latmean_{START_DATE[:4]}-{END_DATE[:4]}.nc"
    out.to_netcdf(out_path, engine="netcdf4", encoding=_encoding_for_dataset(out))
    return out_path


def main():
    p1 = step2_filter_olr()
    print("Saved:", p1)
    p2 = step2_filter_era5_u()
    print("Saved:", p2)
    # ADD: omega (w) bandpass
    p3 = step2_filter_era5_w()
    print("Saved:", p3)


if __name__ == "__main__":
    main()
