# -*- coding: utf-8 -*-
"""
02c_reconstruct_surface_2d.py
对单层变量(SST, LHF, SHF, TP, OLR)做 MJO PC 回归重构（保留 lat 维度）

公式与气压层相同:
  recon = (β₁·PC1/vf₁ + β₂·PC2/vf₂) / 2
  norm  = recon / max(amp, 1.0)

输出: era5_mjo_recon_{var}_norm_2d_1979-2022.nc  dims: (time, lat, lon)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ======================
# PATHS
# ======================
DERIVED_DIR = Path(r"E:\Datas\Derived")
SL_DIR = Path(r"E:\Datas\ERA5\raw\single_level\daily_mean")
SST_DIR = Path(r"E:\Datas\ERA5\raw\single_level\sst_daily")
OLR_PATH = Path(r"E:\Datas\ClimateIndex\raw\olr\olr.day.mean.nc")
STEP3_NC = DERIVED_DIR / "mjo_mvEOF_step3_1979-2022.nc"

START_DATE = "1979-01-01"
END_DATE = "2022-12-31"
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}
LAT_BAND = (-20.0, 20.0)  # 使用ERA5原始数据完整纬度范围
AMP_FLOOR = 1.0

# Variables to reconstruct
SURFACE_VARS = {
    "lhf": {"sl_var": "slhf", "sign": -1, "source": "sl"},
    "shf": {"sl_var": "sshf", "sign": -1, "source": "sl"},
    "tp":  {"sl_var": "tp",   "sign":  1, "source": "sl"},
    "sst": {"sl_var": "sst",  "sign":  1, "source": "sst"},
    "olr": {"sl_var": "olr",  "sign":  1, "source": "olr"},
}


# ======================
# HELPERS
# ======================
def _rename_latlon(ds):
    ren = {}
    if "latitude" in ds.coords: ren["latitude"] = "lat"
    if "longitude" in ds.coords: ren["longitude"] = "lon"
    if ren: ds = ds.rename(ren)
    return ds


def _to_lon_180(ds):
    if "lon" not in ds.coords: return ds
    lon = ds["lon"].values
    lon180 = ((lon + 180) % 360) - 180
    return ds.assign_coords(lon=lon180).sortby("lon")


def _sel_lat_band(ds, lat_band):
    lat = ds["lat"].values
    if lat.size >= 2 and (lat[1] - lat[0]) < 0:
        return ds.sel(lat=slice(lat_band[1], lat_band[0]))
    return ds.sel(lat=slice(lat_band[0], lat_band[1]))


# ======================
# DATA LOADING (keeping lat)
# ======================
def load_surface_2d(var_key, var_info, time_index):
    """Load surface variable WITHOUT lat averaging → (time, lat, lon)."""
    src = var_info["source"]

    if src == "olr":
        return _load_olr_2d(time_index)

    data_dir = SST_DIR if src == "sst" else SL_DIR
    pattern = "era5_sst_dailymean_*.nc" if src == "sst" else "era5_sl_dailymean_*.nc"

    ym_set = set()
    for t in time_index:
        ym_set.add((t.year, t.month))
    ym_list = sorted(ym_set)

    arrays = []
    for year, month in ym_list:
        if src == "sst":
            fname = data_dir / f"era5_sst_dailymean_{year}{month:02d}.nc"
        else:
            fname = data_dir / f"era5_sl_dailymean_{year}{month:02d}.nc"
        if not fname.exists():
            continue
        ds = xr.open_dataset(fname)
        ds = _rename_latlon(ds)
        ds = _to_lon_180(ds)

        tdim = 'valid_time' if 'valid_time' in ds.dims else 'time'
        if tdim != 'time':
            ds = ds.rename({tdim: 'time'})

        sl_var = var_info["sl_var"]
        da = ds[sl_var] * var_info["sign"]

        # Select lat band
        da = _sel_lat_band(da.to_dataset(name=sl_var), LAT_BAND)[sl_var]
        arrays.append(da)
        ds.close()

    combined = xr.concat(arrays, dim="time")
    combined = combined.sel(time=time_index, method="nearest")
    combined = combined.assign_coords(time=time_index)
    print(f"    Shape: {combined.shape}")
    return combined


def _load_olr_2d(time_index):
    """Load OLR, subset lat and interp to ERA5-like grid."""
    ds = xr.open_dataset(str(OLR_PATH))
    ds = _rename_latlon(ds)
    ds = _to_lon_180(ds)
    ds = ds.sel(time=slice(START_DATE, END_DATE))
    da = _sel_lat_band(ds, LAT_BAND)["olr"]

    # Align to time_index
    da = da.sel(time=time_index, method="nearest")
    da = da.assign_coords(time=time_index)
    print(f"    OLR shape: {da.shape}")
    return da


# ======================
# RECONSTRUCTION
# ======================
def reconstruct_field_2d(field, pc1, pc2, winter_mask, var_frac1, var_frac2):
    """
    PC regression reconstruction for 2D field (time, lat, lon).
    """
    time_coord = field["time"]
    lat_coord = field["lat"]
    lon_coord = field["lon"]

    T, Y, X = field.shape
    out = np.full((T, Y, X), np.nan, dtype=np.float32)

    pc_valid = np.isfinite(pc1) & np.isfinite(pc2)
    train_mask = winter_mask & pc_valid

    for lat_idx in range(Y):
        for lon_idx in range(X):
            y = field.values[:, lat_idx, lon_idx].astype(float)
            m = train_mask & np.isfinite(y)

            if m.sum() < 10:
                continue

            X_train = np.column_stack([pc1[m], pc2[m]])
            y_train = y[m]

            try:
                beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
                out[pc_valid, lat_idx, lon_idx] = (
                    beta[0] * pc1[pc_valid] / var_frac1
                    + beta[1] * pc2[pc_valid] / var_frac2
                ) / 2.0
            except:
                pass

        if (lat_idx + 1) % 5 == 0 or lat_idx == Y - 1:
            print(f"      Lat {lat_idx+1}/{Y} done")

    return xr.DataArray(
        out,
        coords={"time": time_coord, "lat": lat_coord, "lon": lon_coord},
        dims=("time", "lat", "lon"),
    )


# ======================
# MAIN
# ======================
def main():
    print("=" * 70)
    print("02c: Surface Variable 2D Reconstruction (keeping lat)")
    print("=" * 70)
    start_time = datetime.now()

    # Load PC data
    if not STEP3_NC.exists():
        raise FileNotFoundError(f"Step3 output not found: {STEP3_NC}")

    ds_pc = xr.open_dataset(STEP3_NC)
    pc1_np = ds_pc["pc1"].values.astype(float)
    pc2_np = ds_pc["pc2"].values.astype(float)
    amp_np = ds_pc["amp"].values.astype(float)
    var_frac1 = float(ds_pc.attrs.get("var_frac1", 1.0))
    var_frac2 = float(ds_pc.attrs.get("var_frac2", 1.0))
    time_index = pd.to_datetime(ds_pc["time"].values)
    ds_pc.close()

    winter = np.array([t.month in WINTER_MONTHS for t in time_index])
    amp_safe = np.maximum(amp_np, AMP_FLOOR)

    print(f"  PC time: {time_index[0]} to {time_index[-1]}, n={len(time_index)}")
    print(f"  var_frac1={var_frac1:.4f}, var_frac2={var_frac2:.4f}")
    print(f"  Winter days: {winter.sum()}")

    for var_key, var_info in SURFACE_VARS.items():
        output_path = DERIVED_DIR / f"era5_mjo_recon_{var_key}_norm_2d_{START_DATE[:4]}-{END_DATE[:4]}.nc"

        if output_path.exists():
            print(f"\n[SKIP] {var_key}: already exists: {output_path.name}")
            continue

        print(f"\n{'='*50}")
        print(f"Processing: {var_key}")
        print("=" * 50)

        # Load data (keeping lat)
        print(f"  Loading {var_key} ...")
        field = load_surface_2d(var_key, var_info, time_index)

        # Reconstruct
        print(f"  Reconstructing ...")
        recon = reconstruct_field_2d(field, pc1_np, pc2_np, winter, var_frac1, var_frac2)

        # Normalize
        recon_norm_np = recon.values.astype(float) / amp_safe[:, None, None]

        recon_norm = xr.DataArray(
            recon_norm_np.astype(np.float32),
            coords=recon.coords,
            dims=recon.dims,
            name=f"{var_key}_mjo_recon_norm_2d"
        )

        ds_out = xr.Dataset({f"{var_key}_mjo_recon_norm_2d": recon_norm})
        ds_out.attrs["description"] = f"MJO-reconstructed {var_key} (2D with lat), normalized by amplitude"
        ds_out.attrs["method"] = "field_norm = (b1*pc1/vf1 + b2*pc2/vf2) / 2 / max(amp, 1.0)"
        ds_out.attrs["created"] = datetime.now().isoformat()

        ds_out.to_netcdf(output_path)
        print(f"  Saved: {output_path}")

    elapsed = datetime.now() - start_time
    print(f"\nAll done in {elapsed}")


if __name__ == "__main__":
    main()
