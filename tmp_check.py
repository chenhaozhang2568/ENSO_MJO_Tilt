import xarray as xr
import pandas as pd
import numpy as np

ds = xr.open_dataset(r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc")
sub = ds.sel(time=slice("2004-01-10", "2004-01-20"))

print("Date         center_lon  olr_center   contour   min_olr_recon")
print("-" * 70)
for i in range(len(sub.time)):
    t = pd.Timestamp(sub.time.values[i])
    clon = float(sub["center_lon_track"].values[i])
    olr_c = float(sub["olr_center_track"].values[i])
    # contour track (olr_thr_centroid_lon) if it exists
    cthr = float(sub["olr_thr_centroid_lon"].values[i]) if "olr_thr_centroid_lon" in sub else np.nan
    # min OLR across all longitudes
    olr_slice = sub["olr_recon"].isel(time=i)
    min_olr = float(olr_slice.min().values)
    print(f"{t.strftime('%Y-%m-%d')}   {clon:7.1f}     {olr_c:7.1f}     {cthr:7.1f}   {min_olr:7.1f}")
