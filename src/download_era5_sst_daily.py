# -*- coding: utf-8 -*-
"""
download_era5_sst_daily.py
下载 ERA5 逐日 SST (按年请求，44次 vs 逐月528次)
  - 分辨率: 2.5° × 2.5°
  - 范围: 20°S–20°N, 全经度
  - 时间: 1979-2022, 每年一个请求 (取 00Z 瞬时值作日均值)
  - 输出: 直接写入 daily_mean/ 与其他变量同目录

SST 日变化极小（<0.1K），取 00Z 即可代表日均值，无需4时次平均。
"""

import os
import calendar
import numpy as np
import xarray as xr
import zipfile
import tempfile
import shutil
from pathlib import Path

OUTPUT_DIR = Path(r"E:\Datas\ERA5\raw\single_level\sst_daily")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 1979
END_YEAR = 2022

AREA = [20, -180, -20, 180]
GRID = [2.5, 2.5]


def download_year(year, output_path):
    """下载一整年的 SST 日数据（00Z 瞬时值）。"""
    import cdsapi

    months = [f"{m:02d}" for m in range(1, 13)]
    days = [f"{d:02d}" for d in range(1, 32)]

    c = cdsapi.Client()
    request = {
        'product_type': 'reanalysis',
        'variable': 'sea_surface_temperature',
        'year': str(year),
        'month': months,
        'day': days,
        'time': '00:00',  # SST 日变化极小，单时次即可
        'area': AREA,
        'grid': GRID,
        'format': 'netcdf',
    }

    print(f"  Requesting CDS: {year} ...")
    c.retrieve('reanalysis-era5-single-levels', request, str(output_path))
    sz = output_path.stat().st_size / 1e6
    print(f"  Downloaded: {output_path.name} ({sz:.1f} MB)")


def unzip_if_needed(filepath):
    """如果文件是 ZIP 格式，解压并返回内部数据文件路径。"""
    with open(filepath, 'rb') as f:
        magic = f.read(2)
    if magic == b'PK':
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(tmp_dir)
        data_files = [f for f in Path(tmp_dir).rglob("*") if f.is_file()]
        return data_files[0], tmp_dir
    return filepath, None


def split_to_monthly(year_file, year):
    """将年文件拆分为逐月文件并保存到 daily_mean/。"""
    actual_path, tmp_dir = unzip_if_needed(year_file)
    try:
        ds = xr.open_dataset(actual_path)
        time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'

        for month in range(1, 13):
            out_path = OUTPUT_DIR / f"era5_sst_dailymean_{year}{month:02d}.nc"
            if out_path.exists() and out_path.stat().st_size > 500:
                continue

            # 筛选该月数据
            times = ds[time_dim].values
            import pandas as pd
            ts = pd.to_datetime(times)
            mask = (ts.month == month) & (ts.year == year)
            if mask.sum() == 0:
                continue

            ds_month = ds.isel({time_dim: mask})
            # SST 单位已经是 K
            for v in ds_month.data_vars:
                ds_month[v].attrs['units'] = 'K'

            encoding = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                        for v in ds_month.data_vars}
            ds_month.to_netcdf(out_path, encoding=encoding)
            print(f"    {out_path.name} ({out_path.stat().st_size/1e6:.2f} MB)")

        ds.close()
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    TMP_DIR = Path(r"E:\Datas\ERA5\raw\single_level\_tmp_sst")
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    total = END_YEAR - START_YEAR + 1
    print("=" * 60)
    print("ERA5 Daily SST Download (by year)")
    print(f"Period: {START_YEAR}-{END_YEAR} ({total} years)")
    print(f"Grid: 2.5° × 2.5°, 20°S–20°N, global longitude")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    for i, year in enumerate(range(START_YEAR, END_YEAR + 1), 1):
        # 检查是否所有月份都已完成
        all_done = all(
            (OUTPUT_DIR / f"era5_sst_dailymean_{year}{m:02d}.nc").exists()
            for m in range(1, 13)
        )
        if all_done:
            print(f"[{i}/{total}] {year} - all months exist, skip")
            continue

        print(f"\n[{i}/{total}] {year}")
        tmp_file = TMP_DIR / f"era5_sst_raw_{year}.nc"

        try:
            if not tmp_file.exists():
                download_year(year, tmp_file)
            split_to_monthly(tmp_file, year)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nDone!")


if __name__ == "__main__":
    main()
