# -*- coding: utf-8 -*-
"""
convert_sl_to_daily.py
将 _tmp_hourly/ 中的原始 4时次单层数据批量转换为日均值。
"""

import numpy as np
import pandas as pd
import xarray as xr
import zipfile
import tempfile
import os
import shutil
from pathlib import Path

RAW_DIR = Path(r"E:\Datas\ERA5\raw\single_level\_tmp_hourly")
OUT_DIR = Path(r"E:\Datas\ERA5\raw\single_level\daily_mean")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _open_maybe_zip(raw_path):
    """打开文件，自动处理 ZIP 压缩（新版 CDS API 返回 ZIP）。"""
    with open(raw_path, 'rb') as f:
        magic = f.read(4)

    if magic[:2] == b'PK':
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(raw_path, 'r') as zf:
            zf.extractall(tmp_dir)
        extracted = list(Path(tmp_dir).rglob("*"))
        data_files = [f for f in extracted if f.is_file()]
        if not data_files:
            raise ValueError(f"No data file found in ZIP: {extracted}")
        # 尝试各种引擎打开
        for df in data_files:
            for engine in ['netcdf4', 'scipy', 'cfgrib']:
                try:
                    ds = xr.open_dataset(df, engine=engine)
                    return ds, tmp_dir
                except Exception:
                    continue
        # 打印实际文件信息帮助调试
        info = [(f.name, f.suffix, f.stat().st_size) for f in data_files]
        raise ValueError(f"Cannot open files: {info}")
    else:
        ds = xr.open_dataset(raw_path)
        return ds, None


def process_one(raw_path, out_path):
    """将单个原始文件转为日均值。"""
    tmp_dir = None
    try:
        ds, tmp_dir = _open_maybe_zip(raw_path)

        # 识别时间维度
        time_dim = 'time' if 'time' in ds.dims else 'valid_time'

        # 按天求均值
        ds_daily = ds.resample({time_dim: '1D'}).mean()

        # 累积变量单位转换
        for var in ds_daily.data_vars:
            short = str(var).lower()
            if short in ('slhf', 'sshf', 'ssr', 'str'):
                ds_daily[var] = ds_daily[var] / 21600.0
                ds_daily[var].attrs['units'] = 'W/m²'
            elif short == 'tp':
                ds_daily[var] = ds_daily[var] * 4.0 * 1000.0
                ds_daily[var].attrs['units'] = 'mm/day'
            elif short == 'sst':
                ds_daily[var].attrs['units'] = 'K'

        # 压缩保存
        encoding = {v: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
                    for v in ds_daily.data_vars}
        ds_daily.to_netcdf(out_path, encoding=encoding)
        ds.close()
        return out_path.stat().st_size
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    raw_files = sorted(RAW_DIR.glob("era5_sl_raw_*.nc"))
    print(f"Found {len(raw_files)} raw files in {RAW_DIR}")
    print(f"Output: {OUT_DIR}\n")

    done, skip, fail = 0, 0, 0
    for i, rf in enumerate(raw_files, 1):
        # era5_sl_raw_197901.nc → era5_sl_dailymean_197901.nc
        ym = rf.stem.replace("era5_sl_raw_", "")
        out_path = OUT_DIR / f"era5_sl_dailymean_{ym}.nc"

        if out_path.exists() and out_path.stat().st_size > 1000:
            skip += 1
            continue

        try:
            sz = process_one(rf, out_path)
            done += 1
            print(f"  [{i}/{len(raw_files)}] {out_path.name}  ({sz/1e6:.1f} MB)")
        except Exception as e:
            fail += 1
            print(f"  [{i}/{len(raw_files)}] FAILED {rf.name}: {e}")

    print(f"\nDone={done}, Skipped={skip}, Failed={fail}")


if __name__ == "__main__":
    main()
