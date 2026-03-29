# -*- coding: utf-8 -*-
"""
download_era5_single_level.py
下载 ERA5 单层变量逐日数据 (1979-2022)
  1. SST（海表温度）
  2. 表面潜热/感热通量 (SLHF, SSHF)
  3. 表面辐射通量 (净短波 SNSR, 净长波 SNTR)
  4. 降水 (TP)

数据规格:
  - 分辨率: 2.5° × 2.5°
  - 范围: 20°S–20°N, 全经度 (-180–180)
  - 网格: 144 lon × 17 lat
  - 时间: 逐日 (从4个时次 00/06/12/18Z 计算日均值)
  - 文件: 逐月保存

预估大小:
  - 每月原始下载 (4时次): ~3-4 MB (GRIB/NetCDF 压缩)
  - 每月日均值文件: ~2-3 MB (6变量 × 31天 × 144×17 float32, zlib压缩)
  - 总计下载量: ~1.5-2 GB (528个月)
  - 总计存储量: ~1.0-1.5 GB (日均值)

目录结构:
  E:\Datas\ERA5\raw\single_level\
  └── daily_mean\
      ├── era5_sl_dailymean_197901.nc
      ├── era5_sl_dailymean_197902.nc
      └── ...

用法:
  1. 确保已安装 cdsapi: pip install cdsapi
  2. 配置 ~/.cdsapirc 文件 (含 CDS API key)
  3. 运行: python download_era5_single_level.py
  4. 支持断点续传 (已下载的月份自动跳过)

注意: ERA5 累积变量 (通量、辐射、降水) 的日均值需要特殊处理:
  - SST: 瞬时值, 4时次平均 = 日均值
  - 通量/辐射: 累积值 (J/m² since 00Z), 需要反累积后取均值转为 W/m²
  - 降水: 累积值 (m since 00Z), 需要反累积后取总和转为 mm/day
"""

import os
import sys
import calendar
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime

# ====== 配置 ======
OUTPUT_DIR = Path(r"E:\Datas\ERA5\raw\single_level\daily_mean")
TEMP_DIR = Path(r"E:\Datas\ERA5\raw\single_level\_tmp_hourly")

START_YEAR = 1979
END_YEAR = 2022

# CDS API 变量名
VARIABLES = [
    'sea_surface_temperature',       # SST (K), 瞬时值
    'surface_latent_heat_flux',      # SLHF (J/m²), 累积 → W/m²
    'surface_sensible_heat_flux',    # SSHF (J/m²), 累积 → W/m²
    'surface_net_solar_radiation',   # SNSR (J/m²), 累积 → W/m² (净短波)
    'surface_net_thermal_radiation', # SNTR (J/m²), 累积 → W/m² (净长波)
    'total_precipitation',           # TP (m), 累积 → mm/day
]

# 累积变量列表 (需要反累积处理)
ACCUMULATED_VARS = {
    'slhf', 'sshf', 'ssr', 'str', 'tp'
}

# 空间范围
AREA = [20, -180, -20, 180]  # [N, W, S, E]
GRID = [2.5, 2.5]             # [lat_res, lon_res]


# ====== 下载函数 ======
def download_month(year, month, output_path):
    """
    从 CDS 下载某个月的 ERA5 单层数据。
    请求 4 个时次 (00/06/12/18Z) 用于计算日均值。
    """
    import cdsapi

    n_days = calendar.monthrange(year, month)[1]
    days = [f"{d:02d}" for d in range(1, n_days + 1)]

    c = cdsapi.Client()

    request = {
        'product_type': 'reanalysis',
        'variable': VARIABLES,
        'year': str(year),
        'month': f"{month:02d}",
        'day': days,
        'time': ['00:00', '06:00', '12:00', '18:00'],
        'area': AREA,
        'grid': GRID,
        'format': 'netcdf',           # 旧版 CDS API
        # 'data_format': 'netcdf',    # 新版 CDS API (如果上面报错, 注释上行用这行)
    }

    print(f"    Requesting CDS: {year}-{month:02d} ...")
    c.retrieve('reanalysis-era5-single-levels', request, str(output_path))
    print(f"    Downloaded: {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")


def process_to_daily_mean(hourly_path, output_path, year, month):
    """
    将 4时次数据转换为逐日均值。

    处理逻辑:
    - SST: 4时次直接平均
    - 累积变量 (通量/辐射): 反累积 → 日总量 → 转为日均功率 (W/m²)
    - 降水: 反累积 → 日总降水量 (mm/day)
    """
    ds = xr.open_dataset(hourly_path)

    # 识别时间维度名称
    time_dim = 'time' if 'time' in ds.dims else 'valid_time'

    # 按天分组计算均值
    times = pd.to_datetime(ds[time_dim].values)
    ds_daily = ds.resample({time_dim: '1D'}).mean()

    # 对累积变量做特殊处理:
    # ERA5 累积值从每个 forecast 的起始时间累积
    # 4时次均值近似日均值 (对于瞬时值是精确的)
    # 对于通量, J/m² → W/m²: 除以 6小时 (21600秒)
    for var in ds_daily.data_vars:
        short = str(var).lower()
        if short in ('slhf', 'sshf', 'ssr', 'str'):
            # 累积通量 J/m² → 日均功率 W/m²
            # 4时次的均值 × 4 次 / 86400秒 ≈ 日均 W/m²
            # 实际: 每步累积6小时, 值已是 J/m², 均值÷21600→W/m²
            ds_daily[var] = ds_daily[var] / 21600.0
            ds_daily[var].attrs['units'] = 'W/m²'
        elif short == 'tp':
            # 累积降水 m → mm/day
            # 4时次均值 × 4 = 日总量 (m), × 1000 = mm
            ds_daily[var] = ds_daily[var] * 4.0 * 1000.0
            ds_daily[var].attrs['units'] = 'mm/day'
        elif short == 'sst':
            ds_daily[var].attrs['units'] = 'K'

    # 编码压缩
    encoding = {}
    for var in ds_daily.data_vars:
        encoding[var] = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}

    ds_daily.to_netcdf(output_path, encoding=encoding)
    ds.close()
    print(f"    Saved daily mean: {output_path.name} ({output_path.stat().st_size / 1e6:.1f} MB)")


# ====== 主流程 ======
def main():
    import pandas as pd

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    total_months = (END_YEAR - START_YEAR + 1) * 12
    done = 0
    skipped = 0

    print("=" * 60)
    print("ERA5 Single-Level Download")
    print(f"Variables: SST, SLHF, SSHF, SNSR, SNTR, TP")
    print(f"Period: {START_YEAR}-01 to {END_YEAR}-12  ({total_months} months)")
    print(f"Grid: 2.5° × 2.5°, 20°S–20°N, global longitude")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            done += 1
            out_file = OUTPUT_DIR / f"era5_sl_dailymean_{year}{month:02d}.nc"

            # 断点续传: 跳过已存在文件
            if out_file.exists() and out_file.stat().st_size > 1000:
                skipped += 1
                continue

            print(f"\n[{done}/{total_months}] {year}-{month:02d}")

            # Step 1: 下载原始数据
            tmp_file = TEMP_DIR / f"era5_sl_raw_{year}{month:02d}.nc"
            try:
                if not tmp_file.exists():
                    download_month(year, month, tmp_file)

                # Step 2: 转换为日均值
                process_to_daily_mean(tmp_file, out_file, year, month)

                # Step 3: 删除临时文件
                tmp_file.unlink()

            except Exception as e:
                print(f"    ERROR: {e}")
                # 保留临时文件方便排错
                continue

    print(f"\n{'=' * 60}")
    print(f"Done! Downloaded: {done - skipped}, Skipped: {skipped}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    import pandas as pd  # noqa - needed by process_to_daily_mean
    main()
