# -*- coding: utf-8 -*-
"""
ERA5 数据下载与日平均处理脚本
- 机制：下载每日 4 个时次 (00, 06, 12, 18) -> 计算日平均 -> 保存 -> 删除原始临时文件
- 时间范围：1979-2022年
- 气压层：1000, 925, 850, 700, 600, 500, 400, 300, 200 hPa（9层）
- 变量：q, u, v, w(omega), T
- 分辨率：1.5° × 1.5°
- 空间范围：[90, -180, -90, 180] (注意：这里保留了你代码中的全图配置，MJO研究通常40S-40N即可)
- 输出：日平均数据 (Daily Mean)
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import calendar
import xarray as xr  # 新增：用于计算日平均
import cdsapi

# ======================
# 路径配置
# ======================
CDSAPI_RC_PATH = r"C:\Users\Lenovo\.cdsapirc"
# 修改输出目录名以体现是 daily mean
OUTPUT_DIR = Path(r"E:\Datas\ERA5\raw\pressure_level\era5_dailymean_1979-2022_quvwT")
README_PATH = Path(r"E:\Datas\readme.txt")

# ======================
# 下载配置
# ======================
CONFIG = {
    "years": list(range(1979, 2023)),   # 1979-2022
    "months": list(range(1, 13)),       # 1-12月
    "variables": [
        "specific_humidity",            # q
        "u_component_of_wind",          # u
        "v_component_of_wind",          # v
        "vertical_velocity",            # w (omega)
        "temperature",                  # T
    ],
    "pressure_levels": [
        "1000", "925", "850", "700", "600", "500", "400", "300", "200"
    ],
    # 注意：你代码中写的是全球(90...-90)，但注释说是40S-40N。
    # 为了安全，这里保留你代码中的全球设置。如果想省空间，可改为 [40, -180, -40, 180]
    "area": [90, -180, -90, 180],       
    "grid": [1.5, 1.5],                 
    
    # === 核心修改：下载4个时次以计算日平均 ===
    "time": [
        "00:00", "06:00", "12:00", "18:00"
    ],
}

# ======================
# 日志配置
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ERA5Downloader")


def _append_readme(record: str) -> None:
    """把下载记录追加到 readme.txt"""
    try:
        README_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not README_PATH.exists():
            README_PATH.write_text("Data download log\n\n", encoding="utf-8")

        with open(README_PATH, "a", encoding="utf-8") as f:
            f.write(record.rstrip() + "\n")
    except Exception as e:
        logger.warning(f"写入 readme 失败（不影响下载本身）: {e}")


def _fmt_size(path: Path) -> str:
    """格式化文件大小"""
    if not path.exists():
        return "NA"
    n = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def process_to_dailymean(temp_path: Path, final_path: Path):
    """
    读取包含4个时次的临时文件，计算日平均，保存为最终文件，并删除临时文件。
    """
    try:
        logger.info(f"  正在计算日平均: {temp_path.name} -> {final_path.name}")
        
        # 打开临时文件
        with xr.open_dataset(temp_path) as ds:
            # 计算日平均 (Resample to 1 Day, calculating mean)
            # 使用 .resample('1D').mean() 或者 .groupby('time.date').mean()
            # 这里由于已经是按月下载，且每天都有4个时次，直接用 resample 最稳妥
            ds_daily = ds.resample(time="1D").mean(keep_attrs=True)
            
            # 继承属性（可选，有时 mean 会丢失部分属性）
            ds_daily.attrs = ds.attrs
            ds_daily.attrs['history'] = f"Computed daily mean from 6-hourly ERA5 data on {datetime.now()}"
            
            # 保存压缩过的 NetCDF
            encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_daily.data_vars}
            ds_daily.to_netcdf(final_path, encoding=encoding)
            
        # 验证成功后，删除临时大文件
        if final_path.exists() and final_path.stat().st_size > 0:
            os.remove(temp_path)
            logger.info(f"  日平均计算完成，已删除临时文件。最终大小: {_fmt_size(final_path)}")
            return True, "OK"
        else:
            return False, "Save Failed"
            
    except Exception as e:
        logger.error(f"  日平均处理失败: {e}")
        return False, f"Process Fail: {e}"


def download_era5_pressure_levels():
    """按月下载并处理 ERA5 数据"""
    name = "ERA5 pressure-levels (Daily Mean, 1979-2022)"

    # 1) cdsapi 配置
    os.environ["CDSAPI_RC"] = CDSAPI_RC_PATH
    if not Path(CDSAPI_RC_PATH).exists():
        logger.error(f"找不到 CDS API 配置文件: {CDSAPI_RC_PATH}")
        return False

    try:
        c = cdsapi.Client(timeout=600)
    except Exception as e:
        logger.error(f"CDSAPI 初始化失败: {e}")
        return False

    # 2) 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3) 打印信息
    logger.info(f"开始下载任务: {name}")
    logger.info(f"下载策略: 下载 00/06/12/18 时次 -> 本地计算日平均 -> 保存")
    logger.info(f"变量: {CONFIG['variables']}")
    
    all_ok = True
    total_files = len(CONFIG["years"]) * len(CONFIG["months"])
    current_file = 0

    for y in CONFIG["years"]:
        for m in CONFIG["months"]:
            current_file += 1
            
            # 最终文件名 (Daily Mean)
            final_out = OUTPUT_DIR / f"era5_pl_{y}{m:02d}_dailymean.nc"
            # 临时文件名 (6-hourly Raw)
            temp_out = OUTPUT_DIR / f"era5_pl_{y}{m:02d}_temp_6hourly.nc"

            if final_out.exists():
                logger.info(f"[{current_file}/{total_files}] 已存在，跳过: {final_out.name}")
                continue

            # 计算当月天数
            ndays = calendar.monthrange(y, m)[1]
            days = [f"{d:02d}" for d in range(1, ndays + 1)]

            logger.info(f"[{current_file}/{total_files}] 下载中: {y}-{m:02d} (6-hourly)")

            msg = "Init"
            try:
                # === 步骤 A: 下载 6小时数据 ===
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": CONFIG["variables"],
                        "pressure_level": CONFIG["pressure_levels"],
                        "year": str(y),
                        "month": f"{m:02d}",
                        "day": days,
                        "time": CONFIG["time"],  # 这里包含了4个时次
                        "area": CONFIG["area"],
                        "grid": CONFIG["grid"],
                        "format": "netcdf",
                    },
                    str(temp_out),
                )
                
                # === 步骤 B: 转为日平均 ===
                success, proc_msg = process_to_dailymean(temp_out, final_out)
                if success:
                    msg = "OK (Downloaded & Processed)"
                else:
                    all_ok = False
                    msg = proc_msg

            except Exception as e:
                all_ok = False
                msg = f"Download FAIL: {e}"
                logger.error(f"ERA5 {y}-{m:02d} 下载/处理失败: {e}")
                # 如果下载了一半失败了，尝试清理临时文件
                if temp_out.exists():
                    try:
                        os.remove(temp_out)
                    except:
                        pass

            # 记录日志
            _append_readme(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {y}-{m:02d} | {msg} | Size: {_fmt_size(final_out)}"
            )

    return all_ok


def main():
    # 简单的启动逻辑，去掉了 dry-run 的复杂打印以便直接运行
    print("=" * 60)
    print("开始执行 ERA5 日平均数据下载与处理任务")
    print(f"目标目录: {OUTPUT_DIR}")
    print("机制: 下载 6-hourly 数据 -> 计算 Mean -> 删除原始数据")
    print("=" * 60)
    
    download_era5_pressure_levels()

if __name__ == "__main__":
    main()