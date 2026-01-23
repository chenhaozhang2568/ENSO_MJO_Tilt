# -*- coding: utf-8 -*-
"""
ERA5 数据下载脚本（新需求）
- 时间范围：1979-2022年
- 气压层：1000, 925, 850, 700, 600, 500, 400, 300, 200 hPa（9层）
- 变量：q, u, v, w(omega), T（5个变量）
- 分辨率：1.5° × 1.5°
- 空间范围：40°S - 40°N，全经度（-180° 至 180°）
- 输出目录：E:\Datas\ERA5\raw\pressure_level\era5_1979-2022_quvwT_9_20 -180 20 180
- 每日 00Z 瞬时场，按月下载
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import calendar

# ======================
# 路径配置
# ======================
CDSAPI_RC_PATH = r"C:\Users\Lenovo\.cdsapirc"
OUTPUT_DIR = Path(r"E:\Datas\ERA5\raw\pressure_level\era5_1979-2022_quvwT_9_20 -180 20 180")
README_PATH = Path(r"E:\Datas\readme.txt")

# ======================
# 下载配置
# ======================
CONFIG = {
    "years": list(range(1979, 2023)),  # 1979-2022
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
    "area": [40, -180, -40, 180],       # [North, West, South, East]
    "grid": [1.5, 1.5],                 # 分辨率 1.5° × 1.5°
    "time": "00:00",                    # 00Z 瞬时场
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


def download_era5_pressure_levels():
    """
    按月下载 ERA5 pressure-levels 数据
    文件命名：era5_pl_YYYYMM_00Z.nc
    """
    name = "ERA5 pressure-levels (1979-2022, 9 levels, quvwT, 1.5deg)"

    # 1) cdsapi 配置
    os.environ["CDSAPI_RC"] = CDSAPI_RC_PATH
    if not Path(CDSAPI_RC_PATH).exists():
        logger.error(f"找不到 CDS API 配置文件: {CDSAPI_RC_PATH}")
        _append_readme(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {name}\n"
            f"  status: FAIL: CDSAPI_RC not found: {CDSAPI_RC_PATH}\n"
        )
        return False

    try:
        import cdsapi
    except ImportError as e:
        logger.error("未安装 cdsapi，请先运行: pip install cdsapi")
        _append_readme(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {name}\n"
            f"  status: FAIL: cdsapi import error: {e}\n"
        )
        return False

    # 2) 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3) 打印下载信息
    logger.info(f"开始下载 {name}")
    logger.info(f"CDSAPI_RC  = {CDSAPI_RC_PATH}")
    logger.info(f"保存目录   = {OUTPUT_DIR}")
    logger.info(f"时间范围   = {CONFIG['years'][0]}-{CONFIG['years'][-1]}")
    logger.info(f"区域 area  = {CONFIG['area']} (N, W, S, E)")
    logger.info(f"分辨率     = {CONFIG['grid']} deg")
    logger.info(f"气压层     = {CONFIG['pressure_levels']}")
    logger.info(f"变量       = {CONFIG['variables']}")
    logger.info("注意：该下载为每日 00Z 瞬时场（不是日平均）。")

    c = cdsapi.Client(timeout=600)

    all_ok = True
    total_files = len(CONFIG["years"]) * len(CONFIG["months"])
    current_file = 0

    for y in CONFIG["years"]:
        for m in CONFIG["months"]:
            current_file += 1
            out = OUTPUT_DIR / f"era5_pl_{y}{m:02d}_00Z.nc"

            # 跳过已存在的文件
            if out.exists():
                logger.info(f"[{current_file}/{total_files}] 已存在，跳过: {out.name}")
                continue

            # 计算当月天数
            ndays = calendar.monthrange(y, m)[1]
            days = [f"{d:02d}" for d in range(1, ndays + 1)]

            logger.info(f"[{current_file}/{total_files}] 下载 ERA5 {y}-{m:02d} -> {out.name}")

            try:
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": CONFIG["variables"],
                        "pressure_level": CONFIG["pressure_levels"],
                        "year": str(y),
                        "month": f"{m:02d}",
                        "day": days,
                        "time": CONFIG["time"],
                        "area": CONFIG["area"],
                        "grid": CONFIG["grid"],
                        "format": "netcdf",
                    },
                    str(out),
                )
                msg = "OK"
                logger.info(f"  完成: {_fmt_size(out)}")
            except Exception as e:
                all_ok = False
                msg = f"FAIL: {e}"
                logger.error(f"ERA5 {y}-{m:02d} 下载失败: {e}")

            # 记录到 readme
            rec = (
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {name}\n"
                f"  year-month: {y}-{m:02d}\n"
                f"  output:     {out}\n"
                f"  size:       {_fmt_size(out)}\n"
                f"  params:     time={CONFIG['time']}, area={CONFIG['area']}, "
                f"grid={CONFIG['grid']}, levels={CONFIG['pressure_levels']}, "
                f"vars={CONFIG['variables']}\n"
                f"  status:     {msg}\n"
            )
            _append_readme(rec)

    if all_ok:
        logger.info("===== 所有文件下载完成 =====")
    else:
        logger.warning("===== 部分文件下载失败，请检查日志 =====")

    return all_ok


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="下载 ERA5 pressure-level 数据 (1979-2022, 9层, quvwT, 1.5deg)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示配置信息，不实际下载"
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("ERA5 下载配置（dry-run 模式，不实际下载）")
        print("=" * 60)
        print(f"时间范围  : {CONFIG['years'][0]} - {CONFIG['years'][-1]}")
        print(f"月份      : 1-12")
        print(f"变量      : {CONFIG['variables']}")
        print(f"气压层    : {CONFIG['pressure_levels']}")
        print(f"区域      : N={CONFIG['area'][0]}, W={CONFIG['area'][1]}, "
              f"S={CONFIG['area'][2]}, E={CONFIG['area'][3]}")
        print(f"分辨率    : {CONFIG['grid'][0]}° × {CONFIG['grid'][1]}°")
        print(f"时间      : {CONFIG['time']} (瞬时场)")
        print(f"输出目录  : {OUTPUT_DIR}")
        print(f"总文件数  : {len(CONFIG['years']) * len(CONFIG['months'])} 个月")
        print("=" * 60)
        return

    download_era5_pressure_levels()


if __name__ == "__main__":
    main()
