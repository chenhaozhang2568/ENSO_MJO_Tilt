"""
数据下载模块（ENSO-MJO Tilt 项目）
- RMM (BoM), ONI (NOAA CPC), OLR (NOAA PSL), ERA5 (CDS API)
- 路径约定：
  ClimateIndex: E:\Datas\ClimateIndex
  ERA5:         E:\Datas\ERA5
  下载记录：    E:\Datas\readme.txt
- ERA5 默认：每日 00Z 瞬时场（非日平均）
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import calendar
import requests

# ======================
# 用户指定路径（硬编码）
# ======================
CDSAPI_RC_PATH = r"C:\Users\Lenovo\.cdsapirc"
INDEX_ROOT = Path(r"E:\Datas\ClimateIndex")
ERA5_ROOT  = Path(r"E:\Datas\ERA5")
README_PATH = Path(r"E:\Datas\readme.txt")

# ======================
# 日志
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataDownloader")


def _append_readme(record: str) -> None:
    """把下载记录追加到 E:\Datas\readme.txt（若不存在则创建）"""
    try:
        README_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not README_PATH.exists():
            README_PATH.write_text("Data download log\n\n", encoding="utf-8")

        with open(README_PATH, "a", encoding="utf-8") as f:
            f.write(record.rstrip() + "\n")
    except Exception as e:
        logger.warning(f"写入 readme 失败（不影响下载本身）: {e}")


def _fmt_size(path: Path) -> str:
    if not path.exists():
        return "NA"
    n = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


class DataDownloader:
    def __init__(self):
        self.index_root = INDEX_ROOT
        self.era5_root = ERA5_ROOT
        self._create_directories()

    def _create_directories(self):
        dirs = [
            self.index_root / "rmm",
            self.index_root / "oni",
            self.index_root / "olr",
            self.era5_root / "raw" / "pressure_level",
            self.era5_root / "raw" / "single_level",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # 1) RMM
    # --------------------------
    def download_rmm_index(self) -> bool:
        """
        RMM 已手动下载：跳过在线下载
        文件应放在：E:\\Datas\\ClimateIndex\\rmm\\rmm.74toRealtime.txt
        """
        output_file = self.index_root / "rmm" / "rmm.74toRealtime.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info(f"RMM 文件已存在，跳过下载: {output_file}")
            return True

        logger.warning(f"RMM 文件不存在，请手动放置到: {output_file}")
        return False


    # --------------------------
    # 2) ONI
    # --------------------------
    def download_oni_index(self) -> bool:
        name = "ONI"
        url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
        out = self.index_root / "oni" / "oni.ascii.txt"

        logger.info(f"下载 {name} -> {out}")
        t0 = datetime.now()

        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            out.write_text(r.text, encoding="utf-8")
            ok = True
            msg = "OK"
        except Exception as e:
            ok = False
            msg = f"FAIL: {e}"

        t1 = datetime.now()
        rec = (
            f"[{t1:%Y-%m-%d %H:%M:%S}] {name}\n"
            f"  source: {url}\n"
            f"  output: {out}\n"
            f"  size:   {_fmt_size(out)}\n"
            f"  status: {msg}\n"
        )
        _append_readme(rec)
        return ok

    # --------------------------
    # 3) OLR (PSL interp_OLR)
    # --------------------------
    def download_olr_data(self) -> bool:
        name = "OLR (PSL interp_OLR daily mean)"
        url = "https://downloads.psl.noaa.gov/Datasets/interp_OLR/olr.day.mean.nc"
        out = self.index_root / "olr" / "olr.day.mean.nc"
        tmp = self.index_root / "olr" / "olr.day.mean.nc.part"

        logger.info(f"下载 {name} -> {out}")

        try:
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()

            total = int(r.headers.get("content-length", 0))
            block = 1024 * 1024  # 1MB
            downloaded = 0
            next_report = 0

            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=block):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        if pct >= next_report:
                            logger.info(f"OLR进度: {pct:.1f}%")
                            next_report += 5

            tmp.replace(out)
            ok = True
            msg = "OK"
        except Exception as e:
            ok = False
            msg = f"FAIL: {e}"

        t1 = datetime.now()
        rec = (
            f"[{t1:%Y-%m-%d %H:%M:%S}] {name}\n"
            f"  source: {url}\n"
            f"  output: {out}\n"
            f"  size:   {_fmt_size(out)}\n"
            f"  status: {msg}\n"
        )
        _append_readme(rec)
        return ok

    # --------------------------
    # 4) ERA5 pressure-levels
    # --------------------------
    def download_era5_pressure_00z(
        self,
        years=None,
        months=None,
        variables=None,
        pressure_levels=None,
        area=None,
    ) -> bool:
        """
        按月下载 ERA5 pressure-levels：每日 00Z 瞬时场（非日平均）
        文件命名：era5_pl_YYYYMM_00Z.nc

        默认设置（按项目需求）：
        - 变量：omega, q, u, v, T
        - 压力层：1000, 925, 850, 700, 500, 300, 200 （够用且避免过大）
        - 区域：20S–20N, 30E–180E
        """
        name = "ERA5 pressure-levels (daily 00Z inst, monthly files)"

        # 1) cdsapi 配置（显式指定 rc 路径）
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
        except Exception as e:
            logger.error("未安装 cdsapi，请先 pip install cdsapi")
            _append_readme(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {name}\n"
                f"  status: FAIL: cdsapi import error: {e}\n"
            )
            return False

        # 2) 默认参数
        nowy = datetime.now().year
        if years is None:
            years = list(range(1979, nowy + 1))
        if months is None:
            months = list(range(1, 13))
        if variables is None:
            variables = [
                "vertical_velocity",       # omega
                "specific_humidity",       # q
                "u_component_of_wind",     # u
                "v_component_of_wind",     # v
                "temperature",             # T
            ]
        if pressure_levels is None:
            pressure_levels = ["1000", "925", "850", "700", "500", "300", "200"]
        if area is None:
            # [North, West, South, East]；经度采用 -180..180 约定，West=30E, East=180E
            area = [20, -180, -20, 180]

        outdir = self.era5_root / "raw" / "pressure_level"
        outdir.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始下载 {name}")
        logger.info(f"CDSAPI_RC = {CDSAPI_RC_PATH}")
        logger.info(f"保存目录  = {outdir}")
        logger.info(f"区域 area = {area} (N, W, S, E)")
        logger.info(f"levels    = {pressure_levels}")
        logger.info(f"vars      = {variables}")
        logger.info("注意：该下载为每日 00Z 瞬时场（不是日平均）。")

        c = cdsapi.Client(timeout=600)

        all_ok = True
        for y in years:
            for m in months:
                out = outdir / f"era5_pl_{y}{m:02d}_00Z.nc"
                if out.exists():
                    logger.info(f"已存在，跳过: {out}")
                    continue

                ndays = calendar.monthrange(y, m)[1]
                days = [f"{d:02d}" for d in range(1, ndays + 1)]

                logger.info(f"下载 ERA5 {y}-{m:02d} -> {out}")

                try:
                    c.retrieve(
                        "reanalysis-era5-pressure-levels",
                        {
                            "product_type": "reanalysis",
                            "variable": variables,
                            "pressure_level": pressure_levels,
                            "year": str(y),
                            "month": f"{m:02d}",
                            "day": days,
                            "time": "00:00",
                            "area": area,
                            "format": "netcdf",
                        },
                        str(out),
                    )
                    msg = "OK"
                except Exception as e:
                    all_ok = False
                    msg = f"FAIL: {e}"
                    logger.error(f"ERA5 {y}-{m:02d} 下载失败: {e}")

                rec = (
                    f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {name}\n"
                    f"  year-month: {y}-{m:02d}\n"
                    f"  output:     {out}\n"
                    f"  size:       {_fmt_size(out)}\n"
                    f"  params:     time=00:00, area={area}, levels={pressure_levels}, vars={variables}\n"
                    f"  status:     {msg}\n"
                )
                _append_readme(rec)

        return all_ok

    # --------------------------
    # 5) 一键下载
    # --------------------------
    def download_all(self, include_era5=False, era5_year_start=1979, era5_year_end=None):
        results = {}
        results["RMM"] = self.download_rmm_index()
        results["ONI"] = self.download_oni_index()
        results["OLR"] = self.download_olr_data()

        if include_era5:
            if era5_year_end is None:
                era5_year_end = datetime.now().year
            years = list(range(era5_year_start, era5_year_end + 1))
            results["ERA5_PL_00Z"] = self.download_era5_pressure_00z(years=years)
        else:
            logger.info("跳过 ERA5（需要 CDS API 配置与较大磁盘空间）")

        logger.info("====== 下载结果汇总 ======")
        for k, v in results.items():
            logger.info(f"{k}: {'成功' if v else '失败'}")
        logger.info("==========================")
        return results


def main():
    import argparse

    p = argparse.ArgumentParser("Download ENSO-MJO Tilt data")
    p.add_argument("--all", action="store_true", help="下载 RMM/ONI/OLR（默认）")
    p.add_argument("--era5", action="store_true", help="额外下载 ERA5（pressure-level，按月，00Z）")
    p.add_argument("--era5-start", type=int, default=1979, help="ERA5起始年份（默认1979）")
    p.add_argument("--era5-end", type=int, default=None, help="ERA5结束年份（默认当前年）")

    p.add_argument("--rmm-only", action="store_true", help="仅下载RMM")
    p.add_argument("--oni-only", action="store_true", help="仅下载ONI")
    p.add_argument("--olr-only", action="store_true", help="仅下载OLR")
    p.add_argument("--era5-only", action="store_true", help="仅下载ERA5（pressure-level，按月，00Z）")

    args = p.parse_args()
    d = DataDownloader()

    if args.rmm_only:
        d.download_rmm_index()
        return
    if args.oni_only:
        d.download_oni_index()
        return
    if args.olr_only:
        d.download_olr_data()
        return
    if args.era5_only:
        yend = args.era5_end if args.era5_end is not None else datetime.now().year
        years = list(range(args.era5_start, yend + 1))
        d.download_era5_pressure_00z(years=years)
        return

    # 默认行为：下载指数与OLR；若 --era5 则额外下载ERA5
    include_era5 = bool(args.era5)
    d.download_all(include_era5=include_era5, era5_year_start=args.era5_start, era5_year_end=args.era5_end)


if __name__ == "__main__":
    main()
