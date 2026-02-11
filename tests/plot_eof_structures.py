# -*- coding: utf-8 -*-
"""
plot_eof_structures.py — MJO EOF 模态空间结构可视化

功能：
    读取 MV-EOF 的 PC 时间序列，通过回归方法反演 MJO 前两个模态的
    2D 空间结构（OLR 异常 + 850hPa 风矢量 + 200hPa 纬向风等值线）。
输入：
    mjo_mvEOF_step3_1979-2022.nc, OLR/U850/U200 的日均原始数据
输出：
    figures/eof_spatial_structure.png
用法：
    python tests/plot_eof_structures.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pathlib import Path

# ======================
# 用户配置 (请确认路径)
# ======================
# 1. 原始带通滤波数据路径 (必须包含 lat 维度)
OLR_BP_PATH  = r"E:\Datas\ClimateIndex\processed\olr_bp_1979-2022.nc"
U_BP_PATH    = r"E:\Datas\ERA5\processed\pressure_level\era5_u850_u200_bp_1979-2022.nc"
MJO_RES_PATH = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
OUT_FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\eof")
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# 绘图时间范围 (仅用于回归计算，建议覆盖全时段)
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"

# 绘图范围
PLOT_EXTENT = [40, 300, -30, 30]  # [LonMin, LonMax, LatMin, LatMax] (跨越日界线可以用 0-360)

def regress_field(field_da, pc_da):
    """
    Regression Map = Cov(F, PC) / Var(PC)

    Robust to field dimension order:
    - field_da can be (time,lat,lon) or (lat,lon,time) etc.
    - also supports (time,lon) for 1D hovmoller-like fields.
    """
    # 1) Time alignment
    common_time = np.intersect1d(field_da.time.values, pc_da.time.values)
    f = field_da.sel(time=common_time)
    p = pc_da.sel(time=common_time)

    # 2) Force dimension order: time first
    if set(["lat", "lon"]).issubset(f.dims):
        f = f.transpose("time", "lat", "lon")
        Y = f.values.astype(float)

        # 计算每个格点在时间维的有效样本数
        valid_n = np.sum(np.isfinite(Y), axis=0)

        # 对有效样本数>=1 的格点计算均值，否则均值设为0（避免空切片警告）
        meanY = np.zeros_like(valid_n, dtype=float)
        ok = valid_n > 0
        meanY[ok] = np.nanmean(Y[:, ok], axis=0)

        # 去均值：只对 ok 格点去；无数据格点保持 NaN
        Y = Y - meanY
        Y[:, ~ok] = np.nan
        common_time = np.intersect1d(field_da.time.values, pc_da.time.values)
        if common_time.size < 10:
            raise RuntimeError(f"Too few common_time samples: {common_time.size}")

        # X
        X = p.values.astype(float)  # (T,)
        X = X - np.nanmean(X)
        num = np.nansum(X[:, None, None] * Y, axis=0)
        den = np.nansum(X**2)
        slope = num / den
        return xr.DataArray(slope, coords={"lat": f.lat, "lon": f.lon}, name="regression_map")

    elif "lon" in f.dims and "lat" not in f.dims:
        # (time, lon) case
        f = f.transpose("time", "lon")
        Y = f.values  # (T, Lon)
        Y = Y - np.nanmean(Y, axis=0)
        X = p.values.astype(float)
        X = X - np.nanmean(X)
        num = np.nansum(X[:, None] * Y, axis=0)
        den = np.nansum(X**2)
        slope = num / den
        return xr.DataArray(slope, coords={"lon": f.lon}, name="regression_map")

    else:
        raise ValueError(f"Unsupported dims for regression: {f.dims}")


def plot_mode(ax, olr_reg, u850_reg, v850_reg, u200_reg, title):
    """绘制单个模态的子图"""
    # 1. OLR 填色 (反转色标: 负值为蓝/绿代表对流, 正值为棕色代表晴空)
    # OLR 异常通常在 +/- 40 W/m2 之间
    clevs = np.linspace(-40, 40, 17)
    cf = ax.contourf(olr_reg.lon, olr_reg.lat, olr_reg, clevs, 
                     cmap='BrBG_r', extend='both', transform=ccrs.PlateCarree())
    
    # 2. U200 等值线 (代表高层辐散/辐合)
    # 红色正值(西风), 蓝色负值(东风)
    cs = ax.contour(u200_reg.lon, u200_reg.lat, u200_reg, levels=[-10, -5, 5, 10],
                    colors=['blue', 'blue', 'red', 'red'], linewidths=1.5,
                    transform=ccrs.PlateCarree())
    
    # 3. U850 风场箭头 (低层环流)
    # 抽稀箭头，避免太密
    skip = 5

    x = u850_reg.lon.values[::skip]     # 1D numpy
    y = u850_reg.lat.values[::skip]     # 1D numpy
    u = u850_reg.values[::skip, ::skip] # 2D numpy (lat, lon)
    v = v850_reg.values[::skip, ::skip] # 2D numpy (lat, lon)

    q = ax.quiver(x, y, u, v,
                scale=100, color='k', alpha=0.8,
                transform=ccrs.PlateCarree())

    
    # 地图装饰
    ax.coastlines(linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    
    # 经纬度标签
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    
    ax.set_title(title, fontsize=12, loc='left')
    return cf

# ======================
# 主程序
# ======================
def main():
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. 读取数据...")
    try:
        ds_res = xr.open_dataset(MJO_RES_PATH)
        ds_olr = xr.open_dataset(OLR_BP_PATH)
        ds_u   = xr.open_dataset(U_BP_PATH)
    except FileNotFoundError as e:
        print(f"错误: 找不到文件 - {e}")
        return

    # 简单的数据检查与对齐
    # 确保风场有 v 分量 (如果没有，就只画 u 的箭头或者设为0)
    # 注意：你的 01 脚本下载了 v 吗？如果没有 v，绘图时可以用 0 代替，或者不画箭头
    has_v = "v850_bp" in ds_u
    if not has_v:
        print("警告: 数据中没有 v850，将只绘制纬向风(u)分量。")

    # 对齐网格 (假设 ds_u 和 ds_olr 网格可能不同，这里简单的插值到 OLR)
    print("2. 网格插值 (ERA5 -> OLR)...")
    ds_u = ds_u.interp(lat=ds_olr.lat, lon=ds_olr.lon, method="nearest")

    # 截取时间
    ds_res = ds_res.sel(time=slice(START_DATE, END_DATE))
    ds_olr = ds_olr.sel(time=slice(START_DATE, END_DATE))
    ds_u   = ds_u.sel(time=slice(START_DATE, END_DATE))

    pc1 = ds_res["pc1"] / ds_res["pc1"].std() # 再次确保标准化
    pc2 = ds_res["pc2"] / ds_res["pc2"].std()

    # 处理变量 (取 20S-20N 绘图即可，或者全场)
    olr = ds_olr["olr_bp"]
    u850 = ds_u["u850_bp"]
    u200 = ds_u["u200_bp"]
    
    if has_v:
        v850 = ds_u["v850_bp"]
    else:
        v850 = xr.zeros_like(u850)

    print("3. 计算回归场 (Regression Patterns)...")
    # Mode 1
    reg1_olr = regress_field(olr, pc1)
    reg1_u850 = regress_field(u850, pc1)
    reg1_v850 = regress_field(v850, pc1)
    reg1_u200 = regress_field(u200, pc1)
    
    # Mode 2
    reg2_olr = regress_field(olr, pc2)
    reg2_u850 = regress_field(u850, pc2)
    reg2_v850 = regress_field(v850, pc2)
    reg2_u200 = regress_field(u200, pc2)

    print("4. 绘图...")
    fig = plt.figure(figsize=(12, 4))
    
    # 子图 1: Mode 1
    ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree(central_longitude=180))
    cf1 = plot_mode(ax1, reg1_olr, reg1_u850, reg1_v850, reg1_u200, 
                    "MJO Mode 1 (Correlation with PC1)\nShading: OLR Anom, Vector: 850hPa Wind, Contour: 200hPa U-Wind")
    
    # 子图 2: Mode 2
    ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree(central_longitude=180))
    cf2 = plot_mode(ax2, reg2_olr, reg2_u850, reg2_v850, reg2_u200, 
                    "MJO Mode 2 (Correlation with PC2)")

    # 添加色标
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cb = fig.colorbar(cf1, cax=cbar_ax, orientation='horizontal', label='OLR Anomaly (W/m²)')
    
    plt.subplots_adjust(hspace=0.3)
    out_path = OUT_FIG_DIR / "mjo_eof_regression_mode1-2_1979-2022.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"完成！图片已保存至: {out_path}")
    # plt.show()

if __name__ == "__main__":
    main()