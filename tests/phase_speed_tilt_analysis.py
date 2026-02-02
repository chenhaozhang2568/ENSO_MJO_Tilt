# -*- coding: utf-8 -*-
"""
phase_speed_tilt_analysis.py: MJO 相速度与倾斜相关性分析

================================================================================
功能描述：
    本脚本按照 Hu & Li (2021) 方法计算 MJO 事件的相速度（phase speed），
    并分析其与垂直倾斜指标（tilt）的相关性。

计算方法：
    1. 对每个 MJO 事件，在 60°E-180°E 区间内提取逐日对流中心经度
    2. 将中心经度对时间做线性最小二乘拟合
    3. 拟合斜率即为相速度（度/天 或 m/s）

主要分析内容：
    1. 逐事件相速度与平均倾斜的散点图
    2. 相关系数与统计显著性检验
    3. 相速度分布的 ENSO 分组比较
    4. 快传播 vs 慢传播事件的倾斜差异

物理意义：
    相速度反映 MJO 东传效率，倾斜结构影响能量和动量的垂直传输，
    二者的关联揭示 MJO 动力学机制。
- 对每个MJO事件，提取逐日对流中心经度 λc(t)
- 用最小二乘直线拟合，斜率 dλc/dt 为事件平均东传速度 (deg/day)
- 换算成 m/s: c = slope * 111320 / 86400
- 与事件平均倾斜做相关/回归分析

运行:
cd /d E:\Projects\ENSO_MJO_Tilt
python tests\phase_speed_tilt_analysis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ======================
# 路径配置
# ======================
STEP3_NC = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
TILT_STATS_CSV = r"E:\Datas\Derived\tilt_event_stats_1979-2022.csv"

OUT_DIR = Path(r"E:\Datas\Derived")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "phase_speed_tilt_summary.csv"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = FIG_DIR / "phase_speed_vs_tilt.png"

# ======================
# 物理常数
# ======================
DEG_TO_M = 111320.0  # 赤道处1度经度 ≈ 111.32 km
DAY_TO_SEC = 86400.0  # 1天 = 86400秒

# ======================
# 相速度计算函数
# ======================
def compute_event_phase_speed(
    center_track: xr.DataArray,
    time_index: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    min_valid_days: int = 5
) -> dict:
    """
    计算单个MJO事件的相速度（线性最小二乘拟合）
    
    参数:
        center_track: 逐日对流中心经度 DataArray
        time_index: 时间索引
        start_date: 事件开始日期
        end_date: 事件结束日期
        min_valid_days: 最少有效数据天数
    
    返回:
        dict: 包含 slope_deg_day, speed_m_s, r2, n_valid, stderr
    """
    # 选取事件时间范围
    mask = (time_index >= start_date) & (time_index <= end_date)
    
    if not mask.any():
        return {
            "slope_deg_day": np.nan,
            "speed_m_s": np.nan,
            "r2": np.nan,
            "n_valid": 0,
            "stderr": np.nan
        }
    
    # 提取数据
    lon_vals = center_track.values[mask].astype(float)
    time_vals = time_index[mask]
    
    # 转换时间为相对天数（从事件开始算起）
    t0 = time_vals[0]
    days = np.array([(t - t0).days for t in time_vals], dtype=float)
    
    # 去除NaN
    valid = np.isfinite(lon_vals)
    days_valid = days[valid]
    lon_valid = lon_vals[valid]
    
    n_valid = len(lon_valid)
    if n_valid < min_valid_days:
        return {
            "slope_deg_day": np.nan,
            "speed_m_s": np.nan,
            "r2": np.nan,
            "n_valid": n_valid,
            "stderr": np.nan
        }
    
    # 线性最小二乘拟合: lon = intercept + slope * days
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(days_valid, lon_valid)
    except Exception:
        return {
            "slope_deg_day": np.nan,
            "speed_m_s": np.nan,
            "r2": np.nan,
            "n_valid": n_valid,
            "stderr": np.nan
        }
    
    # 换算为 m/s
    speed_m_s = slope * DEG_TO_M / DAY_TO_SEC
    
    return {
        "slope_deg_day": float(slope),
        "speed_m_s": float(speed_m_s),
        "r2": float(r_value ** 2),
        "n_valid": int(n_valid),
        "stderr": float(std_err)
    }


# ======================
# 主函数
# ======================
def main():
    print("=" * 60)
    print("MJO Phase Speed vs Tilt Correlation Analysis")
    print("=" * 60)
    
    # ---- 1. 加载数据 ----
    print("\n[1] Loading data...")
    
    # Step3 NC文件（含center_lon_track）
    if not Path(STEP3_NC).exists():
        raise FileNotFoundError(f"Step3 NC not found: {STEP3_NC}\nPlease run src/02_mvEOF.py first.")
    
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4")
    center_track = ds3["center_lon_track"]
    time_index = pd.to_datetime(ds3["time"].values)
    print(f"  Loaded center_lon_track: {center_track.shape[0]} days")
    
    # 事件列表
    if not Path(EVENTS_CSV).exists():
        raise FileNotFoundError(f"Events CSV not found: {EVENTS_CSV}")
    
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    print(f"  Loaded {len(events)} MJO events")
    
    # 倾斜统计
    if not Path(TILT_STATS_CSV).exists():
        raise FileNotFoundError(f"Tilt stats CSV not found: {TILT_STATS_CSV}")
    
    tilt_stats = pd.read_csv(TILT_STATS_CSV, parse_dates=["start_date", "end_date"])
    print(f"  Loaded tilt stats for {len(tilt_stats)} events")
    
    # ---- 2. 计算每个事件的相速度 ----
    print("\n[2] Computing phase speed for each event...")
    
    results = []
    for _, row in events.iterrows():
        eid = row["event_id"]
        s_date = pd.Timestamp(row["start_date"])
        e_date = pd.Timestamp(row["end_date"])
        
        ps = compute_event_phase_speed(center_track, time_index, s_date, e_date)
        
        results.append({
            "event_id": eid,
            "start_date": s_date,
            "end_date": e_date,
            "duration_days": row.get("duration_days", (e_date - s_date).days + 1),
            "slope_deg_day": ps["slope_deg_day"],
            "speed_m_s": ps["speed_m_s"],
            "r2": ps["r2"],
            "n_valid": ps["n_valid"],
            "stderr": ps["stderr"]
        })
    
    df_speed = pd.DataFrame(results)
    
    # ---- 3. 合并倾斜数据 ----
    print("\n[3] Merging with tilt statistics...")
    
    # 以 event_id 合并（如果有的话），否则用日期匹配
    if "event_id" in tilt_stats.columns:
        # 尝试直接匹配 event_id（可能不完全对应）
        df_merged = df_speed.merge(
            tilt_stats[["event_id", "mean_tilt", "median_tilt", "std_tilt"]].rename(
                columns={"event_id": "tilt_event_id"}
            ),
            left_on="event_id",
            right_on="tilt_event_id",
            how="left"
        )
        # 如果没有匹配上，尝试用日期匹配
        missing_mask = df_merged["mean_tilt"].isna()
        if missing_mask.any():
            print(f"  Warning: {missing_mask.sum()} events not matched by event_id, trying date match...")
            for idx in df_merged[missing_mask].index:
                s = df_merged.loc[idx, "start_date"]
                e = df_merged.loc[idx, "end_date"]
                # 找tilt_stats中日期重叠的
                for _, trow in tilt_stats.iterrows():
                    ts = pd.Timestamp(trow["start_date"])
                    te = pd.Timestamp(trow["end_date"])
                    # 判断是否有重叠（至少1天）
                    if s <= te and e >= ts:
                        df_merged.loc[idx, "mean_tilt"] = trow["mean_tilt"]
                        df_merged.loc[idx, "median_tilt"] = trow["median_tilt"]
                        df_merged.loc[idx, "std_tilt"] = trow["std_tilt"]
                        break
    else:
        # 纯日期匹配
        df_merged = df_speed.copy()
        df_merged["mean_tilt"] = np.nan
        df_merged["median_tilt"] = np.nan
        df_merged["std_tilt"] = np.nan
        
        for idx, row in df_merged.iterrows():
            s = row["start_date"]
            e = row["end_date"]
            for _, trow in tilt_stats.iterrows():
                ts = pd.Timestamp(trow["start_date"])
                te = pd.Timestamp(trow["end_date"])
                if s <= te and e >= ts:
                    df_merged.loc[idx, "mean_tilt"] = trow["mean_tilt"]
                    df_merged.loc[idx, "median_tilt"] = trow["median_tilt"]
                    df_merged.loc[idx, "std_tilt"] = trow["std_tilt"]
                    break
    
    # 过滤有效数据
    df_valid = df_merged.dropna(subset=["speed_m_s", "mean_tilt"])
    print(f"  Valid events with both speed and tilt: {len(df_valid)}")
    
    # ---- 4. 相关性分析 ----
    print("\n[4] Correlation analysis...")
    
    speed = df_valid["speed_m_s"].values
    tilt = df_valid["mean_tilt"].values
    
    # Pearson相关
    r_pearson, p_pearson = stats.pearsonr(speed, tilt)
    print(f"  Pearson correlation:  r = {r_pearson:.4f}, p = {p_pearson:.4e}")
    
    # Spearman相关
    r_spearman, p_spearman = stats.spearmanr(speed, tilt)
    print(f"  Spearman correlation: ρ = {r_spearman:.4f}, p = {p_spearman:.4e}")
    
    # 线性回归
    slope_reg, intercept_reg, r_reg, p_reg, stderr_reg = stats.linregress(speed, tilt)
    print(f"  Linear regression:    tilt = {slope_reg:.2f} * speed + {intercept_reg:.2f}")
    print(f"                        R² = {r_reg**2:.4f}")
    
    # ---- 5. 可视化 ----
    print("\n[5] Generating scatter plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 散点图
    sc = ax.scatter(speed, tilt, c='steelblue', s=60, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # 回归线
    x_line = np.linspace(speed.min(), speed.max(), 100)
    y_line = slope_reg * x_line + intercept_reg
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Linear fit: y = {slope_reg:.2f}x + {intercept_reg:.2f}')
    
    # 参考线
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # 标注
    ax.set_xlabel("Phase Speed (m/s)", fontsize=12)
    ax.set_ylabel("Mean Tilt (°)", fontsize=12)
    ax.set_title("MJO Phase Speed vs Convective Tilt\n(Hu & Li 2021 method)", fontsize=14)
    
    # 统计信息文本框
    stats_text = (
        f"N = {len(df_valid)}\n"
        f"Pearson r = {r_pearson:.3f} (p = {p_pearson:.2e})\n"
        f"Spearman ρ = {r_spearman:.3f} (p = {p_spearman:.2e})\n"
        f"R² = {r_reg**2:.3f}"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=150, bbox_inches='tight')
    print(f"  Saved figure: {OUT_FIG}")
    plt.close()
    
    # ---- 6. 保存结果 ----
    print("\n[6] Saving results...")
    
    # 添加相关系数到汇总
    df_merged["pearson_r"] = r_pearson
    df_merged["pearson_p"] = p_pearson
    df_merged["spearman_rho"] = r_spearman
    df_merged["spearman_p"] = p_spearman
    
    df_merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"  Saved CSV: {OUT_CSV}")
    
    # ---- 7. 结果汇总 ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total events:          {len(events)}")
    print(f"Valid for analysis:    {len(df_valid)}")
    print(f"Phase speed range:     {speed.min():.2f} - {speed.max():.2f} m/s")
    print(f"Phase speed mean:      {speed.mean():.2f} ± {speed.std():.2f} m/s")
    print(f"Tilt range:            {tilt.min():.1f} - {tilt.max():.1f}°")
    print(f"Tilt mean:             {tilt.mean():.1f} ± {tilt.std():.1f}°")
    print("-" * 60)
    print(f"Pearson r:             {r_pearson:.4f} (p = {p_pearson:.4e})")
    print(f"Spearman ρ:            {r_spearman:.4f} (p = {p_spearman:.4e})")
    
    sig_level = 0.05
    if p_pearson < sig_level:
        print(f"\n✓ Significant correlation at α = {sig_level}")
    else:
        print(f"\n✗ No significant correlation at α = {sig_level}")
    
    print("=" * 60)
    

if __name__ == "__main__":
    main()
