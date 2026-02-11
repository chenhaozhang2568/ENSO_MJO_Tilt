# -*- coding: utf-8 -*-
"""
validate_events_vs_rmm.py — MJO 事件与 BoM RMM 指数交叉验证

功能：
    将自识别的 MJO 事件与 BoM 官方 RMM 指数对比，
    检查事件期间 RMM 振幅、相位一致性和东传比例。
输入：
    rmm.74toRealtime.txt, mjo_events_step3_1979-2022.csv,
    mjo_mvEOF_step3_1979-2022.nc
输出：
    mjo_events_rmm_validation.csv
用法：
    python tests/validate_events_vs_rmm.py
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# ======================
# PATHS
# ======================
RMM_PATH = r"E:\Datas\ClimateIndex\raw\rmm\rmm.74toRealtime.txt"
EVENTS_PATH = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
STEP3_PATH = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"

# ======================
# 1. 读取 RMM 官方数据
# ======================
def load_rmm(path: str) -> pd.DataFrame:
    """解析 BoM RMM 文件（固定宽度格式）"""
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("RMM") or line.startswith("year"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                rmm1 = float(parts[3])
                rmm2 = float(parts[4])
                phase = int(parts[5])
                amp = float(parts[6])
            except (ValueError, IndexError):
                continue
            
            # 跳过缺测值
            if abs(rmm1) > 1e30 or abs(rmm2) > 1e30 or phase == 999:
                continue
            
            rows.append({
                "date": pd.Timestamp(year, month, day),
                "rmm1": rmm1,
                "rmm2": rmm2,
                "rmm_phase": phase,
                "rmm_amp": amp,
            })
    
    df = pd.DataFrame(rows)
    df = df.set_index("date").sort_index()
    print(f"[RMM] Loaded {len(df)} days: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    return df


# ======================
# 2. 读取自计算 PC 数据
# ======================
def load_step3(path: str) -> pd.DataFrame:
    ds = xr.open_dataset(path)
    df = pd.DataFrame({
        "date": pd.to_datetime(ds["time"].values),
        "my_pc1": ds["pc1"].values,
        "my_pc2": ds["pc2"].values,
        "my_amp": ds["amp"].values,
        "my_phase": ds["phase"].values,
    })
    df = df.set_index("date").sort_index()
    print(f"[Step3] Loaded {len(df)} days")
    return df


# ======================
# 3. 交叉验证
# ======================
def validate_events():
    # 加载数据
    rmm = load_rmm(RMM_PATH)
    step3 = load_step3(STEP3_PATH)
    events = pd.read_csv(EVENTS_PATH)
    
    print(f"\n{'='*70}")
    print(f"MJO 事件交叉验证：{len(events)} 个事件 vs BoM RMM")
    print(f"{'='*70}")
    
    # --- 先做全时段 PC 相关性检查 ---
    common_dates = rmm.index.intersection(step3.index)
    merged = rmm.loc[common_dates].join(step3.loc[common_dates], how="inner")
    valid = merged.dropna(subset=["rmm1", "rmm2", "my_pc1", "my_pc2"])
    
    # PC1/PC2 vs RMM1/RMM2 相关系数（可能需要符号翻转）
    corr_11 = np.corrcoef(valid["my_pc1"], valid["rmm1"])[0, 1]
    corr_12 = np.corrcoef(valid["my_pc1"], valid["rmm2"])[0, 1]
    corr_21 = np.corrcoef(valid["my_pc2"], valid["rmm1"])[0, 1]
    corr_22 = np.corrcoef(valid["my_pc2"], valid["rmm2"])[0, 1]
    
    print(f"\n[全时段 PC vs RMM 相关矩阵] ({len(valid)} 天)")
    print(f"              RMM1      RMM2")
    print(f"  my_PC1    {corr_11:+.3f}    {corr_12:+.3f}")
    print(f"  my_PC2    {corr_21:+.3f}    {corr_22:+.3f}")
    
    # 振幅相关
    corr_amp = np.corrcoef(valid["my_amp"], valid["rmm_amp"])[0, 1]
    print(f"\n  振幅相关: my_amp vs rmm_amp = {corr_amp:+.3f}")
    
    # --- 逐事件验证 ---
    print(f"\n{'='*70}")
    print(f"逐事件验证")
    print(f"{'='*70}")
    
    results = []
    for _, ev in events.iterrows():
        eid = ev["event_id"]
        start = pd.Timestamp(ev["start_date"])
        end = pd.Timestamp(ev["end_date"])
        
        # 提取事件期间的 RMM
        mask = (rmm.index >= start) & (rmm.index <= end)
        rmm_seg = rmm.loc[mask]
        
        if len(rmm_seg) == 0:
            results.append({
                "event_id": eid, "start": str(start.date()), "end": str(end.date()),
                "rmm_days": 0, "rmm_active_frac": np.nan,
                "rmm_amp_mean": np.nan, "rmm_amp_median": np.nan,
                "phase_match": "NO_DATA",
            })
            continue
        
        # RMM 活跃天比例 (amp > 1)
        active_frac = (rmm_seg["rmm_amp"] > 1.0).mean()
        
        # RMM 振幅统计
        amp_mean = rmm_seg["rmm_amp"].mean()
        amp_median = rmm_seg["rmm_amp"].median()
        
        # RMM 相位传播检查：取众数相位范围
        phases = rmm_seg["rmm_phase"].values
        phase_min = int(phases.min())
        phase_max = int(phases.max())
        
        # 计算相位是否有持续东传趋势
        # 简单方法：检查相位序列是否大致递增（允许 8→1 跳转）
        phase_changes = np.diff(phases)
        # 东传 = 相位增加 (+1) 或跨越 (8→1 = -7)
        eastward = ((phase_changes == 1) | (phase_changes == -7)).sum()
        westward = ((phase_changes == -1) | (phase_changes == 7)).sum()
        total_changes = (phase_changes != 0).sum()
        east_frac = eastward / max(total_changes, 1)
        
        # 提取自己的事件段
        my_mask = (step3.index >= start) & (step3.index <= end)
        my_seg = step3.loc[my_mask]
        
        # 振幅对比
        my_amp_mean = my_seg["my_amp"].mean() if len(my_seg) > 0 else np.nan
        
        results.append({
            "event_id": eid,
            "start": str(start.date()),
            "end": str(end.date()),
            "duration": ev["duration_days"],
            "rmm_days": len(rmm_seg),
            "rmm_active_frac": active_frac,
            "rmm_amp_mean": amp_mean,
            "my_amp_mean": my_amp_mean,
            "rmm_phase_range": f"{phase_min}-{phase_max}",
            "east_prop_frac": east_frac,
        })
    
    df_results = pd.DataFrame(results)
    
    # --- 汇总统计 ---
    valid_results = df_results.dropna(subset=["rmm_active_frac"])
    
    print(f"\n[汇总统计]")
    print(f"  总事件数: {len(events)}")
    print(f"  有 RMM 数据的事件: {len(valid_results)}")
    
    # 高质量事件：>50% 天数 RMM 活跃
    high_quality = valid_results[valid_results["rmm_active_frac"] > 0.5]
    medium_quality = valid_results[(valid_results["rmm_active_frac"] > 0.3) & (valid_results["rmm_active_frac"] <= 0.5)]
    low_quality = valid_results[valid_results["rmm_active_frac"] <= 0.3]
    
    print(f"\n  [RMM 活跃比例分级]")
    print(f"    ✅ 高质量 (>50% 天 RMM amp>1): {len(high_quality)} 事件 ({len(high_quality)/len(valid_results)*100:.1f}%)")
    print(f"    ⚠️  中等    (30-50%):            {len(medium_quality)} 事件 ({len(medium_quality)/len(valid_results)*100:.1f}%)")
    print(f"    ❌ 低质量 (<30%):               {len(low_quality)} 事件 ({len(low_quality)/len(valid_results)*100:.1f}%)")
    
    print(f"\n  [RMM 振幅统计]")
    print(f"    事件期间 RMM amp 均值: {valid_results['rmm_amp_mean'].mean():.2f} ± {valid_results['rmm_amp_mean'].std():.2f}")
    print(f"    事件期间 my amp 均值:  {valid_results['my_amp_mean'].mean():.2f} ± {valid_results['my_amp_mean'].std():.2f}")
    
    print(f"\n  [东传一致性]")
    east_prop = valid_results["east_prop_frac"]
    print(f"    东传相位变化比例 (均值): {east_prop.mean():.2f}")
    print(f"    东传相位变化比例 > 0.4 的事件: {(east_prop > 0.4).sum()} / {len(valid_results)}")
    
    # --- 打印详细表格 ---
    print(f"\n{'='*70}")
    print(f"逐事件详情")
    print(f"{'='*70}")
    print(f"{'ID':>3} {'Start':>12} {'End':>12} {'Dur':>4} {'RMM_act%':>8} {'RMM_amp':>8} {'My_amp':>7} {'Phase':>7} {'East%':>6}")
    print("-" * 75)
    
    for _, r in df_results.iterrows():
        act_str = f"{r['rmm_active_frac']*100:.0f}%" if pd.notna(r.get("rmm_active_frac")) else "N/A"
        rmm_amp_str = f"{r['rmm_amp_mean']:.2f}" if pd.notna(r.get("rmm_amp_mean")) else "N/A"
        my_amp_str = f"{r['my_amp_mean']:.2f}" if pd.notna(r.get("my_amp_mean")) else "N/A"
        phase_str = r.get("rmm_phase_range", "N/A")
        east_str = f"{r['east_prop_frac']*100:.0f}%" if pd.notna(r.get("east_prop_frac")) else "N/A"
        
        # 标记质量
        if pd.notna(r.get("rmm_active_frac")):
            if r["rmm_active_frac"] > 0.5:
                mark = "✅"
            elif r["rmm_active_frac"] > 0.3:
                mark = "⚠️"
            else:
                mark = "❌"
        else:
            mark = "❓"
        
        print(f"{int(r['event_id']):3d} {r['start']:>12} {r['end']:>12} {int(r['duration']):4d} {act_str:>8} {rmm_amp_str:>8} {my_amp_str:>7} {phase_str:>7} {east_str:>6} {mark}")
    
    # 保存结果
    out_dir = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\validation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mjo_events_rmm_validation.csv"
    df_results.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    validate_events()
