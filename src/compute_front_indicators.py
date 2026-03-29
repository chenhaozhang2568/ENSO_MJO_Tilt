# -*- coding: utf-8 -*-
"""
compute_front_indicators.py: 计算7种低层前端指标

================================================================================
功能描述：
    计算7种可能与 MJO 相速度正相关的前端指标（零交叉、梯度极值、边界位置）。
    输出逐日值到 NetCDF，汇总逐事件平均值到 CSV。

七种指标：
    F1: q 前端零交叉 (低层q 1000-850hPa层平均, 对流东侧正→负, 5°容忍, 搜索0~180°)
    F2: omega 下沉前端 (低层omega 1000-850hPa层平均, 对流东侧负→正, 5°容忍)
    F3: u 辐合前端 (低层u 1000-850hPa层平均, 正→负, 搜索-90~180°)
    F4: q 梯度极值位置 (低层q dq/dx 对流东侧最大负值经度)
    F5: T 正异常前端 (低层T 1000-850hPa层平均, 对流东侧正→负)
    F6: omega 低层东边界 (低层omega 上升区东边界, 5°容忍)
    F7: u风垂直切变转换 (u_upper(400-200hPa) - u_lower(1000-850hPa), 对流东侧第一个符号转变)

输入数据：
    - mjo_mvEOF_step3_1979-2022.nc (对流中心轨迹)
    - mjo_events_step3_1979-2022.csv (事件列表)
    - phase_speed_q_events.csv (相速度)
    - era5_mjo_recon_{w,q,u,t}_norm_1979-2022.nc (归一化重构场)

输出：
    - front_indicators_daily.nc (逐日7个指标值)
    - front_indicators_event_mean.csv (逐事件均值 + 相速度)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.interpolate import Akima1DInterpolator

# ======================
# PATHS
# ======================
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
SPEED_CSV  = r"E:\Datas\Derived\phase_speed_q_events.csv"
W_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"
Q_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_q_norm_1979-2022.nc"
U_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_u_norm_1979-2022.nc"
T_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_t_norm_1979-2022.nc"

OUT_NC     = r"E:\Datas\Derived\front_indicators_daily.nc"
OUT_CSV    = r"E:\Datas\Derived\front_indicators_event_mean.csv"

# ======================
# SETTINGS
# ======================
START_DATE = "1979-01-01"
END_DATE   = "2022-12-31"
WINTER_MONTHS = {11, 12, 1, 2, 3, 4}

# 层次定义
LOW_LAYER  = (1000.0, 850.0)   # 低层 1000-850 hPa
UP_LAYER   = (400.0, 200.0)    # 高层 400-200 hPa

# 插值/平滑参数 (与 03b 一致)
CSA_TARGET_DLON = 0.25
SMOOTH_WINDOW = 10
MIN_VALID_POINTS = 7

# 搜索范围 (相对经度)
SEARCH_DEFAULT_MIN = 0.0
SEARCH_DEFAULT_MAX = 90.0

# 容忍度 (°): 零交叉后需持续保持新符号至少 TOLERANCE_DEG 度
TOLERANCE_DEG = 5.0


# ======================
# 辅助函数
# ======================
def _load_field(nc_path, var_name):
    """加载归一化重构场，统一维度名和经度到 0-360。"""
    ds = xr.open_dataset(nc_path, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    da = ds[var_name]
    if "pressure_level" in da.dims:
        da = da.rename({"pressure_level": "level"})
    lon_vals = da["lon"].values
    if lon_vals.min() < 0:
        new_lon = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
        da = da.assign_coords(lon=new_lon).sortby("lon")
    return da


def _smooth_1d(profile, window):
    """1D 滑动平均。"""
    if window <= 1:
        return profile
    kernel = np.ones(window) / window
    valid = np.isfinite(profile).astype(float)
    filled = np.where(np.isfinite(profile), profile, 0.0)
    smoothed = np.convolve(filled, kernel, mode='same')
    count = np.convolve(valid, kernel, mode='same')
    count[count < 1e-10] = np.nan
    return smoothed / count


def _interp_to_target(src_lon, profile, target_lon):
    """Akima 插值到目标经度网格。"""
    valid = np.isfinite(profile)
    if valid.sum() < 4:
        return np.full(len(target_lon), np.nan)
    return Akima1DInterpolator(src_lon[valid], profile[valid])(target_lon)


def _prepare_profile(raw_2d, lon_360, center, layer_levels, target_rel):
    """
    对单日数据：截取相对经度范围，层平均，插值+平滑。
    rel_lon 范围由 target_rel 的 min/max 决定（带 5° 余量）。
    """
    rel_lon = lon_360 - center
    # 处理经度跨越 0/360 的情况
    rel_lon = np.where(rel_lon > 180, rel_lon - 360, rel_lon)
    rel_lon = np.where(rel_lon < -180, rel_lon + 360, rel_lon)

    tr_min = target_rel.min() - 5
    tr_max = target_rel.max() + 5
    mask = (rel_lon >= tr_min) & (rel_lon <= tr_max)
    if mask.sum() < MIN_VALID_POINTS:
        return np.full(len(target_rel), np.nan)

    rel_sub = rel_lon[mask]
    data_sub = raw_2d[:, mask]  # (level, lon_sub)

    # 排序（经度跨越时可能乱序）
    sort_idx = np.argsort(rel_sub)
    rel_sub = rel_sub[sort_idx]
    data_sub = data_sub[:, sort_idx]

    # 层平均
    layer_data = data_sub[layer_levels, :]
    if layer_data.shape[0] == 0:
        return np.full(len(target_rel), np.nan)
    profile = np.nanmean(layer_data, axis=0)

    # 插值 + 平滑
    interped = _interp_to_target(rel_sub, profile, target_rel)
    smoothed = _smooth_1d(interped, SMOOTH_WINDOW)
    return smoothed


# ---- 零交叉检测函数 ----

def _find_zero_crossing_pos_to_neg(rel_lon, profile, search_min, search_max,
                                    tolerance=0.0):
    """
    在 [search_min, search_max] 范围内找 profile 从正→负的零交叉点。
    如 tolerance>0, 则零交叉后 profile 需保持 ≤0 至少 tolerance 度。
    """
    mask = (rel_lon >= search_min) & (rel_lon <= search_max)
    if mask.sum() < 2:
        return np.nan
    rr = rel_lon[mask]
    pp = profile[mask]

    for i in range(len(pp) - 1):
        if np.isfinite(pp[i]) and np.isfinite(pp[i+1]):
            if pp[i] > 0 and pp[i+1] <= 0:
                x_zero = rr[i] - pp[i] * (rr[i+1] - rr[i]) / (pp[i+1] - pp[i])
                # 容忍度检查
                if tolerance > 0:
                    if not _check_sign_sustained(rr, pp, i+1, tolerance, sign='neg'):
                        continue
                return float(x_zero)
    return np.nan


def _find_zero_crossing_neg_to_pos(rel_lon, profile, search_min, search_max,
                                    tolerance=0.0):
    """
    在 [search_min, search_max] 范围内找 profile 从负→正的零交叉点。
    如 tolerance>0, 则零交叉后 profile 需保持 ≥0 至少 tolerance 度。
    """
    mask = (rel_lon >= search_min) & (rel_lon <= search_max)
    if mask.sum() < 2:
        return np.nan
    rr = rel_lon[mask]
    pp = profile[mask]

    for i in range(len(pp) - 1):
        if np.isfinite(pp[i]) and np.isfinite(pp[i+1]):
            if pp[i] < 0 and pp[i+1] >= 0:
                x_zero = rr[i] - pp[i] * (rr[i+1] - rr[i]) / (pp[i+1] - pp[i])
                if tolerance > 0:
                    if not _check_sign_sustained(rr, pp, i+1, tolerance, sign='pos'):
                        continue
                return float(x_zero)
    return np.nan


def _check_sign_sustained(rr, pp, start_idx, tolerance_deg, sign='neg'):
    """
    从 start_idx 开始，检查 profile 是否在 tolerance_deg 范围内
    持续保持指定符号。
    sign='neg': 检查 profile ≤ 0
    sign='pos': 检查 profile ≥ 0
    """
    x_start = rr[start_idx]
    x_end = x_start + tolerance_deg
    for j in range(start_idx, len(rr)):
        if rr[j] > x_end:
            return True  # 已检查够 tolerance_deg，通过
        if not np.isfinite(pp[j]):
            continue
        if sign == 'neg' and pp[j] > 0:
            return False
        if sign == 'pos' and pp[j] < 0:
            return False
    # 数据不足以覆盖 tolerance_deg，仍然认为通过
    return True


def _find_first_zero_crossing(rel_lon, profile, search_min=0.0, search_max=90.0):
    """
    在 [search_min, search_max] 范围内找 profile 的第一个零交叉点（任意方向）。
    """
    mask = (rel_lon >= search_min) & (rel_lon <= search_max)
    if mask.sum() < 2:
        return np.nan
    rr = rel_lon[mask]
    pp = profile[mask]

    for i in range(len(pp) - 1):
        if np.isfinite(pp[i]) and np.isfinite(pp[i+1]):
            if pp[i] * pp[i+1] < 0:
                x_zero = rr[i] - pp[i] * (rr[i+1] - rr[i]) / (pp[i+1] - pp[i])
                return float(x_zero)
    return np.nan


def _find_gradient_min_position(rel_lon, profile, search_min=0.0, search_max=90.0):
    """
    在 [search_min, search_max] 范围内找 dq/dx 最大负值的位置。
    """
    mask = (rel_lon >= search_min) & (rel_lon <= search_max)
    if mask.sum() < 3:
        return np.nan
    rr = rel_lon[mask]
    pp = profile[mask]

    dq = np.diff(pp)
    dx = np.diff(rr)
    grad = dq / dx
    rr_mid = 0.5 * (rr[:-1] + rr[1:])

    valid = np.isfinite(grad)
    if valid.sum() < 1:
        return np.nan

    grad_valid = grad.copy()
    grad_valid[~valid] = 0.0
    min_idx = np.nanargmin(grad_valid)
    if grad_valid[min_idx] >= 0:
        return np.nan
    return float(rr_mid[min_idx])


def _find_ascent_east_boundary(rel_lon, profile, tolerance=0.0):
    """
    从对流中心 (rel=0) 向东找上升区 (omega<0) 的东边界。
    如 tolerance>0, 则边界之后 omega 需保持 ≥0 至少 tolerance 度。
    """
    m = np.isfinite(profile) & np.isfinite(rel_lon)
    if m.sum() < MIN_VALID_POINTS:
        return np.nan
    rr = rel_lon[m]
    pp = profile[m]

    pivot_idx = int(np.argmin(np.abs(rr)))
    if pp[pivot_idx] >= 0:
        return np.nan  # 中心无上升运动

    for i in range(pivot_idx, len(pp) - 1):
        if pp[i] < 0 and pp[i+1] >= 0:
            x_zero = rr[i] - pp[i] * (rr[i+1] - rr[i]) / (pp[i+1] - pp[i])
            if tolerance > 0:
                if not _check_sign_sustained(rr, pp, i+1, tolerance, sign='pos'):
                    continue
            return float(x_zero)
    return np.nan


def main():
    print("=" * 60)
    print("compute_front_indicators.py: 计算7种前端指标")
    print("=" * 60)

    Path(OUT_NC).parent.mkdir(parents=True, exist_ok=True)

    # --- 加载事件和轨迹 ---
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ds3 = xr.open_dataset(STEP3_NC, engine="netcdf4").sel(time=slice(START_DATE, END_DATE))
    center_lon = ds3["center_lon_track"].values.astype(float)
    time_all = pd.to_datetime(ds3["time"].values)

    # --- 加载相速度 ---
    df_speed = pd.read_csv(SPEED_CSV)
    print(f"  Events: {len(events)}, Speed records: {len(df_speed)}")

    # --- 加载 4 个归一化重构场 ---
    print("Loading w_norm...")
    w_da = _load_field(W_NORM_NC, "w_mjo_recon_norm")
    print("Loading q_norm...")
    q_da = _load_field(Q_NORM_NC, "q_mjo_recon_norm")
    print("Loading u_norm...")
    u_da = _load_field(U_NORM_NC, "u_mjo_recon_norm")
    print("Loading t_norm...")
    t_da = _load_field(T_NORM_NC, "t_mjo_recon_norm")

    # 提取原始数组
    w_raw = w_da.values    # (time, level, lon)
    q_raw = q_da.values
    u_raw = u_da.values
    t_raw = t_da.values

    levels_w = w_da["level"].values.astype(float)
    levels_q = q_da["level"].values.astype(float)
    levels_u = u_da["level"].values.astype(float)
    levels_t = t_da["level"].values.astype(float)

    lon_w = w_da["lon"].values.astype(float)
    lon_q = q_da["lon"].values.astype(float)
    lon_u = u_da["lon"].values.astype(float)
    lon_t = t_da["lon"].values.astype(float)

    time_w = pd.to_datetime(w_da["time"].values)

    # 层 mask
    low_mask_w = (levels_w >= min(LOW_LAYER)) & (levels_w <= max(LOW_LAYER))
    low_mask_q = (levels_q >= min(LOW_LAYER)) & (levels_q <= max(LOW_LAYER))
    low_mask_u = (levels_u >= min(LOW_LAYER)) & (levels_u <= max(LOW_LAYER))
    low_mask_t = (levels_t >= min(LOW_LAYER)) & (levels_t <= max(LOW_LAYER))
    up_mask_u  = (levels_u >= min(UP_LAYER))  & (levels_u <= max(UP_LAYER))

    print(f"  w levels: {levels_w}, low_mask count: {low_mask_w.sum()}")
    print(f"  q levels: {levels_q}, low_mask count: {low_mask_q.sum()}")
    print(f"  u levels: {levels_u}, low_mask count: {low_mask_u.sum()}, up_mask count: {up_mask_u.sum()}")
    print(f"  t levels: {levels_t}, low_mask count: {low_mask_t.sum()}")

    # 目标相对经度网格 (扩展到 -90~180° 以适应 F1/F3 的宽范围搜索)
    target_rel_wide = np.arange(-90, 180 + CSA_TARGET_DLON, CSA_TARGET_DLON)
    # 默认范围 -90~90
    target_rel_default = np.arange(-90, 90 + CSA_TARGET_DLON, CSA_TARGET_DLON)

    # 事件日 mask
    winter_mask = np.isin(time_w.month, list(WINTER_MONTHS))
    event_mask = np.zeros(len(time_w), dtype=bool)
    for _, row in events.iterrows():
        ts = np.datetime64(row["start_date"])
        te = np.datetime64(row["end_date"])
        m = (time_w >= ts) & (time_w <= te)
        event_mask |= m

    valid_mask = winter_mask & event_mask

    # 时间对齐
    time_w_arr = time_w.values
    time_all_arr = time_all.values

    # pre-allocate
    n = len(time_w)
    f1 = np.full(n, np.nan, np.float32)
    f2 = np.full(n, np.nan, np.float32)
    f3 = np.full(n, np.nan, np.float32)
    f4 = np.full(n, np.nan, np.float32)
    f5 = np.full(n, np.nan, np.float32)
    f6 = np.full(n, np.nan, np.float32)
    f7 = np.full(n, np.nan, np.float32)

    # --- 逐日计算 ---
    n_valid = int(valid_mask.sum())
    print(f"\nProcessing {n_valid} valid days...")

    count = 0
    for i in range(n):
        if not valid_mask[i]:
            continue

        t_val = time_w_arr[i]
        idx_center = np.searchsorted(time_all_arr, t_val)
        if idx_center >= len(center_lon):
            continue
        c = center_lon[idx_center]
        if not np.isfinite(c):
            continue

        # === F1: q front (低层q, 正→负, 搜索0~180°, 5°容忍) ===
        q_profile_wide = _prepare_profile(q_raw[i], lon_q, c, low_mask_q, target_rel_wide)
        f1[i] = _find_zero_crossing_pos_to_neg(
            target_rel_wide, q_profile_wide,
            search_min=0.0, search_max=180.0, tolerance=TOLERANCE_DEG)

        # === F2: omega subsidence front (低层omega, 负→正, 5°容忍) ===
        w_low_profile = _prepare_profile(w_raw[i], lon_w, c, low_mask_w, target_rel_default)
        f2[i] = _find_zero_crossing_neg_to_pos(
            target_rel_default, w_low_profile,
            search_min=0.0, search_max=90.0, tolerance=TOLERANCE_DEG)

        # === F3: u convergence front (低层u, 正→负, 搜索-90~180°, 考虑负值) ===
        u_low_profile_wide = _prepare_profile(u_raw[i], lon_u, c, low_mask_u, target_rel_wide)
        f3[i] = _find_zero_crossing_pos_to_neg(
            target_rel_wide, u_low_profile_wide,
            search_min=-90.0, search_max=180.0, tolerance=0.0)

        # === F4: q gradient max (低层q dq/dx 最大负值位置) ===
        q_profile = _prepare_profile(q_raw[i], lon_q, c, low_mask_q, target_rel_default)
        f4[i] = _find_gradient_min_position(target_rel_default, q_profile,
                                             search_min=0.0, search_max=90.0)

        # === F5: T front (低层T, 正→负零交叉) ===
        t_profile = _prepare_profile(t_raw[i], lon_t, c, low_mask_t, target_rel_default)
        f5[i] = _find_zero_crossing_pos_to_neg(
            target_rel_default, t_profile,
            search_min=0.0, search_max=90.0, tolerance=0.0)

        # === F6: omega low east boundary (低层omega 上升区东边界, 5°容忍) ===
        f6[i] = _find_ascent_east_boundary(target_rel_default, w_low_profile,
                                            tolerance=TOLERANCE_DEG)

        # === F7: u vertical shear (u_upper - u_lower, 第一个符号变化) ===
        u_low_profile = _prepare_profile(u_raw[i], lon_u, c, low_mask_u, target_rel_default)
        u_up_profile = _prepare_profile(u_raw[i], lon_u, c, up_mask_u, target_rel_default)
        shear_profile = u_up_profile - u_low_profile
        f7[i] = _find_first_zero_crossing(target_rel_default, shear_profile,
                                            search_min=0.0, search_max=90.0)

        count += 1
        if count % 500 == 0:
            print(f"  Processed {count}/{n_valid} days...")

    print(f"  Done. Processed {count} days.")

    # --- 保存逐日 NetCDF ---
    ds_out = xr.Dataset(
        {
            "F1_q_front": xr.DataArray(f1, coords={"time": time_w}, dims=("time",),
                            attrs={"desc": "低层q前端零交叉 (正->负, 0~180°, 5°容忍, 1000-850hPa层平均)"}),
            "F2_omega_sub_front": xr.DataArray(f2, coords={"time": time_w}, dims=("time",),
                            attrs={"desc": "低层omega下沉前端 (负->正, 5°容忍, 1000-850hPa层平均)"}),
            "F3_u_conv_front": xr.DataArray(f3, coords={"time": time_w}, dims=("time",),
                            attrs={"desc": "低层u辐合前端 (正->负, -90~180°, 1000-850hPa层平均)"}),
            "F4_q_grad_max": xr.DataArray(f4, coords={"time": time_w}, dims=("time",),
                            attrs={"desc": "低层q梯度极值位置 (dq/dx最大负值, 1000-850hPa层平均)"}),
            "F5_T_front": xr.DataArray(f5, coords={"time": time_w}, dims=("time",),
                            attrs={"desc": "低层T前端零交叉 (正->负, 1000-850hPa层平均)"}),
            "F6_omega_low_east": xr.DataArray(f6, coords={"time": time_w}, dims=("time",),
                            attrs={"desc": "低层omega上升区东边界 (5°容忍, 1000-850hPa层平均)"}),
            "F7_u_shear_change": xr.DataArray(f7, coords={"time": time_w}, dims=("time",),
                            attrs={"desc": "u风垂直切变转换 (u_upper-u_lower 第一个符号变化, 400-200/1000-850hPa层平均)"}),
        },
        attrs={
            "description": "7种低层前端指标逐日值",
            "low_layer": f"{LOW_LAYER[0]}-{LOW_LAYER[1]} hPa",
            "up_layer": f"{UP_LAYER[0]}-{UP_LAYER[1]} hPa",
            "smooth_window": str(SMOOTH_WINDOW),
            "tolerance_deg": str(TOLERANCE_DEG),
        }
    )

    enc = {v: {"zlib": True, "complevel": 4} for v in ds_out.data_vars}
    enc["time"] = {"zlib": False, "_FillValue": None}
    ds_out.to_netcdf(OUT_NC, engine="netcdf4", encoding=enc)
    print(f"\nSaved daily: {OUT_NC}")

    # --- 汇总逐事件均值 ---
    indicator_names = ["F1_q_front", "F2_omega_sub_front", "F3_u_conv_front",
                       "F4_q_grad_max", "F5_T_front", "F6_omega_low_east",
                       "F7_u_shear_change"]

    rows = []
    for _, ev_row in events.iterrows():
        eid = int(ev_row["event_id"])
        ts = np.datetime64(ev_row["start_date"])
        te = np.datetime64(ev_row["end_date"])
        m = (time_w >= ts) & (time_w <= te)

        row_dict = {"event_id": eid}
        for name in indicator_names:
            vals = ds_out[name].values[m].astype(float)
            row_dict[name] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan

        sp = df_speed[df_speed["event_id"] == eid]
        row_dict["phase_speed_m_s"] = float(sp["phase_speed_m_s"].values[0]) if len(sp) > 0 else np.nan
        rows.append(row_dict)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved event mean: {OUT_CSV}")

    # --- 统计摘要 ---
    print(f"\n{'='*60}")
    print("SUMMARY:")
    for name in indicator_names:
        vals = ds_out[name].values.astype(float)
        ok = np.isfinite(vals)
        n_ok = int(ok.sum())
        if n_ok > 0:
            print(f"  {name}: valid={n_ok}/{n_valid} ({100*n_ok/max(n_valid,1):.1f}%), "
                  f"mean={np.nanmean(vals):.1f}, "
                  f"median={np.nanmedian(vals[ok]):.1f}, "
                  f"std={np.nanstd(vals[ok]):.1f}, "
                  f"range=[{np.nanmin(vals[ok]):.1f}, {np.nanmax(vals[ok]):.1f}]")
        else:
            print(f"  {name}: NO valid values!")

    # 事件均值与相速度的相关
    print(f"\nCorrelations with phase speed (event mean):")
    from scipy import stats
    speed = df_out["phase_speed_m_s"].values.astype(float)
    for name in indicator_names:
        ind = df_out[name].values.astype(float)
        ok = np.isfinite(ind) & np.isfinite(speed)
        if ok.sum() > 5:
            r, p = stats.pearsonr(ind[ok], speed[ok])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {name}: r={r:.3f}, p={p:.4f} {sig} (N={ok.sum()})")
        else:
            print(f"  {name}: insufficient data")

    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    main()
