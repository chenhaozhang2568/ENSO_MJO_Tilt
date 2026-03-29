# -*- coding: utf-8 -*-
"""
compare_phase_speed_methods.py: 六种MJO相速度计算方法统一计算

方法：
  M1: 逐日差分法 — 相邻两天OLR中心经度差/时间，事件内取平均
  M2: 逐经度差分法 — 逐经度OLR最小值日，相邻经度中心差分取平均
  M3: 逐日中心线性拟合 — 逐日OLR最小值经度做 lon=a*t+b 拟合
  M4: 逐经度中心线性拟合 — 逐经度OLR最小值日做 lon=a*t+b 拟合
  M5: 逐日50%范围拟合 — 逐日OLR中心向东西找50%强度边界，全部点做拟合
  M6: 逐经度50%范围拟合 — 逐经度OLR中心向前后找50%强度边界，全部点做拟合

输出：
  - CSV: E:/Datas/Derived/phase_speed_6methods.csv
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy import stats

# ======================
# PATHS
# ======================
STEP3_NC   = Path(r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc")
EVENTS_CSV = Path(r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv")
OUT_CSV    = Path(r"E:\Datas\Derived\phase_speed_6methods.csv")

# ======================
# SETTINGS
# ======================
LON_RANGE = (20, 220)
HALF_MAX_FRAC = 0.5       # 50% 强度阈值
MIN_POINTS_FIT = 10       # 拟合最少点数
MIN_VALID_DAYS = 5        # 最少有效数据点

DEG_TO_M = 111320.0
DAY_TO_SEC = 86400.0


def _to_lon360(ds):
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon.values + 360) % 360).sortby("lon")
    return ds


def _deg_day_to_m_s(slope_deg_day):
    """经度度/天 -> m/s"""
    return slope_deg_day * DEG_TO_M / DAY_TO_SEC


def _linear_fit(t_arr, lon_arr):
    """线性拟合 lon = slope*t + intercept，返回 dict 或 None"""
    valid = np.isfinite(t_arr) & np.isfinite(lon_arr)
    if valid.sum() < MIN_VALID_DAYS:
        return None
    t_v = t_arr[valid].astype(float)
    l_v = lon_arr[valid].astype(float)
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(t_v, l_v)
    except Exception:
        return None
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r_value ** 2,
        "speed_m_s": _deg_day_to_m_s(slope),
        "n_points": int(valid.sum()),
    }


def _get_lon_centers(olr_event, lon_arr, day_indices):
    """逐经度找OLR最小值日，返回 center_lon[], center_t[] 数组"""
    n_days, n_lon = olr_event.shape
    center_t = []
    center_lon = []
    for j in range(n_lon):
        olr_col = olr_event[:, j]
        if np.all(~np.isfinite(olr_col)) or np.nanmin(olr_col) >= 0:
            continue
        valid = np.isfinite(olr_col)
        if valid.sum() < 3:
            continue
        min_idx = int(np.nanargmin(olr_col))
        center_t.append(float(day_indices[min_idx]))
        center_lon.append(float(lon_arr[j]))
    return np.array(center_lon), np.array(center_t)


# =====================================================================
# METHOD 1: 逐日差分法
# =====================================================================
def method1_daily_diff(center_lons, day_indices):
    """M1: 相邻两天OLR中心经度差 / 时间差，事件内取平均。"""
    valid = np.isfinite(center_lons)
    if valid.sum() < 2:
        return {"speed_m_s": np.nan, "n_points": 0}

    lons_v = center_lons[valid]
    days_v = day_indices[valid]

    dlon = np.diff(lons_v)
    dt = np.diff(days_v)
    dt[dt == 0] = np.nan

    speeds_deg_day = dlon / dt
    reasonable = np.abs(speeds_deg_day) < 30.0
    speeds_filtered = speeds_deg_day[reasonable & np.isfinite(speeds_deg_day)]

    if len(speeds_filtered) == 0:
        return {"speed_m_s": np.nan, "n_points": 0}

    mean_speed = float(np.mean(speeds_filtered))
    return {
        "speed_m_s": _deg_day_to_m_s(mean_speed),
        "n_points": len(speeds_filtered),
    }


# =====================================================================
# METHOD 2: 逐经度差分法
# =====================================================================
def method2_lon_diff(olr_event, lon_arr, day_indices):
    """M2: 逐经度找OLR最小值日，相邻经度中心差分取平均。"""
    center_lon, center_t = _get_lon_centers(olr_event, lon_arr, day_indices)

    if len(center_lon) < 2:
        return {"speed_m_s": np.nan, "n_points": 0,
                "center_lon": center_lon, "center_t": center_t}

    dlon = np.diff(center_lon)
    dt = np.diff(center_t)
    dt[dt == 0] = np.nan

    speeds_deg_day = dlon / dt
    reasonable = np.abs(speeds_deg_day) < 30.0
    speeds_filtered = speeds_deg_day[reasonable & np.isfinite(speeds_deg_day)]

    if len(speeds_filtered) == 0:
        return {"speed_m_s": np.nan, "n_points": 0,
                "center_lon": center_lon, "center_t": center_t}

    mean_speed = float(np.mean(speeds_filtered))
    return {
        "speed_m_s": _deg_day_to_m_s(mean_speed),
        "n_points": len(speeds_filtered),
        "center_lon": center_lon,
        "center_t": center_t,
    }


# =====================================================================
# METHOD 3: 逐日中心线性拟合 (原M2)
# =====================================================================
def method3_daily_center_lsq(center_lons, day_indices):
    """M3: 逐日OLR最小值中心经度，线性拟合 lon = a*t + b。"""
    result = _linear_fit(day_indices, center_lons)
    if result is None:
        return {"speed_m_s": np.nan, "r2": np.nan, "n_points": 0}
    return result


# =====================================================================
# METHOD 4: 逐经度中心线性拟合 (原M3)
# =====================================================================
def method4_lon_center_lsq(olr_event, lon_arr, day_indices):
    """M4: 逐经度找OLR最小值日，得到 (lon, t_min) 点集，线性拟合。"""
    center_lon, center_t = _get_lon_centers(olr_event, lon_arr, day_indices)

    if len(center_t) < MIN_VALID_DAYS:
        return {"speed_m_s": np.nan, "r2": np.nan, "n_points": 0}

    result = _linear_fit(center_t, center_lon)
    if result is None:
        return {"speed_m_s": np.nan, "r2": np.nan, "n_points": 0}
    return result


# =====================================================================
# METHOD 5: 逐日50%范围拟合 (原M4)
# =====================================================================
def method5_daily_halfmax_lsq(olr_event, lon_arr, day_indices):
    """M5: 逐日找OLR中心，向东西两侧找强度为中心一半的点，
    所有日期的边界内所有点做线性拟合。"""
    n_days, n_lon = olr_event.shape
    active_points = []

    for t in range(n_days):
        olr_row = olr_event[t, :]
        if np.all(~np.isfinite(olr_row)) or np.nanmin(olr_row) >= 0:
            continue
        valid = np.isfinite(olr_row)
        if valid.sum() < 3:
            continue
        min_idx = int(np.nanargmin(olr_row))
        olr_min = float(olr_row[min_idx])
        if olr_min >= 0:
            continue
        threshold = HALF_MAX_FRAC * olr_min

        j_start = min_idx
        for j in range(min_idx - 1, -1, -1):
            if not np.isfinite(olr_row[j]) or olr_row[j] > threshold:
                break
            j_start = j
        j_end = min_idx
        for j in range(min_idx + 1, n_lon):
            if not np.isfinite(olr_row[j]) or olr_row[j] > threshold:
                break
            j_end = j

        for j in range(j_start, j_end + 1):
            active_points.append((float(lon_arr[j]), float(day_indices[t])))

    if len(active_points) < MIN_POINTS_FIT:
        return {"speed_m_s": np.nan, "r2": np.nan, "n_points": 0,
                "active_points": active_points}
    pts = np.array(active_points)
    result = _linear_fit(pts[:, 1], pts[:, 0])
    if result is None:
        return {"speed_m_s": np.nan, "r2": np.nan, "n_points": 0,
                "active_points": active_points}
    result["active_points"] = active_points
    return result


# =====================================================================
# METHOD 6: 逐经度50%范围拟合 (原M5)
# =====================================================================
def method6_lon_halfmax_lsq(olr_event, lon_arr, day_indices):
    """M6: 逐经度找OLR中心日，向前后找强度为中心一半的日期范围，
    所有点做线性拟合。"""
    n_days, n_lon = olr_event.shape
    center_points = []
    active_points = []

    for j in range(n_lon):
        olr_col = olr_event[:, j]
        if np.all(~np.isfinite(olr_col)) or np.nanmin(olr_col) >= 0:
            continue
        valid = np.isfinite(olr_col)
        if valid.sum() < 3:
            continue
        min_idx = int(np.nanargmin(olr_col))
        olr_min = float(olr_col[min_idx])
        if olr_min >= 0:
            continue
        center_points.append((float(lon_arr[j]), float(day_indices[min_idx])))
        threshold = HALF_MAX_FRAC * olr_min

        t_start = min_idx
        for t in range(min_idx - 1, -1, -1):
            if not np.isfinite(olr_col[t]) or olr_col[t] > threshold:
                break
            t_start = t
        t_end = min_idx
        for t in range(min_idx + 1, n_days):
            if not np.isfinite(olr_col[t]) or olr_col[t] > threshold:
                break
            t_end = t

        for t in range(t_start, t_end + 1):
            active_points.append((float(lon_arr[j]), float(day_indices[t])))

    if len(active_points) < MIN_POINTS_FIT:
        return {"speed_m_s": np.nan, "r2": np.nan, "n_points": 0,
                "center_points": center_points, "active_points": active_points}
    pts = np.array(active_points)
    result = _linear_fit(pts[:, 1], pts[:, 0])
    if result is None:
        return {"speed_m_s": np.nan, "r2": np.nan, "n_points": 0,
                "center_points": center_points, "active_points": active_points}
    result["center_points"] = center_points
    result["active_points"] = active_points
    return result


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 60)
    print("compare_phase_speed_methods.py")
    print("六种MJO相速度计算方法统一计算")
    print("=" * 60)

    ds = xr.open_dataset(STEP3_NC)
    ds = _to_lon360(ds)
    df_events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    print(f"  Events: {len(df_events)}")

    olr_full = ds["olr_recon"]
    center_track = ds["center_lon_track"].values.astype(float)
    center_track = (center_track + 360) % 360
    time_index = pd.to_datetime(ds["time"].values)

    results = []

    for idx, row in df_events.iterrows():
        eid = int(row["event_id"])
        t0 = pd.Timestamp(row["start_date"])
        t1 = pd.Timestamp(row["end_date"])

        ds_event = olr_full.sel(time=slice(str(t0), str(t1)))
        if ds_event.sizes["time"] < 3:
            print(f"  Event {eid}: too short, skip")
            results.append({
                "event_id": eid, "start_date": row["start_date"],
                "end_date": row["end_date"],
                "duration_days": row["duration_days"],
                "speed_m1": np.nan, "speed_m2": np.nan,
                "speed_m3": np.nan, "speed_m4": np.nan,
                "speed_m5": np.nan, "speed_m6": np.nan,
                "r2_m3": np.nan, "r2_m4": np.nan,
                "r2_m5": np.nan, "r2_m6": np.nan,
                "n_pts_m4": 0, "n_pts_m5": 0, "n_pts_m6": 0,
            })
            continue

        lon_start = max(float(row["lon_start"]) - 10, LON_RANGE[0])
        lon_end = min(float(row["lon_end"]) + 10, LON_RANGE[1])
        ds_event_lon = ds_event.sel(lon=slice(lon_start, lon_end))

        olr_arr = ds_event_lon.values
        lon_arr = ds_event_lon.lon.values
        day_indices = np.arange(olr_arr.shape[0], dtype=float)

        time_mask = (time_index >= t0) & (time_index <= t1)
        daily_center = center_track[time_mask]
        daily_days = np.arange(len(daily_center), dtype=float)

        r1 = method1_daily_diff(daily_center, daily_days)
        r2 = method2_lon_diff(olr_arr, lon_arr, day_indices)
        r3 = method3_daily_center_lsq(daily_center, daily_days)
        r4 = method4_lon_center_lsq(olr_arr, lon_arr, day_indices)
        r5 = method5_daily_halfmax_lsq(olr_arr, lon_arr, day_indices)
        r6 = method6_lon_halfmax_lsq(olr_arr, lon_arr, day_indices)

        results.append({
            "event_id": eid,
            "start_date": row["start_date"],
            "end_date": row["end_date"],
            "duration_days": row["duration_days"],
            "speed_m1": r1["speed_m_s"],
            "speed_m2": r2["speed_m_s"],
            "speed_m3": r3.get("speed_m_s", np.nan),
            "speed_m4": r4.get("speed_m_s", np.nan),
            "speed_m5": r5.get("speed_m_s", np.nan),
            "speed_m6": r6.get("speed_m_s", np.nan),
            "r2_m3": r3.get("r2", np.nan),
            "r2_m4": r4.get("r2", np.nan),
            "r2_m5": r5.get("r2", np.nan),
            "r2_m6": r6.get("r2", np.nan),
            "n_pts_m4": r4.get("n_points", 0),
            "n_pts_m5": r5.get("n_points", 0),
            "n_pts_m6": r6.get("n_points", 0),
        })

        print(f"  Event {eid:3d}: "
              f"M1={r1['speed_m_s']:6.2f}  "
              f"M2={r2['speed_m_s']:6.2f}  "
              f"M3={r3.get('speed_m_s', np.nan):6.2f}  "
              f"M4={r4.get('speed_m_s', np.nan):6.2f}  "
              f"M5={r5.get('speed_m_s', np.nan):6.2f}  "
              f"M6={r6.get('speed_m_s', np.nan):6.2f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}")

    print(f"\n{'='*60}")
    print("SUMMARY (m/s)")
    print(f"{'='*60}")
    for col, name in [
        ("speed_m1", "M1 DailyDiff"),
        ("speed_m2", "M2 LonDiff"),
        ("speed_m3", "M3 DailyCenter"),
        ("speed_m4", "M4 LonCenter"),
        ("speed_m5", "M5 DailyHalfMax"),
        ("speed_m6", "M6 LonHalfMax"),
    ]:
        v = df_out[col].dropna()
        if len(v) > 0:
            print(f"  {name:18s}: mean={v.mean():6.2f}  std={v.std():5.2f}  "
                  f"median={v.median():6.2f}  [{v.min():6.2f}, {v.max():6.2f}]  "
                  f"N={len(v)}")
        else:
            print(f"  {name:18s}: no valid data")


if __name__ == "__main__":
    main()
