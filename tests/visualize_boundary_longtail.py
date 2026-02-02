# -*- coding: utf-8 -*-
"""
visualize_boundary_longtail.py: MJO 边界长尾效应可视化

================================================================================
功能描述：
    本脚本可视化 MJO 垂直倾斜边界（上层/下层西边界/东边界）的分布特征，
    特别关注长尾效应（极值天）的存在和影响。

主要输出：
    1. 边界变量的直方图密度分布（对数y轴）
    2. ECDF 累积分布图（关注1-5%和95-99%分位数）
    3. 高层 vs 低层边界散点图
    4. 极端尾部天的时间序列点图
    5. 极值日期列表 CSV 输出

科学用途：
    诊断为何 Tilt 指数会出现极大值，识别异常日期进行个案分析。
  (long-tail effect), for both lower and upper layers.

Expected inputs (NetCDF from your Step4 script):
- time coordinate
- boundary variables (relative to convective center, in degrees):
    low_west_rel, low_east_rel, up_west_rel, up_east_rel
  (or similar names; this script will auto-detect common variants)
- an event-day mask variable:
    eventmask / event_mask (1 for event days)

Outputs:
- 1) Histograms (log-y) to highlight long tails
- 2) ECDF plots (tail visibility near 0–5% and 95–100%)
- 3) Scatter: lower vs upper boundary (outliers pop out)
- 4) Optional: per-event-day “extreme tail points” table saved as CSV

Run:
python plot_boundary_longtail_eventdays.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, MaxNLocator
# -----------------------
# USER SETTINGS
# -----------------------
# Step4 output netcdf path (change to yours)
STEP4_NC = r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc"
OUT_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\boundary_longtail_step4_1979-2022")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tail thresholds for “extremes”
TAIL_P_LO = 1.0    # 1st percentile
TAIL_P_HI = 99.0   # 99th percentile

# Histogram bins (relative-lon degrees)
BINS = np.arange(-120, 121, 2)  # adjust if your rel_lon span differs

# -----------------------
# HELPERS
# -----------------------
def _pick_var(ds: xr.Dataset, candidates: list[str], required: bool = True) -> str | None:
    for c in candidates:
        if c in ds.variables:
            return c
    if required:
        raise KeyError(f"None of these variables found: {candidates}\nAvailable: {list(ds.variables)}")
    return None

def _to_1d_series(ds: xr.Dataset, varname: str) -> pd.Series:
    """Return a 1D pandas Series indexed by time."""
    da = ds[varname]
    if "time" not in da.dims:
        raise ValueError(f"{varname} has no 'time' dim; dims={da.dims}")
    # squeeze any singleton dims
    da = da.squeeze(drop=True)
    # if still has extra dims (unlikely for boundaries), reduce safely
    extra_dims = [d for d in da.dims if d != "time"]
    if extra_dims:
        da = da.mean(extra_dims, skipna=True)
    return pd.Series(da.values, index=pd.to_datetime(ds["time"].values), name=varname)

def _event_mask(ds: xr.Dataset) -> pd.Series:
    mvar = _pick_var(ds, ["eventmask", "event_mask", "event_day_mask", "event_days_mask"])
    m = ds[mvar].squeeze(drop=True)
    # allow 0/1 or bool
    if m.dtype != bool:
        m = m.astype(float)
        m = m.where(np.isfinite(m), 0.0)
        m = m > 0.5
    return pd.Series(m.values, index=pd.to_datetime(ds["time"].values), name="eventmask")

def _maybe_winter_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """Nov–Apr mask (if you want it). Not enforced by default."""
    mon = index.month
    return (mon >= 11) | (mon <= 4)

def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """1D Gaussian kernel, normalized."""
    size = int(size)
    if size % 2 == 0:
        size += 1
    half = size // 2
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    return k

def plot_hist_logy(series_dict: dict[str, pd.Series], out_png: Path):
    """
    Histogram density -> interpolate to a fine x grid -> plot smooth-looking continuous curve
    Requirements:
      - empty bins stay 0
      - linear y-axis with uniform tick spacing
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    global BINS  # 你原脚本里 BINS = np.arange(...)
    edges = np.asarray(BINS, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 加密经度网格，让曲线看起来连续
    x_fine = np.linspace(centers.min(), centers.max(), 1200)

    y_max = 0.0
    for name, s in series_dict.items():
        x = s.dropna().values
        if x.size == 0:
            continue

        density, _ = np.histogram(x, bins=edges, density=True)
        density = density.astype(float)  # 这里 0 就是 0，不改成 NaN、不做 floor

        # 插值到细网格：空 bin=0 会自然连到 0，不会断线
        y_fine = np.interp(x_fine, centers, density)

        ax.plot(x_fine, y_fine, linewidth=2, label=name)
        y_max = max(y_max, float(np.max(y_fine)))

    # 线性纵轴（均匀刻度）
    ax.set_yscale("linear")
    ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)

    # 纵轴均匀刻度：两种方式二选一
    # 方式1：自动给“均匀”刻度（推荐，省事）
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # 方式2：强制固定步长（你想完全可控就用这个）
    # step = (ax.get_ylim()[1]) / 5
    # ax.yaxis.set_major_locator(MultipleLocator(step))

    ax.set_xlabel("Boundary position relative to convective center (deg)")
    ax.set_ylabel("Probability density")
    ax.set_title("Event days only: boundary distributions (linear y, zeros kept)")
    ax.axvline(0, linewidth=1)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_ecdf(series_dict: dict[str, pd.Series], out_png: Path):
    """ECDF plots: tails visible near 0–5% and 95–100%."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, s in series_dict.items():
        x = np.sort(s.dropna().values)
        if x.size == 0:
            continue
        y = np.arange(1, x.size + 1) / x.size
        ax.plot(x, y, linewidth=2, label=name)
    ax.set_xlabel("Boundary position relative to convective center (deg)")
    ax.set_ylabel("ECDF")
    ax.set_title("Event days only: ECDF (inspect tails near 0–5% and 95–100%)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def plot_scatter_low_vs_up(low_s: pd.Series, up_s: pd.Series, out_png: Path, label: str):
    """Scatter to show outlier days: low vs up boundary."""
    df = pd.concat([low_s, up_s], axis=1).dropna()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(df.iloc[:, 0].values, df.iloc[:, 1].values, s=10)
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_title(f"Event days only: {label} (outliers = long-tail days)")
    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

def save_tail_days(series_dict: dict[str, pd.Series], out_csv: Path):
    """Save extreme tail days (union across variables) for quick inspection."""
    # compute thresholds per variable
    records = []
    tail_index_union = set()
    for name, s in series_dict.items():
        x = s.dropna().values
        if x.size < 20:
            continue
        lo = np.nanpercentile(x, TAIL_P_LO)
        hi = np.nanpercentile(x, TAIL_P_HI)
        idx_lo = s[s <= lo].index
        idx_hi = s[s >= hi].index
        for t in idx_lo:
            tail_index_union.add(t)
        for t in idx_hi:
            tail_index_union.add(t)
        records.append((name, lo, hi, len(idx_lo), len(idx_hi)))

    # thresholds summary
    thr_df = pd.DataFrame(records, columns=["var", f"p{TAIL_P_LO}", f"p{TAIL_P_HI}", "n_low_tail", "n_high_tail"])
    thr_df.to_csv(out_csv.with_name(out_csv.stem + "_thresholds.csv"), index=False)

    # value table for all tail days
    tail_times = sorted(tail_index_union)
    val_df = pd.DataFrame(index=pd.to_datetime(tail_times))
    for name, s in series_dict.items():
        val_df[name] = s.reindex(val_df.index)
    val_df.index.name = "time"
    val_df.to_csv(out_csv, index=True)

def plot_tail_time_series(series_dict: dict[str, pd.Series], out_png: Path):
    """
    Optional: plot only extreme-tail points in time to visualize sporadic long-tail days.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, s in series_dict.items():
        x = s.dropna().values
        if x.size < 20:
            continue
        lo = np.nanpercentile(x, TAIL_P_LO)
        hi = np.nanpercentile(x, TAIL_P_HI)
        tail = s[(s <= lo) | (s >= hi)]
        ax.scatter(tail.index, tail.values, s=12, label=f"{name} tails (<=p{TAIL_P_LO} or >=p{TAIL_P_HI})")
    ax.set_ylabel("Boundary (deg)")
    ax.set_title("Event days only: extreme tail points over time")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

# -----------------------
# MAIN
# -----------------------
def main():
    ds = xr.open_dataset(STEP4_NC)

    # auto-detect boundary variable names (add your names if different)
    v_low_w = _pick_var(ds, ["low_west_rel", "low_west", "low_west_lonrel", "low_west_boundary_rel"])
    v_low_e = _pick_var(ds, ["low_east_rel", "low_east", "low_east_lonrel", "low_east_boundary_rel"])
    v_up_w  = _pick_var(ds, ["up_west_rel", "upper_west_rel", "up_west", "upper_west", "up_west_lonrel"])
    v_up_e  = _pick_var(ds, ["up_east_rel", "upper_east_rel", "up_east", "upper_east", "up_east_lonrel"])

    eventmask = _event_mask(ds)

    s_low_w = _to_1d_series(ds, v_low_w)
    s_low_e = _to_1d_series(ds, v_low_e)
    s_up_w  = _to_1d_series(ds, v_up_w)
    s_up_e  = _to_1d_series(ds, v_up_e)

    # ---- event days only ----
    mask = eventmask.astype(bool)
    s_low_w = s_low_w[mask]
    s_low_e = s_low_e[mask]
    s_up_w  = s_up_w[mask]
    s_up_e  = s_up_e[mask]

    # If you ALSO want winter-only, uncomment:
    # winter = _maybe_winter_mask(s_low_w.index)
    # s_low_w, s_low_e, s_up_w, s_up_e = s_low_w[winter], s_low_e[winter], s_up_w[winter], s_up_e[winter]

    series_dict = {
        "LOW west boundary": s_low_w,
        "LOW east boundary": s_low_e,
        "UP  west boundary": s_up_w,
        "UP  east boundary": s_up_e,
    }

    plot_hist_logy(series_dict, OUT_DIR / "boundary_hist_linear_eventdays_step4_1979-2022.png")
    plot_ecdf(series_dict, OUT_DIR / "boundary_ecdf_eventdays_step4_1979-2022.png")
    plot_scatter_low_vs_up(s_low_w, s_up_w, OUT_DIR / "boundary_scatter_low_vs_up_west_eventdays_step4_1979-2022.png", "WEST boundary")
    plot_scatter_low_vs_up(s_low_e, s_up_e, OUT_DIR / "boundary_scatter_low_vs_up_east_eventdays_step4_1979-2022.png", "EAST boundary")
    save_tail_days(series_dict, OUT_DIR / "boundary_tail_days_eventdays_step4_1979-2022.csv")
    plot_tail_time_series(series_dict, OUT_DIR / "boundary_tail_points_time_eventdays_step4_1979-2022.png")

    # Print quick stats (so you can confirm tails numerically)
    def _stats(s: pd.Series):
        x = s.dropna().values
        return {
            "n": x.size,
            "p1": np.nanpercentile(x, 1) if x.size else np.nan,
            "p5": np.nanpercentile(x, 5) if x.size else np.nan,
            "p50": np.nanpercentile(x, 50) if x.size else np.nan,
            "p95": np.nanpercentile(x, 95) if x.size else np.nan,
            "p99": np.nanpercentile(x, 99) if x.size else np.nan,
            "min": np.nanmin(x) if x.size else np.nan,
            "max": np.nanmax(x) if x.size else np.nan,
        }

    print("Event-day boundary stats (deg, relative to center):")
    for k, s in series_dict.items():
        st = _stats(s)
        print(f"- {k}: n={st['n']}, min={st['min']:.1f}, p1={st['p1']:.1f}, p5={st['p5']:.1f}, "
              f"p50={st['p50']:.1f}, p95={st['p95']:.1f}, p99={st['p99']:.1f}, max={st['max']:.1f}")

    print(f"Saved figures + CSV to: {OUT_DIR}")

if __name__ == "__main__":
    main()
