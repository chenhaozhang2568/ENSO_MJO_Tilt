"""
为西边界最偏西的5个事件绘制每日垂直剖面图（所有天，非随机采样）
参考 diagnose_three_groups.py 的 plot_daily_omega_profiles 格式
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d
from pathlib import Path

mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

EVENTS_CSV = r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv"
TILT_Q_NC  = r"E:\Datas\Derived\tilt_q_daily_1979-2022.nc"
STEP3_NC   = r"E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc"
W_NORM_NC  = r"E:\Datas\Derived\era5_mjo_recon_w_norm_1979-2022.nc"

FIG_DIR = Path(r"E:\Projects\ENSO_MJO_Tilt\outputs\figures\upper_west_diagnose\top5_westward_profiles")

LEVEL_TO_HEIGHT = {
    1000: 0.1, 925: 0.75, 850: 1.5, 700: 3.0,
    600: 4.2, 500: 5.5, 400: 7.2, 300: 9.2, 200: 12.0
}
SMOOTH_WINDOW = 10

# 目标事件
TARGET_EVENTS = [13, 115, 50, 40, 114]


def _smooth_1d(profile, window):
    if window <= 1:
        return profile
    kernel = np.ones(window) / window
    valid = np.isfinite(profile).astype(float)
    filled = np.where(np.isfinite(profile), profile, 0.0)
    smoothed = np.convolve(filled, kernel, mode='same')
    count = np.convolve(valid, kernel, mode='same')
    count[count < 1e-10] = np.nan
    return smoothed / count


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(EVENTS_CSV, parse_dates=["start_date", "end_date"])
    ds_w = xr.open_dataset(W_NORM_NC)
    ds3 = xr.open_dataset(STEP3_NC)
    ds_tilt = xr.open_dataset(TILT_Q_NC)

    w_all = ds_w["w_mjo_recon_norm"].values
    levels = ds_w["pressure_level"].values if "pressure_level" in ds_w else ds_w["level"].values
    lon = ds_w["lon"].values
    lon_360 = np.where(lon < 0, lon + 360, lon)
    sort_lon = np.argsort(lon_360)
    lon_360 = lon_360[sort_lon]

    time_w = pd.to_datetime(ds_w["time"].values)
    center_lon = ds3["center_lon_track"].values.astype(float)
    time_s3 = pd.to_datetime(ds3["time"].values)

    uw_daily = ds_tilt["up_west_rel"].values.astype(float)
    ue_daily = ds_tilt["up_east_rel"].values.astype(float)
    qmax_daily = ds_tilt["q_max_rel"].values.astype(float)
    time_tq = pd.to_datetime(ds_tilt["time"].values)

    heights = np.array([LEVEL_TO_HEIGHT[int(p)] for p in levels])

    total_plots = 0

    for eid in TARGET_EVENTS:
        ev = events[events["event_id"] == eid].iloc[0]
        out_dir = FIG_DIR / f"event_{eid:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = pd.Timestamp(ev["start_date"])
        te = pd.Timestamp(ev["end_date"])

        # 找事件内所有天
        event_dates = pd.date_range(ts, te, freq='D')
        print(f"\nEvent #{eid}: {ts.date()} ~ {te.date()} ({len(event_dates)}d)")

        for day_num, date in enumerate(event_dates):
            date64 = np.datetime64(date)

            w_idx = np.where(time_w == date64)[0]
            s3_idx = np.where(time_s3 == date64)[0]
            tq_idx = np.where(time_tq == date64)[0]
            if len(w_idx) == 0 or len(s3_idx) == 0 or len(tq_idx) == 0:
                print(f"  Day {day_num+1} ({date.date()}): missing data, skip")
                continue
            w_idx, s3_idx, tq_idx = w_idx[0], s3_idx[0], tq_idx[0]

            c = center_lon[s3_idx]
            if not np.isfinite(c):
                print(f"  Day {day_num+1} ({date.date()}): no center, skip")
                continue

            uw = uw_daily[tq_idx]
            ue = ue_daily[tq_idx]
            qm = qmax_daily[tq_idx]

            # 相对经度
            rel_lon = lon_360 - c
            mask_lon = (rel_lon >= -90) & (rel_lon <= 90)
            rel_lons = rel_lon[mask_lon]

            w_day = w_all[w_idx, :, :][:, sort_lon][:, mask_lon]

            # 平滑
            w_sm = np.full_like(w_day, np.nan)
            for k in range(len(levels)):
                w_sm[k, :] = _smooth_1d(w_day[k, :], SMOOTH_WINDOW)

            # 插值到高度坐标
            target_h = np.linspace(0.0, 12.0, 120)
            w_interp = np.full((len(target_h), len(rel_lons)), np.nan)
            for j in range(len(rel_lons)):
                col = w_sm[:, j]
                valid = np.isfinite(col)
                if valid.sum() >= 2:
                    f = interp1d(heights[valid], col[valid], kind="linear",
                                 bounds_error=False, fill_value=np.nan)
                    w_interp[:, j] = f(target_h)

            # --- 绘图 ---
            fig, ax = plt.subplots(figsize=(14, 7))
            X, Y = np.meshgrid(rel_lons, target_h)

            vmax = np.nanmax(np.abs(w_interp)) * 0.8
            if vmax < 1e-6 or not np.isfinite(vmax):
                vmax = 0.01
            norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            cf = ax.contourf(X, Y, w_interp, levels=np.linspace(-vmax, vmax, 21),
                             cmap="RdBu_r", norm=norm, extend="both")
            ax.contour(X, Y, w_interp, levels=[0], colors="black", linewidths=2.0)

            # 标注 up_west, up_east, q_max 点
            up_h_mid = (LEVEL_TO_HEIGHT[400] + LEVEL_TO_HEIGHT[200]) / 2.0
            low_h_mid = (LEVEL_TO_HEIGHT[1000] + LEVEL_TO_HEIGHT[850]) / 2.0

            if np.isfinite(uw) and np.isfinite(qm):
                tilt_val = qm - uw
                ax.plot([uw, qm], [up_h_mid, low_h_mid], "o-", color="gold",
                        markersize=12, markeredgecolor="black", markeredgewidth=1.5,
                        lw=3, zorder=10, label=f"Tilt_q = {tilt_val:.1f}\u00b0")
                ax.annotate(f"Upper W: {uw:.1f}\u00b0", (uw, up_h_mid),
                            textcoords="offset points", xytext=(10, 10),
                            fontsize=9, color="darkgoldenrod", fontweight="bold")
                ax.annotate(f"q_max: {qm:.1f}\u00b0", (qm, low_h_mid),
                            textcoords="offset points", xytext=(10, -15),
                            fontsize=9, color="darkgoldenrod", fontweight="bold")

            if np.isfinite(ue):
                ax.plot(ue, up_h_mid, "s", color="cyan", markersize=10,
                        markeredgecolor="black", zorder=10, label=f"Upper E: {ue:.1f}\u00b0")

            ax.axvline(0, color="limegreen", lw=2.5, alpha=0.8, label="Conv. Center")

            ax.set_ylim(0, 12)
            ax.set_xlim(-90, 90)
            ax.set_ylabel("Height (km)", fontsize=12)
            ax.set_xlabel("Relative Longitude (\u00b0)", fontsize=12)

            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            pticks = [1000, 925, 850, 700, 600, 500, 400, 300, 200]
            ax2.set_yticks([LEVEL_TO_HEIGHT[p] for p in pticks])
            ax2.set_yticklabels([str(p) for p in pticks])
            ax2.set_ylabel("Pressure (hPa)", fontsize=12)

            cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.12, shrink=0.8)
            cbar.set_label("\u03c9 (normalized)", fontsize=10)

            title = (f"Event #{eid} \u2014 Day {day_num+1}/{len(event_dates)} "
                     f"({date.strftime('%Y-%m-%d')})\n"
                     f"Center: {c:.1f}\u00b0E, Up-West: {uw:.1f}\u00b0")
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.legend(loc="upper right", fontsize=9)

            out = out_dir / f"day{day_num+1:02d}_{date.strftime('%Y-%m-%d')}.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
            total_plots += 1

        print(f"  Saved {len(list(out_dir.glob('*.png')))} profiles to {out_dir}")

    print(f"\nTotal: {total_plots} profiles saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
