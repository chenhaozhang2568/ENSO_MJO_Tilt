import pandas as pd
import numpy as np
import xarray as xr

events = pd.read_csv(r"E:\Datas\Derived\mjo_events_step3_1979-2022.csv")
ds_tilt = xr.open_dataset(r"E:\Datas\Derived\tilt_daily_step4_layermean_1979-2022.nc")
tilt = ds_tilt['tilt'].values
time = pd.to_datetime(ds_tilt['time'].values)

event_tilts = []
for _, ev in events.iterrows():
    start = pd.Timestamp(ev['start_date'])
    end = pd.Timestamp(ev['end_date'])
    mask = (time >= start) & (time <= end)
    tv = tilt[mask]
    valid = np.isfinite(tv)
    if valid.sum() > 0:
        event_tilts.append({'event_id': ev['event_id'], 'mean_tilt': np.nanmean(tv[valid])})

df = pd.DataFrame(event_tilts)
m = df['mean_tilt'].mean()
s = df['mean_tilt'].std()

stg = df[df['mean_tilt'] > m + 0.5 * s]
wtg = df[df['mean_tilt'] < m - 0.5 * s]

print(f"Mean tilt: {m:.2f}, Std: {s:.2f}")
print(f"STG threshold: > {m + 0.5 * s:.2f}")
print(f"WTG threshold: < {m - 0.5 * s:.2f}")
print(f"STG events: {len(stg)}, mean tilt: {stg['mean_tilt'].mean():.2f}")
print(f"WTG events: {len(wtg)}, mean tilt: {wtg['mean_tilt'].mean():.2f}")
