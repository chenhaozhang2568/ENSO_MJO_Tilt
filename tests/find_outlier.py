# -*- coding: utf-8 -*-
"""对比 STG vs WTG 的事件特征"""
import numpy as np, pandas as pd, xarray as xr
from scipy import stats
from scipy.signal import savgol_filter

ds3 = xr.open_dataset(r'E:\Datas\Derived\mjo_mvEOF_step3_1979-2022.nc')
center = ds3['center_lon_track'].values
amp = ds3['amp'].values
time = pd.to_datetime(ds3['time'].values)
events = pd.read_csv(r'E:\Datas\Derived\mjo_events_step3_1979-2022.csv', parse_dates=['start_date','end_date'])
enso = pd.read_csv(r'E:\Datas\Derived\tilt_event_stats_with_enso_1979-2022.csv')

mt, st = enso['mean_tilt'].mean(), enso['mean_tilt'].std()
enso['group'] = 'Normal'
enso.loc[enso['mean_tilt'] > mt + 0.7*st, 'group'] = 'STG'
enso.loc[enso['mean_tilt'] < mt - 0.7*st, 'group'] = 'WTG'

V_THR = 2.5
rows = []
for _, ev in events.iterrows():
    eid = ev['event_id']
    mask = (time >= ev['start_date']) & (time <= ev['end_date'])
    lons = center[mask]
    amps = amp[mask]
    valid = np.isfinite(lons)
    if valid.sum() < 5:
        continue
    
    lons_filled = lons.copy().astype(float)
    nans = ~valid
    if nans.any() and valid.any():
        lons_filled[nans] = np.interp(np.where(nans)[0], np.where(valid)[0], lons[valid])
    win = min(5, len(lons_filled))
    if win % 2 == 0: win -= 1
    if win >= 3:
        lons_smooth = savgol_filter(lons_filled, win, min(2, win-1))
    else:
        lons_smooth = lons_filled
    velocity = np.gradient(lons_smooth)
    active = velocity > V_THR
    active_idx = np.where(active)[0]
    
    # event characteristics
    lon_start = lons[valid][0]
    lon_end = lons[valid][-1]
    total_disp = lon_end - lon_start
    has_jump = bool(np.any(np.abs(np.diff(lons[valid])) > 20))
    n_jump_days = int(np.sum(np.abs(np.diff(lons[valid])) > 20))
    mean_amp = float(np.nanmean(amps))
    
    if len(active_idx) >= 3:
        slope, *_ = stats.linregress(active_idx.astype(float), lons_filled[active_idx])
        ps = slope * 111e3 / 86400
    else:
        ps = np.nan
    
    rows.append({
        'event_id': eid, 'ps': ps, 'duration': int(valid.sum()),
        'lon_start': lon_start, 'lon_end': lon_end, 'total_disp': total_disp,
        'has_jump': has_jump, 'n_jump_days': n_jump_days,
        'active_days': int(active.sum()), 'active_frac': active.sum()/len(lons),
        'mean_amp': mean_amp,
    })

df = pd.DataFrame(rows).merge(enso[['event_id','group','mean_tilt','enso_phase']], on='event_id')

print('=== STG vs WTG 事件特征对比 ===\n')
for g in ['STG', 'WTG']:
    sub = df[df['group']==g]
    print('%s (N=%d):' % (g, len(sub)))
    print('  phase speed:    %.2f +/- %.2f m/s' % (sub.ps.mean(), sub.ps.std()))
    print('  duration:       %.1f +/- %.1f days' % (sub.duration.mean(), sub.duration.std()))
    print('  lon_start:      %.1f +/- %.1f' % (sub.lon_start.mean(), sub.lon_start.std()))
    print('  lon_end:        %.1f +/- %.1f' % (sub.lon_end.mean(), sub.lon_end.std()))
    print('  total_disp:     %.1f +/- %.1f deg' % (sub.total_disp.mean(), sub.total_disp.std()))
    print('  has MC jump:    %d/%d (%.0f%%)' % (sub.has_jump.sum(), len(sub), 100*sub.has_jump.mean()))
    print('  active_days:    %.1f +/- %.1f' % (sub.active_days.mean(), sub.active_days.std()))
    print('  active_frac:    %.1f%%' % (100*sub.active_frac.mean()))
    print('  mean_amp:       %.2f +/- %.2f' % (sub.mean_amp.mean(), sub.mean_amp.std()))
    print('  ENSO: %s' % sub.enso_phase.value_counts().to_dict())
    print()

# t-tests
stg = df[df['group']=='STG']
wtg = df[df['group']=='WTG']
print('=== t-test (STG vs WTG) ===')
for col in ['ps','duration','lon_start','lon_end','total_disp','active_days','active_frac','mean_amp']:
    s, w = stg[col].dropna(), wtg[col].dropna()
    t,p = stats.ttest_ind(s, w, equal_var=False)
    print('  %-15s: STG=%.2f vs WTG=%.2f  t=%+.2f p=%.4f %s' % (
        col, s.mean(), w.mean(), t, p, '*' if p<0.05 else ''))

# MC jump comparison
print('\n=== MC jump rate ===')
for g in ['STG','WTG','Normal']:
    sub = df[df['group']==g]
    print('  %s: %d/%d have jump (%.0f%%)' % (g, sub.has_jump.sum(), len(sub), 100*sub.has_jump.mean()))
