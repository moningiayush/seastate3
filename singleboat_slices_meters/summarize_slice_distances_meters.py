import os, json
from dataclasses import replace
from datetime import timedelta
import numpy as np
import pandas as pd

import trajectory_modeling_meters as tmm

TRIP_ID = 395
SLICE_MINUTES = 10


def estimate_median_dt_seconds(df, cfg) -> float:
    g = df.sort_values([cfg.trip_col, cfg.time_col]).copy()
    dt = g.groupby(cfg.trip_col)[cfg.time_col].diff().dt.total_seconds()
    med = float(np.nanmedian(dt))
    if not np.isfinite(med) or med <= 0:
        med = 60.0
    return med


def build_time_slices(times: pd.Series, minutes: int):
    if times.empty:
        return []
    start = times.min().floor('min')
    end = times.max().ceil('min')
    window = timedelta(minutes=minutes)
    t = start
    out = []
    while t < end:
        out.append((t, t + window))
        t = t + window
    return out


def run():
    df = tmm.load_and_clean(tmm.CFG)
    df = df[df[tmm.CFG.trip_col] == TRIP_ID].copy()
    if df.empty:
        raise SystemExit(f'No rows for {tmm.CFG.trip_col}={TRIP_ID}')
    df = df.sort_values([tmm.CFG.trip_col, tmm.CFG.time_col]).reset_index(drop=True)

    median_dt = estimate_median_dt_seconds(df, tmm.CFG)
    steps = max(1, int(round(SLICE_MINUTES * 60.0 / median_dt)))
    cfg = replace(tmm.CFG, horizon=steps)

    df_xy = tmm.build_xy_features(df, tmm.CFG)
    X, y, meta, feats = tmm.make_windows(df_xy, cfg)
    if len(X) == 0:
        raise SystemExit('Not enough windows')

    lat_next = np.array([m['lat_next'] for m in meta])
    lon_next = np.array([m['lon_next'] for m in meta])
    times = pd.Series([m['time'] for m in meta])

    slices = build_time_slices(times, SLICE_MINUTES)
    records = []
    for i, (t0, t1) in enumerate(slices, start=1):
        mask = (times >= t0) & (times < t1)
        idx = np.where(mask.values)[0]
        if len(idx) < 2:
            continue
        lt = lat_next[idx]
        ln = lon_next[idx]
        # sum haversine between consecutive next-points
        d = 0.0
        for k in range(1, len(idx)):
            d += float(tmm.haversine_meters(lt[k-1], ln[k-1], lt[k], ln[k]))
        records.append({
            'slice': i,
            'start': t0.isoformat(),
            'end': t1.isoformat(),
            'points': len(idx),
            'distance_m': d,
        })

    out_dir = os.path.join(os.getcwd(), 'outputs_singleboat_slices_meters', f'trip_{TRIP_ID}', f'h_{SLICE_MINUTES}min')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'slice_distances.json')
    with open(out_path, 'w') as f:
        json.dump({'tripId': TRIP_ID, 'horizon_minutes': SLICE_MINUTES, 'median_dt_seconds': median_dt, 'slices': records,
                   'mean_distance_m': float(np.mean([r['distance_m'] for r in records])) if records else None,
                   'median_distance_m': float(np.median([r['distance_m'] for r in records])) if records else None}, f, indent=2)
    print('Saved to', out_path)


if __name__ == '__main__':
    run()
