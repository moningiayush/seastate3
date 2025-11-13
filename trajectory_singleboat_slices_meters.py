import os
import json
from dataclasses import replace
from datetime import timedelta
import numpy as np
import pandas as pd

import trajectory_modeling_meters as tmm

TRIP_ID = 395
SLICE_MINUTES = 10
OUT_ROOT = 'outputs_singleboat_slices_meters'


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
    print('Loading & cleaning...')
    df = tmm.load_and_clean(tmm.CFG)
    df = df[df[tmm.CFG.trip_col] == TRIP_ID].copy()
    if df.empty:
        raise SystemExit(f'No rows for {tmm.CFG.trip_col}={TRIP_ID}')
    df = df.sort_values([tmm.CFG.trip_col, tmm.CFG.time_col]).reset_index(drop=True)

    median_dt = estimate_median_dt_seconds(df, tmm.CFG)
    steps = max(1, int(round(SLICE_MINUTES * 60.0 / median_dt)))
    cfg = replace(tmm.CFG, horizon=steps)
    print(f'Horizon {SLICE_MINUTES} min -> ~{steps} steps (median dt {median_dt:.2f}s)')

    # Build features, windows
    df_xy = tmm.build_xy_features(df, tmm.CFG)
    X, y, meta, feats = tmm.make_windows(df_xy, cfg)
    if len(X) == 0:
        raise SystemExit('Not enough windows')

    # Split train/val/test
    tr, va, te = tmm.time_split(meta, cfg.test_size, cfg.val_size)
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]

    # Scale
    Xtr_s, Xva_s, _, scaler = tmm.scale_fit(Xtr, Xva, Xva)

    # Train best model (Conv1D_LSTM)
    inp_shape = Xtr_s.shape[1:]
    model = tmm.build_conv1d_lstm(inp_shape)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    from tensorflow.keras import callbacks
    es = callbacks.EarlyStopping(monitor='val_loss', patience=tmm.CFG.patience, restore_best_weights=True)
    model.fit(Xtr_s, ytr, validation_data=(Xva_s, yva), epochs=tmm.CFG.epochs, batch_size=tmm.CFG.batch_size, verbose=0, callbacks=[es])

    # Predict on all windows
    n_steps, n_feat = X.shape[1], X.shape[2]
    Xfull_s = scaler.transform(X.reshape(-1, n_feat)).reshape(-1, n_steps, n_feat)
    y_pred = model.predict(Xfull_s, verbose=0)

    # Reconstruct abs lat/lon for plotting
    lat_base = np.array([m['lat'] for m in meta])
    lon_base = np.array([m['lon'] for m in meta])
    lat_true = np.array([m['lat_next'] for m in meta])
    lon_true = np.array([m['lon_next'] for m in meta])
    # meters to degrees
    dlat_pred_deg = np.degrees(y_pred[:,1] / 6371000.0)
    dlon_pred_deg = np.degrees(y_pred[:,0] / (6371000.0 * np.cos(np.radians(lat_base))))
    lat_pred = lat_base + dlat_pred_deg
    lon_pred = lon_base + dlon_pred_deg
    times = pd.Series([m['time'] for m in meta])

    # Slices and outputs
    slices = build_time_slices(times, SLICE_MINUTES)
    out_root = os.path.join(os.getcwd(), OUT_ROOT, f'trip_{TRIP_ID}', f'h_{SLICE_MINUTES}min')
    os.makedirs(out_root, exist_ok=True)

    index = []
    for i, (t0, t1) in enumerate(slices, start=1):
        mask = (times >= t0) & (times < t1)
        if not mask.any():
            continue
        lt = lat_true[mask.values]
        ln_t = lon_true[mask.values]
        lp = lat_pred[mask.values]
        ln_p = lon_pred[mask.values]
        slice_dir = os.path.join(out_root, f'slice_{i:03d}')
        os.makedirs(slice_dir, exist_ok=True)
        # Save figures
        tmm.plot_path_overlay(lt, ln_t, lp, ln_p, os.path.join(slice_dir, 'path_overlay.png'))
        tmm.make_folium_map(lt, ln_t, lp, ln_p, os.path.join(slice_dir, 'map.html'))
        # Metrics
        hv = tmm.haversine_meters(lt, ln_t, lp, ln_p)
        rec = {
            'slice': i,
            'start': t0.isoformat(),
            'end': t1.isoformat(),
            'count': int(mask.sum()),
            'hv_mae_m': float(np.mean(np.abs(hv))) if len(hv) else None,
        }
        index.append(rec)

    with open(os.path.join(out_root, 'index.json'), 'w') as f:
        json.dump({'tripId': TRIP_ID, 'horizon_minutes': SLICE_MINUTES, 'median_dt_seconds': median_dt, 'slices': index}, f, indent=2)

    print('Done. Outputs at', out_root)


if __name__ == '__main__':
    run()
