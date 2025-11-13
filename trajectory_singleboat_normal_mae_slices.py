import os
import json
from dataclasses import replace
from datetime import timedelta
import numpy as np
import pandas as pd

import trajectory_modeling as tm

# Configuration
TRIP_ID = 395
SLICE_MINUTES = 10  # change to 5 or 15 as needed
OUT_ROOT = 'outputs_singleboat_slices'


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
    slices = []
    t = start
    while t < end:
        slices.append((t, t + window))
        t = t + window
    return slices


def delta_meters(true_lat, true_lon, pred_lat, pred_lon, base_lat):
    dlat = true_lat - pred_lat
    dlon = true_lon - pred_lon
    dx, dy = tm.deltas_to_meters(dlat, dlon, base_lat)
    return dx, dy


def run():
    print('Loading/cleaning data...')
    df = tm.load_and_clean(tm.CFG)
    df = df[df[tm.CFG.trip_col] == TRIP_ID].copy()
    if df.empty:
        raise SystemExit(f'No rows for {tm.CFG.trip_col}={TRIP_ID}.')

    df = df.sort_values([tm.CFG.trip_col, tm.CFG.time_col]).reset_index(drop=True)

    median_dt = estimate_median_dt_seconds(df, tm.CFG)
    steps = max(1, int(round(SLICE_MINUTES * 60.0 / median_dt)))
    cfg = replace(tm.CFG, horizon=steps)
    print(f'Horizon: {SLICE_MINUTES} minutes -> ~{steps} steps (median dt {median_dt:.2f}s)')

    # Build sequences and split
    X, y, meta, features = tm.build_sequences(df, cfg)
    if len(X) == 0:
        raise SystemExit('Not enough sequences to build windows.')

    tr_idx, va_idx, te_idx = tm.time_based_split(meta, cfg.test_size, cfg.val_size, cfg.random_state)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[va_idx], y[va_idx]

    # Fit scaler
    X_train_s, X_val_s, _, scaler = tm.scale_by_fit(X_train, X_val, X_val)

    # Train best-performing architecture quickly (Conv1D_LSTM) for consistency
    input_shape = X_train_s.shape[1:]
    model = tm.build_conv1d_lstm(input_shape)
    model.compile(optimizer='adam', loss='mse')
    from tensorflow.keras import callbacks
    es = callbacks.EarlyStopping(monitor='val_loss', patience=tm.CFG.patience, restore_best_weights=True)
    model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=tm.CFG.epochs, batch_size=tm.CFG.batch_size, verbose=0, callbacks=[es])

    # Predict on all windows
    n_steps, n_feat = X.shape[1], X.shape[2]
    X_full_s = scaler.transform(X.reshape(-1, n_feat)).reshape(-1, n_steps, n_feat)
    y_pred = model.predict(X_full_s, verbose=0)

    # Reconstruct absolute predictions
    lat_base = np.array([m['lat'] for m in meta])
    lon_base = np.array([m['lon'] for m in meta])
    lat_true = np.array([m['lat_next'] for m in meta])
    lon_true = np.array([m['lon_next'] for m in meta])
    lat_pred = lat_base + y_pred[:, 0]
    lon_pred = lon_base + y_pred[:, 1]
    times = pd.Series([m['time'] for m in meta])

    # Slices
    slices = build_time_slices(times, SLICE_MINUTES)
    outdir_root = os.path.join(os.getcwd(), OUT_ROOT, f'trip_{TRIP_ID}', f'h_{SLICE_MINUTES}min')
    os.makedirs(outdir_root, exist_ok=True)

    index = []
    for i, (t0, t1) in enumerate(slices, start=1):
        mask = (times >= t0) & (times < t1)
        if not mask.any():
            continue
        lt = lat_true[mask.values]
        ln_t = lon_true[mask.values]
        lp = lat_pred[mask.values]
        ln_p = lon_pred[mask.values]
        lb = lat_base[mask.values]

        # Component-wise absolute errors in meters
        dx, dy = delta_meters(lt, ln_t, lp, ln_p, lb)
        mae_dx = float(np.mean(np.abs(dx))) if len(dx) else None
        mae_dy = float(np.mean(np.abs(dy))) if len(dy) else None
        mae_mean = float(np.mean((np.abs(dx) + np.abs(dy)) / 2.0)) if len(dx) else None

        rec = {
            'slice': i,
            'start': t0.isoformat(),
            'end': t1.isoformat(),
            'count': int(mask.sum()),
            'mae_dx_m': mae_dx,
            'mae_dy_m': mae_dy,
            'mae_xy_mean_m': mae_mean
        }
        index.append(rec)

    out_json = os.path.join(outdir_root, 'index_normal_mae.json')
    with open(out_json, 'w') as f:
        json.dump({'tripId': TRIP_ID, 'horizon_minutes': SLICE_MINUTES, 'median_dt_seconds': median_dt, 'slices': index}, f, indent=2)
    print(f'Done. Saved normal MAE per slice to {out_json}')


if __name__ == '__main__':
    run()
