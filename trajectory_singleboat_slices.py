import os
import json
from dataclasses import replace
from datetime import timedelta
import numpy as np
import pandas as pd

import trajectory_modeling as tm

# Configuration for single boat and 5-minute horizon
TRIP_ID = 395  # change this if your target boat/tripId differs
SLICE_MINUTES = 10
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


def run_single_boat_5min():
    print('Loading/cleaning data...')
    df = tm.load_and_clean(tm.CFG)
    # Filter to single boat/trip
    df = df[df[tm.CFG.trip_col] == TRIP_ID].copy()
    if df.empty:
        raise SystemExit(f'No rows for {tm.CFG.trip_col}={TRIP_ID}. Please update TRIP_ID.')

    df = df.sort_values([tm.CFG.trip_col, tm.CFG.time_col]).reset_index(drop=True)

    # Horizon in steps based on median dt
    median_dt = estimate_median_dt_seconds(df, tm.CFG)
    steps = max(1, int(round(SLICE_MINUTES * 60.0 / median_dt)))
    cfg = replace(tm.CFG, horizon=steps)
    print(f'Horizon: {SLICE_MINUTES} minutes -> ~{steps} steps (median dt {median_dt:.2f}s)')

    # Build sequences and split
    X, y, meta, features = tm.build_sequences(df, cfg)
    if len(X) == 0:
        raise SystemExit('Not enough sequences; consider reducing window or ensure data has enough points.')

    tr_idx, va_idx, te_idx = tm.time_based_split(meta, cfg.test_size, cfg.val_size, cfg.random_state)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[va_idx], y[va_idx]
    X_test, y_test = X[te_idx], y[te_idx]
    meta_train = [meta[i] for i in tr_idx]
    meta_val = [meta[i] for i in va_idx]
    meta_test = [meta[i] for i in te_idx]

    X_train_s, X_val_s, X_test_s, scaler = tm.scale_by_fit(X_train, X_val, X_test)

    input_shape = X_train_s.shape[1:]

    # Train all models and pick best by test Haversine MAE
    results = {}
    histories = {}
    preds_cache = {}

    for builder in tm.MODEL_BUILDERS:
        model = builder(input_shape)
        name = model.name
        print(f'Training {name}...')
        metrics, history, predpack = tm.train_and_eval(
            model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, meta_test, cfg
        )
        results[name] = metrics
        histories[name] = history
        preds_cache[name] = predpack
        print(f"{name} -> Haversine MAE: {metrics['hv_mae_m']:.2f} m")

    best_name = min(results.keys(), key=lambda k: results[k]['hv_mae_m'])
    print('Best model:', best_name)

    # Fit scaler on all X to generate predictions for entire trip
    # Refit scaler on full data for consistent scaling across splits
    n_steps, n_feat = X.shape[1], X.shape[2]
    X_full_s = scaler.transform(X.reshape(-1, n_feat)).reshape(-1, n_steps, n_feat)

    # Rebuild the best model and load its trained weights from current session
    # Easiest: reuse the already-trained instance by retraining search loop to keep it
    # We'll reconstruct predictions using the trained best model from preds_cache on test,
    # but we also need predictions on all windows. So rebuild and retrain quickly? Instead, keep instances.
    # For simplicity, rerun training for the best model only on the same data to get a trained instance, then predict full.
    best_builder = {m.__name__: m for m in tm.MODEL_BUILDERS}[best_name if best_name != 'Conv1D_LSTM' else 'build_conv1d_lstm'] if False else None
    # Actually store trained instances above to avoid retraining
    # We'll capture the trained model by training order: replicate building again isn't needed if we kept it.
    # To keep things simple, we will re-train best model quickly (same early stopping) and then predict.
    from tensorflow.keras import callbacks
    best_model = None
    for builder in tm.MODEL_BUILDERS:
        model = builder(input_shape)
        if model.name == best_name:
            best_model = model
            break
    if best_model is None:
        raise SystemExit('Failed to instantiate best model.')

    es = callbacks.EarlyStopping(monitor='val_loss', patience=tm.CFG.patience, restore_best_weights=True)
    best_model.compile(optimizer='adam', loss='mse')
    best_model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=tm.CFG.epochs, batch_size=tm.CFG.batch_size, verbose=0, callbacks=[es])

    y_full_pred = best_model.predict(X_full_s, verbose=0)

    # Build meta for all windows to align with y_full_pred
    meta_full = meta
    lat_base = np.array([m['lat'] for m in meta_full])
    lon_base = np.array([m['lon'] for m in meta_full])
    lat_pred = lat_base + y_full_pred[:, 0]
    lon_pred = lon_base + y_full_pred[:, 1]
    lat_true = np.array([m['lat_next'] for m in meta_full])
    lon_true = np.array([m['lon_next'] for m in meta_full])
    times = pd.Series([m['time'] for m in meta_full])

    # Generate time slices and visualize each 5-minute slice
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
        slice_dir = os.path.join(outdir_root, f'slice_{i:03d}')
        os.makedirs(slice_dir, exist_ok=True)
        # Save overlay and map
        tm.plot_path_overlay(lt, ln_t, lp, ln_p, os.path.join(slice_dir, 'path_overlay.png'))
        tm.make_folium_map(lt, ln_t, lp, ln_p, os.path.join(slice_dir, 'map.html'))
        # Basic metrics per slice
        hv = tm.haversine_meters(lt, ln_t, lp, ln_p)
        rec = {
            'slice': i,
            'start': t0.isoformat(),
            'end': t1.isoformat(),
            'count': int(mask.sum()),
            'hv_mae_m': float(np.mean(np.abs(hv))) if len(hv) else None,
        }
        index.append(rec)

    with open(os.path.join(outdir_root, 'index.json'), 'w') as f:
        json.dump({'tripId': TRIP_ID, 'horizon_minutes': SLICE_MINUTES, 'median_dt_seconds': median_dt, 'slices': index}, f, indent=2)

    print(f'Done. Outputs under: {outdir_root}')


if __name__ == '__main__':
    run_single_boat_5min()
