import os
import json
from dataclasses import replace
import numpy as np

import trajectory_modeling as tm

# Utility: estimate median time gap (seconds) between consecutive points per trip
def estimate_median_dt_seconds(df, cfg) -> float:
    g = df.sort_values([cfg.trip_col, cfg.time_col]).copy()
    dt = g.groupby(cfg.trip_col)[cfg.time_col].diff().dt.total_seconds()
    med = float(np.nanmedian(dt))
    if not np.isfinite(med) or med <= 0:
        med = 60.0
    return med


def run_for_minutes(df, base_cfg, minutes: int, out_root: str):
    median_dt = estimate_median_dt_seconds(df, base_cfg)
    steps = max(1, int(round(minutes * 60.0 / median_dt)))
    cfg = replace(base_cfg, horizon=steps)

    print(f"\n=== Horizon: {minutes} min (~{steps} steps at median dt {median_dt:.2f}s) ===")

    # Build sequences for this horizon
    X, y, meta, features = tm.build_sequences(df, cfg)
    if len(X) == 0:
        print("Not enough sequences for this horizon.")
        return

    tr_idx, va_idx, te_idx = tm.time_based_split(meta, cfg.test_size, cfg.val_size, cfg.random_state)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[va_idx], y[va_idx]
    X_test, y_test = X[te_idx], y[te_idx]
    meta_train = [meta[i] for i in tr_idx]
    meta_val = [meta[i] for i in va_idx]
    meta_test = [meta[i] for i in te_idx]

    X_train_s, X_val_s, X_test_s, scaler = tm.scale_by_fit(X_train, X_val, X_test)

    # Baselines (reported only on test for reference)
    bl_last = tm.baseline_last_value(meta_test)
    bl_const = tm.baseline_const_velocity(meta_test)

    input_shape = X_train_s.shape[1:]
    results = {}
    histories = {}
    preds_cache = {}
    split_results = {}

    for builder in tm.MODEL_BUILDERS:
        model = builder(input_shape)
        name = model.name
        print(f"Training {name}...")
        metrics, history, predpack = tm.train_and_eval(
            model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, meta_test, cfg
        )

        # Split metrics (train/val/test)
        def pack_metrics_for_split(Xs, ys, meta_s):
            y_pred = model.predict(Xs, verbose=0)
            lat_base = np.array([m['lat'] for m in meta_s])
            lon_base = np.array([m['lon'] for m in meta_s])
            lat_true = np.array([m['lat_next'] for m in meta_s])
            lon_true = np.array([m['lon_next'] for m in meta_s])
            lat_pred = lat_base + y_pred[:, 0]
            lon_pred = lon_base + y_pred[:, 1]
            hv_err = tm.haversine_meters(lat_true, lon_true, lat_pred, lon_pred)
            hv_mae = float(np.mean(np.abs(hv_err)))
            mse = float(np.mean((ys - y_pred) ** 2))
            # r2 on delta meters
            r2 = tm.r2_on_delta_meters(lat_base, lon_base, lat_true, lon_true, y_pred[:, 0], y_pred[:, 1])
            return {
                'hv_mae_m': hv_mae,
                'component_mse': mse,
                'r2_delta': r2,
            }

        split_results[name] = {
            'train': pack_metrics_for_split(X_train_s, y_train, meta_train),
            'val': pack_metrics_for_split(X_val_s, y_val, meta_val),
            'test': {
                'hv_mae_m': metrics['hv_mae_m'],
                'component_mse': metrics['component_mse'],
                'r2_delta': tm.r2_on_delta_meters(
                    np.array(predpack['lat_base']), np.array(predpack['lon_base']),
                    np.array(predpack['lat_true']), np.array(predpack['lon_true']),
                    (np.array(predpack['lat_pred']) - np.array(predpack['lat_base'])) ,
                    (np.array(predpack['lon_pred']) - np.array(predpack['lon_base']))
                ),
            },
        }

        results[name] = metrics
        histories[name] = history
        preds_cache[name] = predpack
        print(f"{name} -> Haversine MAE: {metrics['hv_mae_m']:.2f} m, P90: {metrics['hv_p90_m']:.2f} m")

    # Best model by Haversine MAE
    best_name = min(results.keys(), key=lambda k: results[k]['hv_mae_m'])

    # Output dir for this horizon
    outdir = os.path.join(os.getcwd(), out_root, f"h_{minutes}min")
    os.makedirs(outdir, exist_ok=True)

    # Save metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({
            'info': {
                'minutes': minutes,
                'steps': steps,
            },
            'baselines': {'last_value': bl_last, 'const_vel': bl_const},
            'models': results,
            'models_split': split_results,
            'best': best_name
        }, f, indent=2)

    # Learning curves
    tm.plot_learning_curves(histories, outdir)

    # Comparison figure across splits/models
    tm.plot_model_comparison(split_results, os.path.join(outdir, 'models_comparison.png'))

    # Map and overlay for best model
    best = preds_cache[best_name]
    lat_true = np.array(best['lat_true'])
    lon_true = np.array(best['lon_true'])
    lat_pred = np.array(best['lat_pred'])
    lon_pred = np.array(best['lon_pred'])
    tm.plot_path_overlay(lat_true, lon_true, lat_pred, lon_pred, os.path.join(outdir, f"{best_name}_path_overlay.png"))
    map_html = os.path.join(outdir, f"{best_name}_map.html")
    tm.make_folium_map(lat_true, lon_true, lat_pred, lon_pred, map_html)


if __name__ == '__main__':
    print('Loading/cleaning data...')
    df = tm.load_and_clean(tm.CFG)
    out_root = 'outputs_multihorizon'
    for minutes in [5, 10, 15]:
        run_for_minutes(df, tm.CFG, minutes, out_root)
    print(f'Done. See {os.path.join(os.getcwd(), out_root)}')
