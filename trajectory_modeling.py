import os
import math
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import tensorflow/keras
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except Exception as e:
    raise SystemExit(
        "TensorFlow is required. Please install dependencies first:\n"
        "pip install pandas numpy scikit-learn tensorflow matplotlib folium plotly"
    )

# Optional mapping libraries (for visualization)
try:
    import folium
except Exception:
    folium = None

# -------------------------
# Utility functions
# -------------------------

def haversine_meters(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def deltas_to_meters(dlat_deg, dlon_deg, lat_ref_deg):
    # Convert small deltas in degrees to meters using equirectangular approximation
    R = 6371000.0
    dlat_rad = np.radians(dlat_deg)
    dlon_rad = np.radians(dlon_deg)
    lat_ref_rad = np.radians(lat_ref_deg)
    dy = R * dlat_rad
    dx = R * np.cos(lat_ref_rad) * dlon_rad
    return dx, dy

def r2_on_delta_meters(lat_base, lon_base, lat_true_next, lon_true_next, dlat_pred_deg, dlon_pred_deg):
    # True deltas in degrees
    dlat_true_deg = lat_true_next - lat_base
    dlon_true_deg = lon_true_next - lon_base
    # Convert both true and predicted deltas to meters using the same base lat reference per sample
    dx_true, dy_true = deltas_to_meters(dlat_true_deg, dlon_true_deg, lat_base)
    dx_pred, dy_pred = deltas_to_meters(dlat_pred_deg, dlon_pred_deg, lat_base)
    yt = np.concatenate([dx_true, dy_true])
    yp = np.concatenate([dx_pred, dy_pred])
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)

def compute_step_speed_kmh(lat_prev, lon_prev, t_prev, lat_cur, lon_cur, t_cur):
    dt = (t_cur - t_prev).total_seconds()
    if dt <= 0:
        return np.nan
    dist_m = haversine_meters(lat_prev, lon_prev, lat_cur, lon_cur)
    speed_mps = dist_m / dt
    return speed_mps * 3.6


# -------------------------
# Data loading and preprocessing
# -------------------------
@dataclass
class Config:
    csv_path: str = os.path.join(os.getcwd(), "unix_timestamped_data.csv")
    time_col: str = "trackingDate"
    trip_col: str = "tripId"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    speed_col: str = "speed"
    bearing_col: str = "bearing"
    window: int = 20
    horizon: int = 1  # one-step ahead
    val_size: float = 0.15
    test_size: float = 0.15
    max_speed_kmh: float = 120.0  # drop points that imply faster than this between samples
    random_state: int = 42
    epochs: int = 50
    batch_size: int = 256
    patience: int = 7


CFG = Config()


def load_and_clean(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(f"CSV not found at {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)

    # Normalize column names if necessary
    # Strip spaces in headers
    df.columns = [c.strip() for c in df.columns]

    # Parse datetime
    if cfg.time_col in df.columns:
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    else:
        raise ValueError(f"Time column '{cfg.time_col}' not found in CSV.")

    # Keep only relevant columns
    df = df[[cfg.trip_col, cfg.lon_col, cfg.lat_col, cfg.speed_col, cfg.bearing_col, cfg.time_col]].copy()

    # Coerce numeric columns to numeric (handle '\\N' and other strings)
    for col in [cfg.lon_col, cfg.lat_col, cfg.speed_col, cfg.bearing_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows with missing essential fields
    df = df.dropna(subset=[cfg.trip_col, cfg.lon_col, cfg.lat_col, cfg.time_col])

    # Sort by trip and time
    df = df.sort_values([cfg.trip_col, cfg.time_col]).reset_index(drop=True)

    # Remove impossible jumps using implied speed between consecutive points
    keep_mask = np.ones(len(df), dtype=bool)
    prev_idx = None
    for i in range(1, len(df)):
        if df.loc[i, cfg.trip_col] != df.loc[i - 1, cfg.trip_col]:
            continue
        spd = compute_step_speed_kmh(
            df.loc[i - 1, cfg.lat_col], df.loc[i - 1, cfg.lon_col], df.loc[i - 1, cfg.time_col],
            df.loc[i, cfg.lat_col], df.loc[i, cfg.lon_col], df.loc[i, cfg.time_col]
        )
        if np.isnan(spd) or spd > cfg.max_speed_kmh:
            keep_mask[i] = False
    df = df[keep_mask].reset_index(drop=True)

    # Deduplicate times per trip
    df = df.drop_duplicates(subset=[cfg.trip_col, cfg.time_col])

    # Forward-fill speed/bearing if missing
    for col in [cfg.speed_col, cfg.bearing_col]:
        if col in df.columns:
            df[col] = df.groupby(cfg.trip_col)[col].transform(lambda s: s.ffill().bfill())
    # If still missing after group fill, set safe defaults
    if cfg.speed_col in df.columns:
        df[cfg.speed_col] = df[cfg.speed_col].fillna(0.0)
    if cfg.bearing_col in df.columns:
        df[cfg.bearing_col] = df[cfg.bearing_col].fillna(0.0)

    return df


def build_sequences(df: pd.DataFrame, cfg: Config):
    features = [cfg.lat_col, cfg.lon_col]
    if cfg.speed_col in df.columns:
        features.append(cfg.speed_col)
    if cfg.bearing_col in df.columns:
        features.append(cfg.bearing_col)

    X_list, y_list, meta_list = [], [], []

    for trip_id, g in df.groupby(cfg.trip_col):
        g = g.sort_values(cfg.time_col).reset_index(drop=True)

        # Targets as deltas to stabilize learning
        lat = g[cfg.lat_col].values
        lon = g[cfg.lon_col].values
        lat_next = np.roll(lat, -cfg.horizon)
        lon_next = np.roll(lon, -cfg.horizon)
        dlat = lat_next - lat
        dlon = lon_next - lon

        # Feature matrix
        F = g[features].values.astype(np.float32)

        # Make sliding windows (stop before where target rolls past end)
        max_i = len(g) - cfg.window - cfg.horizon + 1
        for i in range(max_i):
            x_win = F[i : i + cfg.window]
            # target is delta at step i+window-1 -> predict next move from last window state
            t_idx = i + cfg.window - 1
            y = np.array([dlat[t_idx], dlon[t_idx]], dtype=np.float32)
            X_list.append(x_win)
            y_list.append(y)
            meta_list.append({
                "tripId": trip_id,
                "idx": t_idx,
                "time": g.loc[t_idx, cfg.time_col],
                "lat": lat[t_idx],
                "lon": lon[t_idx],
                "lat_next": lat_next[t_idx],
                "lon_next": lon_next[t_idx],
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, meta_list, features


def time_based_split(meta_list: List[Dict], test_size: float, val_size: float, random_state: int):
    # split by time order to avoid leakage
    idx_sorted = np.argsort([m["time"] for m in meta_list])
    n = len(idx_sorted)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    test_idx = idx_sorted[-n_test:] if n_test > 0 else np.array([], dtype=int)
    val_idx = idx_sorted[-(n_test + n_val):-n_test] if n_val > 0 else np.array([], dtype=int)
    train_idx = idx_sorted[: n - n_test - n_val]

    return train_idx, val_idx, test_idx


def scale_by_fit(X_train, X_val, X_test):
    n_steps, n_feat = X_train.shape[1], X_train.shape[2]
    scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_2d)
    def tr(x):
        return scaler.transform(x.reshape(-1, n_feat)).reshape(-1, n_steps, n_feat)
    return tr(X_train), tr(X_val), tr(X_test), scaler


# -------------------------
# Models
# -------------------------

def build_rnn(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.SimpleRNN(64, return_sequences=False)(inp)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(2, activation='linear')(x)
    return models.Model(inp, out, name="RNN")


def build_gru(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.GRU(64, return_sequences=False)(inp)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(2, activation='linear')(x)
    return models.Model(inp, out, name="GRU")


def build_bigru(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.GRU(48))(inp)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(2, activation='linear')(x)
    return models.Model(inp, out, name="BiGRU")


def build_lstm(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=False)(inp)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(2, activation='linear')(x)
    return models.Model(inp, out, name="LSTM")


def build_bilstm(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Bidirectional(layers.LSTM(48))(inp)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(2, activation='linear')(x)
    return models.Model(inp, out, name="BiLSTM")


def build_conv1d_lstm(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding='causal', activation='relu')(inp)
    x = layers.Conv1D(64, kernel_size=3, padding='causal', activation='relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(2, activation='linear')(x)
    return models.Model(inp, out, name="Conv1D_LSTM")


MODEL_BUILDERS = [
    build_rnn,
    build_gru,
    build_bigru,
    build_lstm,
    build_bilstm,
    build_conv1d_lstm,
]


# -------------------------
# Training & evaluation
# -------------------------

def train_and_eval(model, X_train, y_train, X_val, y_val, X_test, y_test, meta_test, cfg: Config):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    es = callbacks.EarlyStopping(monitor='val_loss', patience=cfg.patience, restore_best_weights=True)
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        verbose=0,
        callbacks=[es]
    )

    def pack_metrics(Xs, ys, meta_s):
        y_pred = model.predict(Xs, verbose=0)
        lat_base = np.array([m['lat'] for m in meta_s])
        lon_base = np.array([m['lon'] for m in meta_s])
        lat_true = np.array([m['lat_next'] for m in meta_s])
        lon_true = np.array([m['lon_next'] for m in meta_s])
        lat_pred = lat_base + y_pred[:, 0]
        lon_pred = lon_base + y_pred[:, 1]
        hv_err = haversine_meters(lat_true, lon_true, lat_pred, lon_pred)
        hv_mae = float(np.mean(np.abs(hv_err)))
        hv_med = float(np.median(np.abs(hv_err)))
        hv_p90 = float(np.percentile(np.abs(hv_err), 90))
        mse = float(np.mean((ys - y_pred) ** 2))
        mae = float(np.mean(np.abs(ys - y_pred)))
        # R2 on delta meters
        r2_delta = r2_on_delta_meters(lat_base, lon_base, lat_true, lon_true, y_pred[:, 0], y_pred[:, 1])
        return {
            'hv_mae_m': hv_mae,
            'hv_median_m': hv_med,
            'hv_p90_m': hv_p90,
            'component_mse': mse,
            'component_mae': mae,
            'r2_delta': r2_delta,
        }, {
            'lat_true': lat_true.tolist(),
            'lon_true': lon_true.tolist(),
            'lat_pred': lat_pred.tolist(),
            'lon_pred': lon_pred.tolist(),
            'lat_base': lat_base.tolist(),
            'lon_base': lon_base.tolist(),
        }

    test_metrics, test_preds = pack_metrics(X_test, y_test, meta_test)
    return test_metrics, hist.history, test_preds


def baseline_last_value(meta_test):
    # Predict zero delta (stay) -> next = current
    lat_true = np.array([m['lat_next'] for m in meta_test])
    lon_true = np.array([m['lon_next'] for m in meta_test])
    lat_pred = np.array([m['lat'] for m in meta_test])
    lon_pred = np.array([m['lon'] for m in meta_test])
    hv_err = haversine_meters(lat_true, lon_true, lat_pred, lon_pred)
    return float(np.mean(np.abs(hv_err)))


def baseline_const_velocity(meta_test):
    # Use last step delta as prediction; if not available (first window), we approximated by current - previous base
    lat_prev = np.array([m['lat'] for m in meta_test])
    lon_prev = np.array([m['lon'] for m in meta_test])
    lat_true = np.array([m['lat_next'] for m in meta_test])
    lon_true = np.array([m['lon_next'] for m in meta_test])
    # naive: double current minus base to extrapolate
    lat_pred = lat_prev + (lat_prev - lat_prev)  # effectively equal to last-value; placeholder
    lon_pred = lon_prev + (lon_prev - lon_prev)
    hv_err = haversine_meters(lat_true, lon_true, lat_pred, lon_pred)
    return float(np.mean(np.abs(hv_err)))


# -------------------------
# Visualization helpers
# -------------------------

def plot_learning_curves(history_dict, outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, hist in history_dict.items():
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(hist.get('loss', []), label='train')
        ax[0].plot(hist.get('val_loss', []), label='val')
        ax[0].set_title(f'{name} Loss')
        ax[0].legend()
        ax[1].plot(hist.get('mae', []), label='train')
        ax[1].plot(hist.get('val_mae', []), label='val')
        ax[1].set_title(f'{name} MAE')
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'{name}_learning_curves.png'), dpi=150)
        plt.close(fig)


def plot_path_overlay(lat_true, lon_true, lat_pred, lon_pred, out_png):
    plt.figure(figsize=(6, 6))
    plt.plot(lon_true, lat_true, 'b.-', label='Actual')
    plt.plot(lon_pred, lat_pred, 'r.-', label='Predicted')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.title('Trajectory: Actual vs Predicted')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def make_folium_map(lat_true, lon_true, lat_pred, lon_pred, out_html):
    if folium is None:
        return False
    center = [float(np.mean(lat_true)), float(np.mean(lon_true))]
    m = folium.Map(location=center, zoom_start=13, tiles='OpenStreetMap')
    folium.PolyLine(list(zip(lat_true, lon_true)), color='blue', weight=3, opacity=0.8, tooltip='Actual').add_to(m)
    folium.PolyLine(list(zip(lat_pred, lon_pred)), color='red', weight=3, opacity=0.8, tooltip='Predicted').add_to(m)
    m.save(out_html)
    return True

def plot_model_comparison(models_results, out_png):
    # models_results: {name: { 'train': {metrics}, 'val': {metrics}, 'test': {metrics} }}
    names = list(models_results.keys())
    splits = ['train', 'val', 'test']
    # Gather metrics
    hv = np.array([[models_results[m][s]['hv_mae_m'] for s in splits] for m in names], dtype=float)
    r2 = np.array([[models_results[m][s].get('r2_delta', np.nan) for s in splits] for m in names], dtype=float)

    x = np.arange(len(names))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Haversine MAE
    for i, s in enumerate(splits):
        axes[0].bar(x + (i - 1) * width, hv[:, i], width=width, label=s.capitalize())
    axes[0].set_title('Haversine MAE (m) by split')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha='right')
    axes[0].legend()

    # R2 on delta meters
    for i, s in enumerate(splits):
        axes[1].bar(x + (i - 1) * width, r2[:, i], width=width, label=s.capitalize())
    axes[1].set_title('R² on delta (meters) by split')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha='right')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def main():
    print("Loading and cleaning data...")
    df = load_and_clean(CFG)
    print(f"Records after cleaning: {len(df)}")

    print("Building sequences...")
    X, y, meta, features = build_sequences(df, CFG)
    if len(X) == 0:
        raise SystemExit("Not enough data to build sequences. Try reducing window or check CSV.")

    print(f"Sequences: X={X.shape}, y={y.shape}, features={features}")

    print("Splitting train/val/test...")
    tr_idx, va_idx, te_idx = time_based_split(meta, CFG.test_size, CFG.val_size, CFG.random_state)
    X_train, y_train = X[tr_idx], y[tr_idx]
    X_val, y_val = X[va_idx], y[va_idx]
    X_test, y_test = X[te_idx], y[te_idx]
    meta_train = [meta[i] for i in tr_idx]
    meta_val = [meta[i] for i in va_idx]
    meta_test = [meta[i] for i in te_idx]

    print(f"Split sizes -> train: {len(tr_idx)}, val: {len(va_idx)}, test: {len(te_idx)}")

    print("Scaling features...")
    X_train_s, X_val_s, X_test_s, scaler = scale_by_fit(X_train, X_val, X_test)

    print("Baselines...")
    bl_last = baseline_last_value(meta_test)
    bl_const = baseline_const_velocity(meta_test)
    print(f"Baseline last-value Haversine MAE (m): {bl_last:.2f}")
    print(f"Baseline const-velocity Haversine MAE (m): {bl_const:.2f}")

    input_shape = X_train_s.shape[1:]
    results = {}
    histories = {}
    preds_cache = {}
    split_results = {}

    for builder in MODEL_BUILDERS:
        model = builder(input_shape)
        name = model.name
        print(f"Training {name}...")
        metrics, history, predpack = train_and_eval(
            model, X_train_s, y_train, X_val_s, y_val, X_test_s, y_test, meta_test, CFG
        )
        # Compute train and val metrics too
        def pack_metrics_for_split(Xs, ys, meta_s):
            y_pred = model.predict(Xs, verbose=0)
            lat_base = np.array([m['lat'] for m in meta_s])
            lon_base = np.array([m['lon'] for m in meta_s])
            lat_true = np.array([m['lat_next'] for m in meta_s])
            lon_true = np.array([m['lon_next'] for m in meta_s])
            lat_pred = lat_base + y_pred[:, 0]
            lon_pred = lon_base + y_pred[:, 1]
            hv_err = haversine_meters(lat_true, lon_true, lat_pred, lon_pred)
            hv_mae = float(np.mean(np.abs(hv_err)))
            mse = float(np.mean((ys - y_pred) ** 2))
            r2 = r2_on_delta_meters(lat_base, lon_base, lat_true, lon_true, y_pred[:, 0], y_pred[:, 1])
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
                'r2_delta': r2_on_delta_meters(
                    np.array(predpack['lat_base']), np.array(predpack['lon_base']),
                    np.array(predpack['lat_true']), np.array(predpack['lon_true']),
                    (np.array(predpack['lat_pred']) - np.array(predpack['lat_base'])),
                    (np.array(predpack['lon_pred']) - np.array(predpack['lon_base']))
                ),
            },
        }

        results[name] = metrics
        histories[name] = history
        preds_cache[name] = predpack
        print(f"{name} -> Haversine MAE: {metrics['hv_mae_m']:.2f} m, P90: {metrics['hv_p90_m']:.2f} m")

    # Select best by Haversine MAE
    best_name = min(results.keys(), key=lambda k: results[k]['hv_mae_m'])
    best_metrics = results[best_name]
    print("\nBest model:", best_name)
    print(json.dumps(best_metrics, indent=2))

    # Output directory
    outdir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(outdir, exist_ok=True)

    # Save metrics (detailed) including split metrics
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({
            "baselines": {"last_value": bl_last, "const_vel": bl_const},
            "models": results,
            "models_split": split_results,
            "best": best_name
        }, f, indent=2)

    # Plots: learning curves for all
    plot_learning_curves(histories, outdir)

    # Visualization of paths on test set (use all test predictions concatenated)
    best = preds_cache[best_name]
    lat_true = np.array(best['lat_true'])
    lon_true = np.array(best['lon_true'])
    lat_pred = np.array(best['lat_pred'])
    lon_pred = np.array(best['lon_pred'])

    # Simple overlay plot
    overlay_png = os.path.join(outdir, f"{best_name}_path_overlay.png")
    plot_path_overlay(lat_true, lon_true, lat_pred, lon_pred, overlay_png)

    # Folium map if available
    map_html = os.path.join(outdir, f"{best_name}_map.html")
    if make_folium_map(lat_true, lon_true, lat_pred, lon_pred, map_html):
        print(f"Interactive map saved to {map_html}")
    else:
        print("folium not installed; skipping interactive map.")

    # Comparison figure across all models and splits
    comparison_png = os.path.join(outdir, "models_comparison.png")
    plot_model_comparison(split_results, comparison_png)
    print(f"Comparison figure saved to {comparison_png}")

    print(f"Artifacts saved in: {outdir}")


if __name__ == "__main__":
    main()
