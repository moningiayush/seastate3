import os
import json
import warnings
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
try:
    import folium
except Exception:
    folium = None

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks
except Exception:
    raise SystemExit("Please install tensorflow, pandas, numpy, scikit-learn, matplotlib")


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
    R = 6371000.0
    dlat_rad = np.radians(dlat_deg)
    dlon_rad = np.radians(dlon_deg)
    lat_ref_rad = np.radians(lat_ref_deg)
    dy = R * dlat_rad
    dx = R * np.cos(lat_ref_rad) * dlon_rad
    return dx, dy


@dataclass
class Config:
    csv_path: str = os.path.join(os.getcwd(), "Sample Trip Data - new.csv")
    time_col: str = "trackingDate"
    trip_col: str = "tripId"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    speed_col: str = "speed"
    bearing_col: str = "bearing"
    window: int = 20
    horizon: int = 1
    val_size: float = 0.15
    test_size: float = 0.15
    max_speed_kmh: float = 120.0
    random_state: int = 42
    epochs: int = 60
    batch_size: int = 256
    patience: int = 8


CFG = Config()


def load_and_clean(cfg: Config) -> pd.DataFrame:
    if not os.path.exists(cfg.csv_path):
        raise FileNotFoundError(cfg.csv_path)
    df = pd.read_csv(cfg.csv_path)
    df.columns = [c.strip() for c in df.columns]
    if cfg.time_col not in df.columns:
        raise ValueError(f"Missing time column {cfg.time_col}")
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    # keep cols
    df = df[[cfg.trip_col, cfg.lon_col, cfg.lat_col, cfg.speed_col, cfg.bearing_col, cfg.time_col]].copy()
    # coerce numerics
    for col in [cfg.lon_col, cfg.lat_col, cfg.speed_col, cfg.bearing_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=[cfg.trip_col, cfg.lon_col, cfg.lat_col, cfg.time_col])
    df = df.sort_values([cfg.trip_col, cfg.time_col]).reset_index(drop=True)
    # ffill/bfill speed/bearing per trip
    for col in [cfg.speed_col, cfg.bearing_col]:
        df[col] = df.groupby(cfg.trip_col)[col].transform(lambda s: s.ffill().bfill())
    # dt
    df['dt_s'] = df.groupby(cfg.trip_col)[cfg.time_col].diff().dt.total_seconds()
    df['dt_s'] = df['dt_s'].fillna(df['dt_s'].median()).clip(lower=1.0)
    # remove impossible jumps by implied speed
    keep = np.ones(len(df), dtype=bool)
    for i in range(1, len(df)):
        if df.loc[i, cfg.trip_col] != df.loc[i-1, cfg.trip_col]:
            continue
        dist = haversine_meters(df.loc[i-1, cfg.lat_col], df.loc[i-1, cfg.lon_col], df.loc[i, cfg.lat_col], df.loc[i, cfg.lon_col])
        spd_kmh = (dist / df.loc[i, 'dt_s']) * 3.6
        if spd_kmh > cfg.max_speed_kmh:
            keep[i] = False
    df = df[keep].reset_index(drop=True)
    return df


def build_xy_features(df: pd.DataFrame, cfg: Config):
    # choose a per-trip local origin (first point) for stable meters coordinates
    df_list = []
    for trip, g in df.groupby(cfg.trip_col):
        g = g.sort_values(cfg.time_col).copy()
        lat0 = g[cfg.lat_col].iloc[0]
        lon0 = g[cfg.lon_col].iloc[0]
        dx, dy = deltas_to_meters(g[cfg.lat_col] - lat0, g[cfg.lon_col] - lon0, lat0)
        g['x_m'] = dx
        g['y_m'] = dy
        # velocities (m/s)
        g['vx_mps'] = g['x_m'].diff() / g['dt_s']
        g['vy_mps'] = g['y_m'].diff() / g['dt_s']
        g['vx_mps'] = g['vx_mps'].fillna(0.0)
        g['vy_mps'] = g['vy_mps'].fillna(0.0)
        # bearing sin/cos
        g['bearing_sin'] = np.sin(np.radians(g[CFG.bearing_col].fillna(0.0)))
        g['bearing_cos'] = np.cos(np.radians(g[CFG.bearing_col].fillna(0.0)))
        # speed to m/s if looks like km/h
        sp = g[CFG.speed_col].fillna(0.0)
        # assume speed was in whatever units; we won't rely on it strongly
        g['speed_mps'] = sp.astype(float)
        df_list.append(g)
    return pd.concat(df_list, ignore_index=True)


def make_windows(df: pd.DataFrame, cfg: Config):
    features = ['x_m','y_m','vx_mps','vy_mps','bearing_sin','bearing_cos','speed_mps','dt_s']
    Xs, ys, meta = [], [], []
    for trip, g in df.groupby(cfg.trip_col):
        g = g.sort_values(cfg.time_col).reset_index(drop=True)
        # targets: delta meters over next horizon steps
        x = g['x_m'].values
        y = g['y_m'].values
        x_next = np.roll(x, -cfg.horizon)
        y_next = np.roll(y, -cfg.horizon)
        dx_next = x_next - x
        dy_next = y_next - y
        F = g[features].values.astype(np.float32)
        max_i = len(g) - cfg.window - cfg.horizon + 1
        for i in range(max_i):
            Xs.append(F[i:i+cfg.window])
            t_idx = i + cfg.window - 1
            ys.append(np.array([dx_next[t_idx], dy_next[t_idx]], dtype=np.float32))
            meta.append({
                'tripId': trip,
                'time': g.loc[t_idx, cfg.time_col],
                'x': x[t_idx], 'y': y[t_idx],
                'x_next': x_next[t_idx], 'y_next': y_next[t_idx],
                'lat': g.loc[t_idx, CFG.lat_col], 'lon': g.loc[t_idx, CFG.lon_col],
                'lat_next': g.loc[min(t_idx+cfg.horizon, len(g)-1), CFG.lat_col],
                'lon_next': g.loc[min(t_idx+cfg.horizon, len(g)-1), CFG.lon_col],
            })
    if not Xs:
        return np.empty((0, cfg.window, len(features)), np.float32), np.empty((0,2), np.float32), [], features
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32), meta, features


def time_split(meta: List[Dict], test_size: float, val_size: float):
    idx_sorted = np.argsort([m['time'] for m in meta])
    n = len(idx_sorted)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    te = idx_sorted[-n_test:] if n_test>0 else np.array([], int)
    va = idx_sorted[-(n_test+n_val):-n_test] if n_val>0 else np.array([], int)
    tr = idx_sorted[: n - n_test - n_val]
    return tr, va, te


def scale_fit(X_train, X_val, X_test):
    n_steps, n_feat = X_train.shape[1], X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))
    def tr(a):
        return scaler.transform(a.reshape(-1, n_feat)).reshape(-1, n_steps, n_feat)
    return tr(X_train), tr(X_val), tr(X_test), scaler


# Models

def build_conv1d_lstm(input_shape):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, padding='causal', activation='relu')(inp)
    x = layers.Conv1D(64, 3, padding='causal', activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(2)(x)
    return models.Model(inp, out, name='Conv1D_LSTM')


def train_eval(cfg: Config):
    print('Loading data...')
    df = load_and_clean(CFG)
    df_xy = build_xy_features(df, CFG)
    print(f"Rows after clean: {len(df_xy)}")

    X, y, meta, feats = make_windows(df_xy, CFG)
    if len(X) == 0:
        raise SystemExit('Not enough windows')
    tr, va, te = time_split(meta, CFG.test_size, CFG.val_size)
    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]
    meta_te = [meta[i] for i in te]

    Xtr_s, Xva_s, Xte_s, scaler = scale_fit(Xtr, Xva, Xte)

    model = build_conv1d_lstm(Xtr_s.shape[1:])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    cb = [callbacks.EarlyStopping(monitor='val_loss', patience=CFG.patience, restore_best_weights=True)]
    hist = model.fit(Xtr_s, ytr, validation_data=(Xva_s, yva), epochs=CFG.epochs, batch_size=CFG.batch_size, verbose=0, callbacks=cb)

    y_pred = model.predict(Xte_s, verbose=0)

    # Component MAE in meters (already meters)
    comp_mae = float(np.mean(np.abs(yte - y_pred)))
    comp_mse = float(np.mean((yte - y_pred)**2))

    # Convert to absolute lat/lon for hv
    lat_base = np.array([m['lat'] for m in meta_te])
    lon_base = np.array([m['lon'] for m in meta_te])
    lat_true = np.array([m['lat_next'] for m in meta_te])
    lon_true = np.array([m['lon_next'] for m in meta_te])
    # Reconstruct predicted next lat/lon via meters-to-deg approx around base lat
    dlat_pred_deg = np.degrees(y_pred[:,1] / 6371000.0)
    dlon_pred_deg = np.degrees(y_pred[:,0] / (6371000.0 * np.cos(np.radians(lat_base))))
    lat_pred = lat_base + dlat_pred_deg
    lon_pred = lon_base + dlon_pred_deg

    hv_err = haversine_meters(lat_true, lon_true, lat_pred, lon_pred)
    hv_mae = float(np.mean(np.abs(hv_err)))

    outdir = os.path.join(os.getcwd(), 'outputs_meters')
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump({'component_mae_m': comp_mae, 'component_mse_m2': comp_mse, 'hv_mae_m': hv_mae}, f, indent=2)

    # Simple overlay plot
    plot_path_overlay(lat_true, lon_true, lat_pred, lon_pred, os.path.join(outdir, 'path_overlay.png'))

    print('Saved outputs to', outdir)


def plot_path_overlay(lat_true, lon_true, lat_pred, lon_pred, out_png):
    plt.figure(figsize=(6,6))
    plt.plot(lon_true, lat_true, 'b.-', label='Actual')
    plt.plot(lon_pred, lat_pred, 'r.-', label='Predicted')
    plt.legend(); plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.title('Actual vs Predicted (meters model)')
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def make_folium_map(lat_true, lon_true, lat_pred, lon_pred, out_html):
    if folium is None:
        return False
    center = [float(np.mean(lat_true)), float(np.mean(lon_true))]
    m = folium.Map(location=center, zoom_start=13, tiles='OpenStreetMap')
    folium.PolyLine(list(zip(lat_true, lon_true)), color='blue', weight=3, opacity=0.8, tooltip='Actual').add_to(m)
    folium.PolyLine(list(zip(lat_pred, lon_pred)), color='red', weight=3, opacity=0.8, tooltip='Predicted').add_to(m)
    m.save(out_html)
    return True


if __name__ == '__main__':
    train_eval(CFG)
