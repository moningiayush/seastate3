# Single-Boat Slices (Meters-Based) Workflow

## What this does
End-to-end pipeline to train a trajectory model in meters, then generate per-slice (e.g., every 10 minutes) maps, overlays, and distance summaries for a single boat (`tripId`). Primary accuracy metric is Haversine MAE (meters).

## Key scripts
- trajectory_modeling_meters.py
  - Trains Conv1D+LSTM on Δx/Δy (meters) with engineered features (vx, vy, sin/cos(bearing), dt_s).
  - Saves overall metrics and overlay to `outputs_meters/`.
- trajectory_singleboat_slices_meters.py
  - Filters one boat (`TRIP_ID`) and horizon (slice size in minutes → steps via median dt).
  - Trains best model for that boat and generates per-slice overlay and map.
  - Outputs to `outputs_singleboat_slices_meters/trip_<ID>/h_<minutes>min/`.
- summarize_slice_distances_meters.py
  - Computes actual distance traveled per slice (sum of geodesic distances).
  - Saves `slice_distances.json` in the same folder.
- plot_slice_distances.py
  - Reads `slice_distances.json` and produces `slice_distances_bar.png`.

## Metrics
- Haversine MAE (hv_mae): Average great‑circle distance between predicted and ground‑truth GPS points (meters). Primary reporting metric.
- Component MAE/MSE (meters): Errors on Δx/Δy. Helpful for diagnostics.

## Recommended run order
1) Train meters-based model
```
python trajectory_modeling_meters.py
```
Outputs: `outputs_meters/metrics.json`, `outputs_meters/path_overlay.png`

2) Generate single-boat slice outputs (set TRIP_ID and SLICE_MINUTES inside the script)
```
python trajectory_singleboat_slices_meters.py
```
Outputs: `outputs_singleboat_slices_meters/trip_<ID>/h_<minutes>min/`
- `slice_XXX/path_overlay.png`
- `slice_XXX/map.html`
- `index.json` (per-slice hv_mae and metadata)

3) Compute per-slice distance traveled (meters)
```
python summarize_slice_distances_meters.py
```
Outputs: `slice_distances.json` (per-slice distance, mean/median)

4) Plot distance per slice (bar chart)
```
python plot_slice_distances.py
```
Outputs: `slice_distances_bar.png`

## Notes & tips
- Change horizon window by editing `SLICE_MINUTES` in `trajectory_singleboat_slices_meters.py` (e.g., 5/10/15).
- Change boat by editing `TRIP_ID` in the same script.
- The meters-based training improves stability and interpretability versus degree-based (Δlat/Δlon) training.
- Use hv_mae for headline accuracy; component MAE for diagnosing axis-wise errors.

## Output folders overview
- `outputs_meters/` — overall metrics and overlay for the meters-trained model.
- `outputs_singleboat_slices_meters/trip_<ID>/h_<minutes>min/` — per-slice overlays, maps, metrics, distances, and bar chart.
