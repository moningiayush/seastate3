import os, json, numpy as np

base = os.path.join(os.getcwd(), "outputs_singleboat_slices", "trip_395")
results = {}
for sub in ["h_5min", "h_10min"]:
    p = os.path.join(base, sub, "index.json")
    if os.path.exists(p):
        with open(p, "r") as f:
            data = json.load(f)
        hv = [s.get("hv_mae_m") for s in data.get("slices", []) if s.get("hv_mae_m") is not None]
        if hv:
            results[sub] = {
                "num_slices": len(hv),
                "mean_hv_mae_m": float(np.mean(hv)),
                "median_hv_mae_m": float(np.median(hv)),
                "min_hv_mae_m": float(np.min(hv)),
                "max_hv_mae_m": float(np.max(hv)),
            }
        else:
            results[sub] = {"num_slices": 0}
    else:
        results[sub] = {"missing": True}

print(json.dumps(results, indent=2))
