import os, json
import numpy as np
import matplotlib.pyplot as plt

trip_dir = os.path.join(os.getcwd(), 'outputs_singleboat_slices_meters', 'trip_395', 'h_10min')
with open(os.path.join(trip_dir, 'slice_distances.json'), 'r') as f:
    data = json.load(f)

slices = data['slices']
idx = [s['slice'] for s in slices]
vals = [s['distance_m'] for s in slices]

plt.figure(figsize=(12, 5))
plt.bar(idx, vals, color='#4C78A8')
plt.xlabel('Slice number (10 min each)')
plt.ylabel('Distance (m)')
plt.title('Distance traveled per 10-minute slice (trip {0})'.format(data['tripId']))
plt.tight_layout()
out_path = os.path.join(trip_dir, 'slice_distances_bar.png')
plt.savefig(out_path, dpi=150)
plt.close()
print('Saved', out_path)
