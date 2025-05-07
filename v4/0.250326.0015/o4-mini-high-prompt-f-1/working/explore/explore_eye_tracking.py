"""
Exploration of eye-tracking data from sub-692072 session 1298465622.
Generates basic statistics and a scatter plot of the first 1000 eye-tracking samples.
"""
import os
import h5py
import pynwb
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Ensure explore directory exists
os.makedirs(os.path.dirname(__file__), exist_ok=True)

# Remote NWB file URL
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, mode='r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access eye tracking spatial series
eye_tracking = nwb.acquisition["EyeTracking"].spatial_series["eye_tracking"]

# Print basic sample count
total_samples = eye_tracking.data.shape[0]
print(f"Total eye-tracking samples: {total_samples}")

# Load a subset for plotting
n = min(1000, total_samples)
data = eye_tracking.data[:n, :]
timestamps = eye_tracking.timestamps[:n]

# Estimate sampling rate
if n > 1:
    diffs = np.diff(eye_tracking.timestamps[:n])
    sr = 1.0 / np.mean(diffs)
    print(f"Estimated sampling rate: {sr:.2f} Hz")
else:
    print("Not enough data to estimate sampling rate.")

# Create scatter plot
plt.figure(figsize=(6,6))
plt.scatter(data[:, 0], data[:, 1], c=timestamps, cmap='viridis', s=5)
plt.colorbar(label='Time (s)')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title(f'Eye tracking positions (first {n} samples)')

# Save the figure
output_path = os.path.join(os.path.dirname(__file__), "eye_tracking_scatter.png")
plt.savefig(output_path)
plt.close()
print(f"Saved scatter plot to {output_path}")