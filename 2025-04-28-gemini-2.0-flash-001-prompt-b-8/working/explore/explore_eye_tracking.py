import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get eye tracking data
eye_tracking = nwb.acquisition['EyeTracking']
corneal_reflection_tracking = eye_tracking.spatial_series['corneal_reflection_tracking']
data = corneal_reflection_tracking.data
timestamps = corneal_reflection_tracking.timestamps[:]

# Plot the first 1000 data points
n_samples = min(1000, data.shape[0])
plt.figure(figsize=(10, 5))
plt.plot(timestamps[:n_samples], data[:n_samples, 0], label='X')
plt.plot(timestamps[:n_samples], data[:n_samples, 1], label='Y')
plt.xlabel('Time (s)')
plt.ylabel('Position (meters)')
plt.title('Corneal Reflection Tracking Data (First 1000 Samples)')
plt.legend()
plt.savefig('explore/corneal_reflection_tracking.png')
plt.close()

print("Eye tracking plot saved to explore/corneal_reflection_tracking.png")