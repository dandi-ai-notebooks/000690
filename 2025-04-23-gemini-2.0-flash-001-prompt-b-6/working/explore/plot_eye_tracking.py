import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Plot eye tracking data
eye_tracking_data = nwb.acquisition["EyeTracking"].spatial_series["eye_tracking"].data[:]
eye_tracking_timestamps = nwb.acquisition["EyeTracking"].spatial_series["eye_tracking"].timestamps[:]

plt.figure(figsize=(10, 5))
plt.plot(eye_tracking_timestamps, eye_tracking_data[:, 0], label="x")
plt.plot(eye_tracking_timestamps, eye_tracking_data[:, 1], label="y")
plt.xlabel("Time (s)")
plt.ylabel("Eye Position (meters)")
plt.title("Eye Tracking")
plt.legend()
plt.savefig("explore/eye_tracking.png")
plt.close()