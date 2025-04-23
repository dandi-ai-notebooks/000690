import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Script to explore the EyeTracking data in the NWB file.
# This includes plotting the eye position data and blink data.

url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access EyeTracking data
eye_tracking = nwb.acquisition["EyeTracking"]
eye_position = eye_tracking.spatial_series["eye_tracking"].data[:]
eye_timestamps = eye_tracking.spatial_series["eye_tracking"].timestamps[:]
likely_blink = eye_tracking.likely_blink.data[:]

# Plot eye position
plt.figure(figsize=(10, 6))
plt.plot(eye_timestamps, eye_position[:, 0], label="X Position")
plt.plot(eye_timestamps, eye_position[:, 1], label="Y Position")
plt.xlabel("Time (s)")
plt.ylabel("Eye Position (meters)")
plt.title("Eye Position Over Time")
plt.legend()
plt.savefig("explore/eye_position.png")

# Plot blink data
plt.figure(figsize=(10, 4))
plt.plot(eye_timestamps, likely_blink, label="Likely Blink")
plt.xlabel("Time (s)")
plt.ylabel("Blink (boolean)")
plt.title("Blink Detection Over Time")
plt.legend()
plt.savefig("explore/blink_detection.png")

# Plot histogram of eye position data
plt.figure(figsize=(8, 6))
sns.histplot(eye_position[:, 0], kde=True, label="X Position")
sns.histplot(eye_position[:, 1], kde=True, label="Y Position")
plt.xlabel("Eye Position (meters)")
plt.ylabel("Frequency")
plt.title("Distribution of Eye Positions")
plt.legend()
plt.savefig("explore/eye_position_histogram.png")

print("Eye tracking exploration script completed. Plots saved to explore/ directory.")