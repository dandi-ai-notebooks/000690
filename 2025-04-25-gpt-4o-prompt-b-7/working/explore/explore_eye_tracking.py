# Script to explore and visualize eye tracking data from the NWB file
# We will examine eye-tracking spatial series data

import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Define the URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Access the correct dataset for eye tracking spatial series
timestamps = h5_file['/acquisition/EyeTracking/eye_tracking/timestamps'][:]
data = h5_file['/acquisition/EyeTracking/eye_tracking/data'][:]

# Sample a subset of data for visualization
sample_size = 1000
timestamps_sample = timestamps[:sample_size]
data_sample = data[:sample_size, :]

# Plot the eye-tracking data
plt.figure(figsize=(10, 6))
plt.plot(timestamps_sample, data_sample[:, 0], label="X coordinate")
plt.plot(timestamps_sample, data_sample[:, 1], label="Y coordinate")
plt.xlabel("Time (seconds)")
plt.ylabel("Position (meters)")
plt.title("Eye Tracking - Position over Time")
plt.legend()
plt.savefig("explore/eye_tracking.png")
plt.close()

# Close the file
h5_file.close()