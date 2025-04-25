# Script to explore and visualize running wheel data from the NWB file
# We will examine running wheel rotation data

import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Define the URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Access the dataset for running wheel rotation
timestamps = h5_file['/acquisition/raw_running_wheel_rotation/timestamps'][:]
data = h5_file['/acquisition/raw_running_wheel_rotation/data'][:]

# Sample a subset of data for visualization
sample_size = 1000
timestamps_sample = timestamps[:sample_size]
data_sample = data[:sample_size]

# Plot the running wheel data
plt.figure(figsize=(10, 6))
plt.plot(timestamps_sample, data_sample, label="Wheel Rotation")
plt.xlabel("Time (seconds)")
plt.ylabel("Rotation (radians)")
plt.title("Running Wheel Rotation over Time")
plt.legend()
plt.savefig("explore/running_wheel_rotation.png")
plt.close()

# Close the file
h5_file.close()