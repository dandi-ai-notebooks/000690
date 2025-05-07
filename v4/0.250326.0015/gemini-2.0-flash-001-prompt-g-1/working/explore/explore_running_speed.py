import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to explore the running_speed data in the NWB file

url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the running speed data
running_speed = nwb.processing["running"].data_interfaces["running_speed"]
running_speed_data = running_speed.data
running_speed_timestamps = running_speed.timestamps

# Plot the running speed over time
num_timepoints = min(1000, len(running_speed_timestamps))  # Limit the number of timepoints

plt.figure(figsize=(10, 5))
plt.plot(running_speed_timestamps[:num_timepoints], running_speed_data[:num_timepoints])
plt.xlabel("Time (s)")
plt.ylabel("Running Speed (cm/s)")
plt.title("Running Speed Over Time")
plt.savefig("explore/running_speed.png")

plt.close()

print("Running speed data exploration script completed. Plot saved to explore/running_speed.png")