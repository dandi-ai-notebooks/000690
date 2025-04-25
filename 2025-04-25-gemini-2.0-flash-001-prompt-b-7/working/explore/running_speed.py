# explore/running_speed.py
# This script explores the running speed data in the NWB file and plots running speed over time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the running speed data
running = nwb.processing["running"]
running_speed = running.data_interfaces["running_speed"]

# Load a subset of the data and timestamps
start = 0
end = 1000  # Load the first 1000 samples
speed_data = running_speed.data[:end]
timestamps = running_speed.timestamps[:end]

# Plot the running speed over time
plt.figure(figsize=(10, 5))
plt.plot(timestamps, speed_data)
plt.xlabel("Time (s)")
plt.ylabel("Running Speed (cm/s)")
plt.title("Running Speed Over Time")
plt.savefig("explore/running_speed.png")
plt.close()