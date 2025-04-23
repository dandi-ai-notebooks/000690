import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Script to explore the running wheel data in the NWB file.
# This includes plotting the running speed over time.

url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access running wheel data
running_speed = nwb.processing["running"].data_interfaces["running_speed"].data[:]
running_speed_timestamps = nwb.processing["running"].data_interfaces["running_speed"].timestamps[:]

# Plot running speed
plt.figure(figsize=(10, 6))
plt.plot(running_speed_timestamps, running_speed, label="Running Speed")
plt.xlabel("Time (s)")
plt.ylabel("Running Speed (cm/s)")
plt.title("Running Speed Over Time")
plt.legend()
plt.savefig("explore/running_speed.png")

print("Running wheel exploration script completed. Plot saved to explore/ directory.")