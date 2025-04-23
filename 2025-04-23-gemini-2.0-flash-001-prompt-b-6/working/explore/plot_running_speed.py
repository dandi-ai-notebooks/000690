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

# Plot running speed
running_speed = nwb.processing["running"].data_interfaces["running_speed"].data[:]
running_speed_timestamps = nwb.processing["running"].data_interfaces["running_speed"].timestamps[:]

plt.figure(figsize=(10, 5))
plt.plot(running_speed_timestamps, running_speed)
plt.xlabel("Time (s)")
plt.ylabel("Running Speed (cm/s)")
plt.title("Running Speed")
plt.savefig("explore/running_speed.png")
plt.close()