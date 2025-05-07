import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the running speed data
running = nwb.processing['running']
running_speed = running.data_interfaces['running_speed']
timestamps = np.array(running_speed.timestamps[:])
data = np.array(running_speed.data[:])

# Print the first 10 running speeds
print("First 10 running speeds:", data[:10])

# Plot running speed vs. time
plt.figure(figsize=(10, 6))
plt.plot(timestamps, data)
plt.xlabel("Time (s)")
plt.ylabel("Running Speed (cm/s)")
plt.title("Running Speed vs. Time")
plt.savefig("explore/running_speed_vs_time.png")
plt.close()