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

# Get running speed data
running_speed = nwb.processing['running'].data_interfaces['running_speed']
data = running_speed.data
timestamps = running_speed.timestamps

# Plot the first 1000 data points
n_samples = min(1000, data.shape[0])
plt.figure(figsize=(10, 5))
plt.plot(timestamps[:n_samples], data[:n_samples], label='Running Speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Running Speed Data (First 1000 Samples)')
plt.legend()
plt.savefig('explore/running_speed.png')
plt.close()

print("Running speed plot saved to explore/running_speed.png")