import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Script to explore running wheel data from the NWB file

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get running wheel data
running_wheel_rotation = nwb.processing['running'].data_interfaces['running_wheel_rotation']
running_wheel_data = running_wheel_rotation.data
running_wheel_timestamps = running_wheel_rotation.timestamps[:]

# Plot the running wheel data (first 1000 samples)
num_samples = min(1000, len(running_wheel_data))
plt.figure(figsize=(10, 5))
plt.plot(running_wheel_timestamps[:num_samples], running_wheel_data[:num_samples], label='Rotation')
plt.xlabel('Time (s)')
plt.ylabel('Rotation (radians)')
plt.title('Running Wheel Data')
plt.legend()
plt.savefig('explore/running_wheel_plot.png')
plt.close()

print("Running wheel data exploration complete. Plot saved to explore/running_wheel_plot.png")