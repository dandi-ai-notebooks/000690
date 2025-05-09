# Explore running speed data
# This script loads running speed data and corresponding timestamps
# from the NWB file and plots them. It saves the plot to explore/running_speed.png.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r') # Ensure read-only mode
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Ensure read-only mode for NWBHDF5IO
nwb = io.read()

# Access running speed data
running_speed_ts = nwb.processing['running'].data_interfaces['running_speed']
timestamps = running_speed_ts.timestamps[:]
speed_data = running_speed_ts.data[:]

# Select a subset of data to plot (e.g., first 5000 points)
num_points_to_plot = 5000
timestamps_subset = timestamps[:num_points_to_plot]
speed_data_subset = speed_data[:num_points_to_plot]

# Create plot
sns.set_theme()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(timestamps_subset, speed_data_subset, label='Running Speed')
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'Speed ({running_speed_ts.unit})')
ax.set_title(f'Running Speed (First {num_points_to_plot} points)')
ax.legend()
plt.tight_layout()

# Save the plot
plt.savefig('explore/running_speed.png')

print(f"Plot saved to explore/running_speed.png")
# Close the HDF5 file and the NWBHDF5IO object
h5_file.close()
io.close()
print("NWB file processing complete.")