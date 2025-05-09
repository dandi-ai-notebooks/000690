# Explore pupil tracking data
# This script loads pupil tracking data (x, y coordinates) and corresponding timestamps
# from the NWB file and plots them. It saves the plot to explore/pupil_tracking_xy.png.

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

# Access pupil tracking data
pupil_tracking = nwb.acquisition['EyeTracking'].spatial_series['pupil_tracking']
timestamps = pupil_tracking.timestamps[:]  # Load all timestamps
pupil_data_xy = pupil_tracking.data[:]  # Load all x, y data

# Select a subset of data to plot (e.g., first 1000 points)
num_points_to_plot = 1000
timestamps_subset = timestamps[:num_points_to_plot]
pupil_data_xy_subset = pupil_data_xy[:num_points_to_plot, :]

# Create plot
sns.set_theme()
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(timestamps_subset, pupil_data_xy_subset[:, 0], label='Pupil X')
ax.plot(timestamps_subset, pupil_data_xy_subset[:, 1], label='Pupil Y')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position (meters)')
ax.set_title(f'Pupil Tracking Data (First {num_points_to_plot} points)')
ax.legend()
plt.tight_layout()

# Save the plot
plt.savefig('explore/pupil_tracking_xy.png')

print(f"Plot saved to explore/pupil_tracking_xy.png")
# Close the HDF5 file and the NWBHDF5IO object
h5_file.close()
io.close()
print("NWB file processing complete.")