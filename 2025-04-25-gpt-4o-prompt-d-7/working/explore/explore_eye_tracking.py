# This script explores EyeTracking data from the NWB file, specifically looking at corneal reflection tracking data.
# It loads the NWB file from a remote location and generates a plot of the data output in the explore directory.

import matplotlib.pyplot as plt
import numpy as np
import pynwb
import h5py
import remfile

# Connect to NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Accessing corneal reflection tracking data
corneal_reflection_tracking = nwb.acquisition["EyeTracking"].spatial_series["corneal_reflection_tracking"]

# Slice the first 1000 data points for visualization
data = corneal_reflection_tracking.data[:1000, :]

plt.figure(figsize=(10, 5))
plt.plot(data[:, 0], label='x-coordinate')
plt.plot(data[:, 1], label='y-coordinate')
plt.title('Corneal Reflection Tracking (First 1000 points)')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Position (meters)')
plt.legend()

# Save the plot as PNG in the explore directory
plt.savefig('explore/corneal_reflection_tracking.png')

# Close io and file handles properly
io.close()
h5_file.close()
remote_file.close()