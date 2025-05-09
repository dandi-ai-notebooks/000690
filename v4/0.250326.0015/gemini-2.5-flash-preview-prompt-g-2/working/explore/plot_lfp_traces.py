# Script to plot a subset of LFP traces

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data and timestamps
lfp_data = nwb.acquisition["probe_0_lfp_data"]
timestamps = lfp_data.timestamps

# Determine a subset of data to load (e.g., 1 second from the beginning for the first 5 channels)
# LFP sampling rate is 625 Hz (from electrode_groups['probeA'].lfp_sampling_rate)
sampling_rate = nwb.electrode_groups['probeA'].lfp_sampling_rate
num_timepoints_to_load = int(sampling_rate * 1) # Load 1 second of data
num_channels_to_load = 5

# Access the subset of data and timestamps using slicing
lfp_subset = lfp_data.data[0:num_timepoints_to_load, 0:num_channels_to_load]
timestamps_subset = timestamps[0:num_timepoints_to_load]

# Plot LFP traces
sns.set_theme()
plt.figure(figsize=(12, 8))
for i in range(num_channels_to_load):
    plt.plot(timestamps_subset, lfp_subset[:, i] + i * 500, label=f'Channel {i}') # Add offset for clarity

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (arbitrary units)') # Units are technically volts, but relative amplitude is more informative here
plt.title('Subset of LFP Traces')
plt.legend()
plt.savefig('explore/lfp_traces_subset.png')
# plt.show() # Do not show the plot to avoid hanging

io.close() # Close the NWB file