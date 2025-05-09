# Script to calculate standard deviation of LFP data for a few channels

import pynwb
import h5py
import remfile
import numpy as np

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data
lfp_data = nwb.acquisition["probe_0_lfp_data"]

# Determine a subset of data to load (e.g., 30 seconds from the beginning for the first 5 channels)
sampling_rate = nwb.electrode_groups['probeA'].lfp_sampling_rate
num_timepoints_to_load = int(sampling_rate * 30) # Load 30 seconds of data
num_channels_to_load = 5

# Access the subset of data
lfp_subset = lfp_data.data[0:num_timepoints_to_load, 0:num_channels_to_load]

# Calculate standard deviation for each channel in the subset
std_devs = np.std(lfp_subset, axis=0)

print("Standard deviations of LFP data for the first 5 channels (30 seconds):")
for i, std_dev in enumerate(std_devs):
    print(f"Channel {i}: {std_dev}")

io.close() # Close the NWB file