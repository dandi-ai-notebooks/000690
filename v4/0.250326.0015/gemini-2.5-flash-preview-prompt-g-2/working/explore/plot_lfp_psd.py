# Script to plot power spectral density of LFP data for a few channels

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import welch

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data
lfp_data = nwb.acquisition["probe_0_lfp_data"]

# Determine a subset of data to load (e.g., 60 seconds from a later time point for the first 5 channels)
sampling_rate = nwb.electrode_groups['probeA'].lfp_sampling_rate
data_shape = lfp_data.data.shape
start_time_point = int(data_shape[0] / 2) # Start from the middle of the recording
num_timepoints_to_load = int(sampling_rate * 60) # Load 60 seconds of data
num_channels_to_load = 5

# Ensure we don't go out of bounds
if start_time_point + num_timepoints_to_load > data_shape[0]:
    num_timepoints_to_load = data_shape[0] - start_time_point

# Access the subset of data
lfp_subset = lfp_data.data[start_time_point : start_time_point + num_timepoints_to_load, 0:num_channels_to_load]

# Calculate Power Spectral Density using Welch's method
freqs, psd = welch(lfp_subset, fs=sampling_rate, nperseg=1024, axis=0)

# Plot PSD
sns.set_theme()
plt.figure(figsize=(10, 6))
for i in range(num_channels_to_load):
    plt.semilogy(freqs, psd[:, i], label=f'Channel {i}')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Power Spectral Density of LFP')
plt.legend()
plt.grid(True)
plt.savefig('explore/lfp_psd.png')
# plt.show() # Do not show the plot to avoid hanging

io.close() # Close the NWB file