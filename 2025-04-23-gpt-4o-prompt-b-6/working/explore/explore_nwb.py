"""
This script loads data from the NWB file and plots LFP data for the first 10 channels over a selected time interval.
The resulting plot is saved to a PNG file for review.
"""

import matplotlib.pyplot as plt
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Load data
lfp_data = h5_file['acquisition']['probe_0_lfp']['probe_0_lfp_data']['data']
timestamps = h5_file['acquisition']['probe_0_lfp']['probe_0_lfp_data']['timestamps']

# Select a subset of data for plotting
select_channels = range(10)  # First 10 channels
select_time = slice(0, 1000) # First 1000 samples
data_subset = lfp_data[select_time, select_channels]
time_subset = timestamps[select_time]

# Plotting
plt.figure(figsize=(12, 6))
for i, channel in enumerate(select_channels):
    plt.plot(time_subset, data_subset[:, i] + (i * 0.1), label=f'Channel {channel}')  # Offset for clarity

plt.xlabel('Time (s)')
plt.ylabel('LFP Signal (Offset for clarity)')
plt.title('LFP Signals for First 10 Channels')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.savefig('explore/lfp_first_10_channels.png')
plt.close()