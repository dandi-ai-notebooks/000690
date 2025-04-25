# Explore LFP data: Plot a short snippet from a few channels.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True) # Need load_namespaces=True
nwb = io.read()

# Get LFP data and timestamps
lfp_data = nwb.acquisition["probe_0_lfp"].electrical_series["probe_0_lfp_data"]
lfp_dataset = lfp_data.data
lfp_timestamps_dataset = lfp_data.timestamps

# Define time window and channels
start_index = 100000  # Start 160 seconds in (100000 / 625 Hz)
num_samples = 625     # ~1 second of data (sampling rate is 625 Hz)
channel_indices = [0, 1, 2] # First 3 channels

# Load data subset
lfp_snippet = lfp_dataset[start_index : start_index + num_samples, channel_indices]
time_snippet = lfp_timestamps_dataset[start_index : start_index + num_samples]

# Get channel IDs from the electrodes table
electrodes_df = nwb.electrodes.to_dataframe()
channel_ids = electrodes_df.index[channel_indices].tolist()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
for i, chan_idx in enumerate(channel_indices):
    # Offset traces for visualization
    offset = (np.max(lfp_snippet) - np.min(lfp_snippet)) * i * 1.5
    ax.plot(time_snippet, lfp_snippet[:, i] + offset, label=f'Channel {channel_ids[i]}')

ax.set_xlabel("Time (s)")
ax.set_ylabel("LFP (Volts + offset)")
ax.set_title(f"LFP Snippet (Channels {channel_ids})")
ax.legend(loc='upper right')
plt.tight_layout()

# Save plot
plt.savefig("explore/lfp_snippet.png")

print("Saved plot to explore/lfp_snippet.png")

# Close resources
io.close()
remote_file.close() # Important to close the remfile