"""
This script explores the LFP data from the probe_0_ecephys.nwb file to understand:
1. The structure of the LFP data
2. The electrode locations
3. Sample waveforms from the LFP data
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb

# Save figures with seaborn style
import seaborn as sns
sns.set_theme()

# Set the URL for the probe_0 data
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"

# Load the data
print("Loading data from remote NWB file...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("NWB file loaded. Exploring data structure...")

# Basic information
print(f"Session ID: {nwb.session_id}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")

# Get electrode information
electrodes_df = nwb.electrodes.to_dataframe()
print(f"\nNumber of electrodes: {len(electrodes_df)}")
print("\nElectrode columns:")
for col in electrodes_df.columns:
    print(f"  - {col}")

print("\nSample of electrode data:")
print(electrodes_df.head())

# Get information about brain regions represented
brain_regions = electrodes_df['location'].unique()
print(f"\nBrain regions in recording: {brain_regions}")

# Get access to LFP data
probe_0_lfp = nwb.acquisition['probe_0_lfp']
probe_0_lfp_data = probe_0_lfp.electrical_series['probe_0_lfp_data']

print(f"\nLFP data shape: {probe_0_lfp_data.data.shape}")
print(f"LFP time points: {len(probe_0_lfp_data.timestamps)}")
print(f"LFP sampling rate: {1/(probe_0_lfp_data.timestamps[1] - probe_0_lfp_data.timestamps[0])} Hz")
print(f"LFP duration: {probe_0_lfp_data.timestamps[-1] - probe_0_lfp_data.timestamps[0]} seconds")

# Plot electrode locations
plt.figure(figsize=(10, 8))
plt.scatter(
    electrodes_df['probe_horizontal_position'], 
    electrodes_df['probe_vertical_position'],
    c=electrodes_df.index, 
    cmap='viridis', 
    alpha=0.8
)
plt.colorbar(label='Electrode index')
plt.xlabel('Horizontal position (µm)')
plt.ylabel('Vertical position (µm)')
plt.title('Probe electrode positions')
plt.grid(True)
plt.savefig('explore/electrode_positions.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot a sample of LFP data from 5 channels for 1 second
# Get a short segment of data (1 second) at the beginning
sample_length = 1250  # at 1250 Hz sampling rate (1 second)
start_offset = 10000  # Skip first 8 seconds which may have artifacts

# Select 5 channels evenly spaced through the probe
num_channels = probe_0_lfp_data.data.shape[1]
channel_indices = np.linspace(0, num_channels-1, 5, dtype=int)

# Get sample data
sample_data = probe_0_lfp_data.data[start_offset:start_offset+sample_length, channel_indices]
sample_time = probe_0_lfp_data.timestamps[start_offset:start_offset+sample_length]

# Plot sample LFP traces
plt.figure(figsize=(15, 10))
for i, channel_idx in enumerate(channel_indices):
    # Offset each trace for visibility
    offset = i * 0.0005  # Adjust based on data amplitude
    plt.plot(
        sample_time, 
        sample_data[:, i] + offset, 
        label=f'Channel {channel_idx}'
    )

plt.xlabel('Time (s)')
plt.ylabel('LFP (V)')
plt.title('Sample LFP traces from different channels')
plt.legend()
plt.grid(True)
plt.savefig('explore/lfp_sample_traces.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a heatmap of LFP activity across channels
plt.figure(figsize=(15, 8))
plt.imshow(
    sample_data.T, 
    aspect='auto',
    extent=[sample_time[0], sample_time[-1], 0, len(channel_indices)-1],
    origin='lower', 
    cmap='viridis'
)
plt.colorbar(label='LFP (V)')
plt.xlabel('Time (s)')
plt.ylabel('Channel index')
plt.title('LFP activity across channels')
channel_labels = [str(idx) for idx in channel_indices]
plt.yticks(range(len(channel_indices)), channel_labels)
plt.savefig('explore/lfp_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Script completed successfully!")