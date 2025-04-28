import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading NWB file...")

# Load the file
url = "https://api.dandiarchive.org/api/assets/ecaed1ec-a8b5-4fe7-87c1-baf68cfa900f/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the LFP data
probe_0_lfp_data = nwb.acquisition["probe_0_lfp_data"]

# Get electrode information as a dataframe
electrodes_df = nwb.electrodes.to_dataframe()
print("\nElectrode locations:", electrodes_df['location'].unique())

# Load a 1-second segment of data (625 samples since sampling rate is 625 Hz)
start_idx = 0
n_samples = 625
data = probe_0_lfp_data.data[start_idx:start_idx+n_samples, :]
timestamps = probe_0_lfp_data.timestamps[start_idx:start_idx+n_samples]

# Create figure with two subplots
fig = plt.figure(figsize=(15, 10))

# Plot 1: LFP traces
ax1 = plt.subplot(121)
# Plot a subset of channels for clarity
channels_to_plot = np.arange(0, 84, 4)  # Plot every 4th channel
offset = 0.0005  # Offset between channels for visualization

for i, chan_idx in enumerate(channels_to_plot):
    trace = data[:, chan_idx]
    ax1.plot(timestamps, trace + (i * offset), 'k-', linewidth=0.5, alpha=0.8)
    
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('LFP (V)')
ax1.set_title('LFP Traces (subset of channels)')

# Plot 2: Electrode positions
ax2 = plt.subplot(122)
scatter = ax2.scatter(electrodes_df['x'], 
                     electrodes_df['y'],
                     c=electrodes_df['probe_vertical_position'],
                     cmap='viridis',
                     s=50)
plt.colorbar(scatter, label='Probe Vertical Position (μm)')
ax2.set_xlabel('X Position (posterior +)')
ax2.set_ylabel('Y Position (inferior +)')
ax2.set_title('Electrode Positions')

plt.tight_layout()
plt.savefig('explore/lfp_overview.png')
plt.close()

# Print some basic statistics
print("\nData shape:", probe_0_lfp_data.data.shape)
print("Sampling rate:", nwb.electrode_groups['probeA'].lfp_sampling_rate, "Hz")
print("Recording duration:", len(probe_0_lfp_data.timestamps) / nwb.electrode_groups['probeA'].lfp_sampling_rate, "seconds")