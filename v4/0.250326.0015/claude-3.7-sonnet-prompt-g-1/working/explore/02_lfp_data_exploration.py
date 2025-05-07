# This script explores LFP (Local Field Potential) data from one of the probes
# to understand the neural recordings during the experiment

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# URL for the probe 0 file which contains LFP data
url = "https://api.dandiarchive.org/api/assets/59aa163a-187a-414e-ba78-01163c2a789b/download/"
print(f"Loading NWB file from {url}")
print("This might take a moment as we're accessing a remote file...")

# Load the file with a more efficient approach - only access what we need
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Basic info
print(f"\nProbe LFP file loaded.")
print(f"Session ID: {nwb.session_id}")
print(f"Session description: {nwb.session_description}")

# Get acquisition data - specifically LFP
print("\nAccessing LFP data...")
lfp = nwb.acquisition['probe_0_lfp']
lfp_data = lfp.electrical_series['probe_0_lfp_data']

# Get basic details about the LFP data
num_channels = lfp_data.data.shape[1]
num_timepoints = lfp_data.data.shape[0]
sampling_rate = 1000.0 / np.median(np.diff(lfp_data.timestamps[:1000])) # estimate from first 1000 samples
total_duration = (lfp_data.timestamps[-1] - lfp_data.timestamps[0])

print(f"Number of channels: {num_channels}")
print(f"Number of timepoints: {num_timepoints}")
print(f"Sampling rate (estimated): {sampling_rate:.2f} Hz")
print(f"Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")

# Extract information about the electrodes/channels
electrodes_table = lfp_data.electrodes.table
electrodes_info = electrodes_table.to_dataframe()
print(f"\nElectrode locations: {electrodes_info['location'].unique()}")

# Get a short segment of data for visualization (5 seconds from a point 60s into recording)
# This is to avoid potential startup artifacts
start_time = 60  # seconds into recording
segment_duration = 5  # seconds
start_idx = int(start_time * sampling_rate)
end_idx = start_idx + int(segment_duration * sampling_rate)

# Make sure we don't exceed data bounds
if end_idx > num_timepoints:
    end_idx = num_timepoints
    print(f"Warning: Requested segment exceeds data bounds. Adjusting to end of data.")

# Select a subset of channels to visualize (every 10th channel)
channel_step = 10
channels_to_plot = list(range(0, num_channels, channel_step))
num_plot_channels = len(channels_to_plot)

print(f"\nExtracting data segment from t={start_time}s to t={start_time + segment_duration}s")
print(f"Plotting {num_plot_channels} channels (every {channel_step}th channel)")

try:
    # Extract the timestamps for this segment
    timestamps = lfp_data.timestamps[start_idx:end_idx]
    
    # Extract data for selected channels and time segment
    data_segment = lfp_data.data[start_idx:end_idx, channels_to_plot]
    
    # Create plots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(num_plot_channels, 1, figure=fig)
    
    # Plot individual channels
    for i, channel_idx in enumerate(channels_to_plot):
        ax = fig.add_subplot(gs[i, 0])
        ax.plot(timestamps - timestamps[0], data_segment[:, i], linewidth=0.8)
        
        # Label only every few channels for clarity
        if i % 3 == 0:
            ax.set_ylabel(f"Ch {channel_idx}")
        
        # Remove x labels except for bottom subplot
        if i < num_plot_channels - 1:
            ax.set_xticks([])
        
    # Add x-axis label to the bottom subplot
    ax.set_xlabel("Time (seconds)")
    
    plt.suptitle(f"LFP Data from Probe 0 - Sample Segment (t = {start_time}-{start_time + segment_duration}s)")
    plt.tight_layout()
    plt.savefig('explore/lfp_sample.png', dpi=300)
    print("Plot saved to explore/lfp_sample.png")
    
except Exception as e:
    print(f"Error during data extraction or plotting: {str(e)}")

# Create a spectrogram (time-frequency analysis) for one channel
try:
    print("\nCreating spectrogram for one channel...")
    # Select middle channel for spectrogram
    middle_channel = num_channels // 2
    
    # Get longer segment for better frequency resolution (30 seconds)
    spec_duration = min(30, total_duration-start_time)
    spec_end_idx = start_idx + int(spec_duration * sampling_rate)
    
    spec_timestamps = lfp_data.timestamps[start_idx:spec_end_idx]
    spec_data = lfp_data.data[start_idx:spec_end_idx, middle_channel]
    
    plt.figure(figsize=(15, 8))
    
    # Calculate and plot spectrogram
    plt.subplot(211)
    plt.title(f"LFP Data - Channel {middle_channel} (Time Domain)")
    plt.plot(spec_timestamps - spec_timestamps[0], spec_data, linewidth=0.5)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (V)")
    
    plt.subplot(212)
    plt.title(f"LFP Data - Channel {middle_channel} (Spectrogram)")
    # Use specgram for time-frequency analysis
    Pxx, freqs, bins, im = plt.specgram(spec_data, NFFT=1024, Fs=sampling_rate, 
                                        noverlap=512, cmap='viridis')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 100)  # Focus on frequencies up to 100 Hz which are most relevant for LFP
    plt.colorbar(label="Power (dB)")
    
    plt.tight_layout()
    plt.savefig('explore/lfp_spectrogram.png', dpi=300)
    print("Spectrogram saved to explore/lfp_spectrogram.png")
    
except Exception as e:
    print(f"Error creating spectrogram: {str(e)}")

print("\nExploration complete!")