# This script explores the LFP (Local Field Potential) data from one probe
# We want to study the neural activity patterns and visualize the data

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory for plots
os.makedirs('explore', exist_ok=True)

# Load the probe0 ecephys NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
print(f"Loading probe-0 ecephys NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the file
print(f"Session ID: {nwb.session_id}")
print(f"Session description: {nwb.session_description}")
print(f"Institution: {nwb.institution}")

# Get information about the LFP data
probe_lfp = nwb.acquisition['probe_0_lfp_data']
print(f"\nLFP data shape: {probe_lfp.data.shape}")
print(f"Number of electrodes: {probe_lfp.data.shape[1]}")
print(f"Number of time points: {probe_lfp.data.shape[0]}")
print(f"Data unit: {probe_lfp.unit}")

# Get electrode information
electrodes_table = nwb.electrodes
print(f"\nNumber of electrodes in table: {len(electrodes_table.id)}")

# Try to access location data for electrodes
try:
    locations = electrodes_table.location[:]
    unique_locations = np.unique(locations)
    print(f"\nUnique brain regions recorded from: {len(unique_locations)}")
    for i, loc in enumerate(unique_locations):
        print(f"{i+1}. {loc}")
except Exception as e:
    print(f"Could not access electrode locations: {e}")

# Plot a short segment of LFP data for a few channels
segment_size = 5000  # Data points to plot
num_channels_to_plot = 5  # Number of channels to plot
start_time = 10000  # Starting point

# Select channels at regular intervals
channel_indices = np.linspace(0, probe_lfp.data.shape[1]-1, num_channels_to_plot, dtype=int)

plt.figure(figsize=(12, 8))
for i, ch_idx in enumerate(channel_indices):
    # Offset each channel for visibility
    offset = i * 200  # Microvolts offset for visualization
    data_segment = probe_lfp.data[start_time:start_time+segment_size, ch_idx]
    plt.plot(probe_lfp.timestamps[start_time:start_time+segment_size], 
             data_segment - np.mean(data_segment) + offset, 
             label=f'Channel {ch_idx}')

plt.title('LFP Traces from Different Channels')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV) + offset')
plt.legend()
plt.grid(True)
plt.savefig('explore/lfp_traces.png')
plt.close()

# Plot a spectrogram for one channel to see frequency content
from scipy import signal

# Select a single channel
channel_to_analyze = channel_indices[0]
data_to_analyze = probe_lfp.data[:, channel_to_analyze]

# Take a longer segment for spectral analysis
segment_length = 30000  # 30 seconds if sampled at 1000 Hz
fs = 1250.0  # Sampling frequency in Hz

# Compute spectrogram
f, t, Sxx = signal.spectrogram(data_to_analyze[:segment_length], fs=fs, nperseg=256, noverlap=128)

# Plot spectrogram
plt.figure(figsize=(12, 8))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.title(f'Spectrogram of LFP Channel {channel_to_analyze}')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.ylim(0, 100)  # Limit to 0-100 Hz range
plt.savefig('explore/lfp_spectrogram.png')
plt.close()

# Compute power spectral density for the same channel
f, Pxx = signal.welch(data_to_analyze[:segment_length], fs=fs, nperseg=1024, noverlap=512)

plt.figure(figsize=(10, 6))
plt.semilogy(f, Pxx)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title(f'Power Spectral Density of Channel {channel_to_analyze}')
plt.xlim(0, 100)
plt.grid(True)
plt.savefig('explore/lfp_psd.png')
plt.close()

# Plot average power for specific frequency bands across channels
def bandpower(data, fs, fmin, fmax):
    """Calculate power in specific frequency band."""
    f, Pxx = signal.welch(data, fs=fs, nperseg=1024, noverlap=512)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

# Define frequency bands
bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 100)
}

# Sample a subset of channels to speed up computation
num_sample_channels = 20
sample_channels = np.linspace(0, probe_lfp.data.shape[1]-1, num_sample_channels, dtype=int)

band_powers = {band: [] for band in bands}

for ch_idx in sample_channels:
    data_segment = probe_lfp.data[:20000, ch_idx]  # Use first 20000 samples
    for band, (fmin, fmax) in bands.items():
        power = bandpower(data_segment, fs, fmin, fmax)
        band_powers[band].append(power)

# Plot band powers across channels
plt.figure(figsize=(12, 8))
x = np.arange(len(sample_channels))
width = 0.15
multiplier = 0

for band, powers in band_powers.items():
    normalized_powers = powers / np.max(powers)  # Normalize to make comparable
    offset = width * multiplier
    plt.bar(x + offset, normalized_powers, width, label=band)
    multiplier += 1

plt.xlabel('Channel Index')
plt.ylabel('Normalized Band Power')
plt.title('Frequency Band Power Distribution Across Channels')
plt.xticks(x + width * 2, sample_channels)
plt.legend(loc='upper left')
plt.savefig('explore/band_powers.png')
plt.close()

print("LFP data exploration completed. Check explore directory for plots.")