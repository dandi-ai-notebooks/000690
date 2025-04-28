# Script to explore the structure of units (neural activity) data from a probe ecephys file
# This script explores basic properties of neural recordings and electrodes

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configure matplotlib to save rather than display
plt.style.use('default')
plt.rcParams['figure.figsize'] = (10, 6)

# Load a probe ecephys file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("\nBasic NWB file info:")
print(f"Session ID: {nwb.session_id}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Probe Description: {nwb.electrode_groups['probeA'].description}")
print(f"Probe Location: {nwb.electrode_groups['probeA'].location}")
print(f"Probe Sampling Rate: {nwb.electrode_groups['probeA'].device.sampling_rate} Hz")

# Convert electrodes to a DataFrame
electrodes_df = nwb.electrodes.to_dataframe()

# Print electrode information
print("\nElectrode information summary:")
print(f"Number of electrodes: {len(electrodes_df)}")
print(f"Number of valid electrodes: {sum(electrodes_df['valid_data'])}")
print(f"Electrode brain regions (locations): {electrodes_df['location'].unique()}")

# Create a plot of electrode positions
plt.figure(figsize=(10, 10))
# Color by brain region
unique_regions = electrodes_df['location'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
region_color_map = dict(zip(unique_regions, colors))

for region in unique_regions:
    region_electrodes = electrodes_df[electrodes_df['location'] == region]
    plt.scatter(
        region_electrodes['y'], 
        region_electrodes['probe_vertical_position'], 
        label=region,
        c=[region_color_map[region]],
        alpha=0.7
    )

plt.xlabel('AP Position (y, μm)')
plt.ylabel('Depth (probe_vertical_position, μm)')
plt.title('Electrode Positions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('explore/electrode_positions.png')
plt.close()

print("\nChecking for units data:")
if hasattr(nwb, 'units') and nwb.units is not None:
    # Get units information
    units_df = nwb.units.to_dataframe()
    print(f"Number of units (neurons): {len(units_df)}")
    print(f"Number of units by quality: {units_df['quality'].value_counts().to_dict()}")

    # Plot unit quality metrics
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(units_df['amplitude'], units_df['snr'], alpha=0.5)
    plt.xlabel('Spike Amplitude')
    plt.ylabel('Signal-to-Noise Ratio')
    plt.title('Amplitude vs SNR')

    plt.subplot(2, 2, 2)
    plt.scatter(units_df['firing_rate'], units_df['isi_violations'], alpha=0.5)
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('ISI Violations')
    plt.title('Firing Rate vs ISI Violations')

    plt.subplot(2, 2, 3)
    plt.hist(units_df['firing_rate'], bins=30)
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Count')
    plt.title('Distribution of Firing Rates')

    plt.subplot(2, 2, 4)
    plt.hist(units_df['snr'], bins=30)
    plt.xlabel('Signal-to-Noise Ratio')
    plt.ylabel('Count')
    plt.title('Distribution of SNR')

    plt.tight_layout()
    plt.savefig('explore/unit_quality_metrics.png')
    plt.close()

    # Get spike times for a subset of units to analyze temporal patterns
    spike_times_sample = {}
    max_units = 5
    good_units = units_df[units_df['quality'] == 'good'].iloc[:max_units]

    print(f"\nExamining spike times for {len(good_units)} good units:")
    for i, (_, unit) in enumerate(good_units.iterrows()):
        spike_times = nwb.units['spike_times'][i]
        spike_times_sample[i] = spike_times
        print(f"Unit {i}: {len(spike_times)} spikes, mean rate: {len(spike_times)/3600:.2f} Hz")

    # Plot spike raster for these units
    plt.figure(figsize=(15, 6))
    for i, (unit_id, spike_times) in enumerate(spike_times_sample.items()):
        # Get a sample of spike times (first 1000)
        sample_times = spike_times[:1000] if len(spike_times) > 1000 else spike_times
        plt.scatter(sample_times, np.ones_like(sample_times) * i, marker='|', s=20, linewidth=1)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Unit ID')
    plt.yticks(range(len(spike_times_sample)))
    plt.title('Spike Raster Plot (sample of spike times)')
    plt.savefig('explore/spike_raster.png')
    plt.close()
else:
    print("No units data available in this file. This may be an LFP-only file.")

# Let's examine the LFP data structure
print("\nLFP data structure:")
lfp = nwb.acquisition['probe_0_lfp']
lfp_data = lfp.electrical_series['probe_0_lfp_data']
print(f"LFP data shape: {lfp_data.data.shape}")
print(f"LFP sampling rate: {1 / (lfp_data.timestamps[1] - lfp_data.timestamps[0])} Hz")
print(f"LFP duration: {lfp_data.timestamps[-1] - lfp_data.timestamps[0]:.2f} seconds")

# Let's try different times and channels to find actual brain signals in the LFP data
def plot_lfp_sample(start_time, duration, channels, offset=200):
    """Plot a sample of LFP data from specified channels at a given time period."""
    sample_rate = 1 / (lfp_data.timestamps[1] - lfp_data.timestamps[0])
    num_samples = int(duration * sample_rate)
    
    # Find start index corresponding to start_time
    start_idx = np.argmin(np.abs(lfp_data.timestamps[:] - start_time))
    
    plt.figure(figsize=(15, 10))
    
    # Get different electrode regions to sample from different brain areas
    brain_regions = electrodes_df['location'].unique()
    electrodes_by_region = {}
    for region in brain_regions:
        # Get the actual indices in the dataframe, not just the position in the dataframe
        region_electrodes = electrodes_df[electrodes_df['location'] == region]
        electrodes_by_region[region] = region_electrodes.index.values
    
    # Choose channels from different regions if specified, otherwise use provided channels
    if channels == 'from_regions':
        channels_to_plot = []
        # Try to get one channel from each of 5 different regions
        regions_to_sample = brain_regions[:min(5, len(brain_regions))]
        for region in regions_to_sample:
            if len(electrodes_by_region[region]) > 0:
                # Take the middle electrode from each region
                channel_idx = electrodes_by_region[region][len(electrodes_by_region[region])//2]
                # Make sure channel is in range of the actual data
                if channel_idx < lfp_data.data.shape[1]:
                    channels_to_plot.append(channel_idx)
    else:
        channels_to_plot = channels
    
    for i, channel_idx in enumerate(channels_to_plot):
        # Get the data
        lfp_snippet = lfp_data.data[start_idx:start_idx+num_samples, channel_idx]
        
        # Apply a high-pass filter to remove DC offset and see if there's any signal
        from scipy.signal import butter, filtfilt
        def butter_highpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='high', analog=False)
            return b, a
        
        def highpass_filter(data, cutoff, fs, order=5):
            b, a = butter_highpass(cutoff, fs, order=order)
            y = filtfilt(b, a, data)
            return y
        
        # Apply high-pass filter to see if there's oscillatory content above 0.5 Hz
        lfp_filtered = highpass_filter(lfp_snippet, cutoff=0.5, fs=sample_rate)
        
        # Plot with offset
        region = electrodes_df.iloc[channel_idx]['location'] if channel_idx < len(electrodes_df) else 'Unknown'
        plt.plot(lfp_data.timestamps[start_idx:start_idx+num_samples], 
                 lfp_filtered + i*offset,  # Offset for visibility
                 label=f"Channel {channel_idx} ({region})")
    
    plt.xlabel('Time (s)')
    plt.ylabel('LFP Amplitude (μV) + offset')
    plt.title(f'LFP Signals (High-pass filtered) from {len(channels_to_plot)} Channels ({duration} second sample)')
    plt.legend(loc='upper right')
    
    # Create filename with time info
    filename = f'explore/lfp_sample_{start_time:.0f}s_to_{start_time+duration:.0f}s.png'
    plt.savefig(filename)
    plt.close()
    return filename

# Try a few different time periods to find interesting activity
plot_times = [
    (1000, 5),    # Early in the recording
    (4000, 5),    # Middle of the recording
    (8000, 5)     # Late in the recording
]

# Also try random channels from different brain regions
for start_time, duration in plot_times:
    filename = plot_lfp_sample(start_time, duration, 'from_regions', offset=1000)
    print(f"Plotted LFP sample at {start_time}s for {duration}s: {filename}")

# Also plot power spectrum of LFP data to see frequency content
plt.figure(figsize=(15, 8))
from scipy import signal

# Calculate power spectrum for a longer segment
segment_length = 60  # seconds
sample_rate = 1 / (lfp_data.timestamps[1] - lfp_data.timestamps[0])
segment_samples = int(segment_length * sample_rate)
start_idx = len(lfp_data.timestamps) // 2  # Middle of recording
end_idx = min(start_idx + segment_samples, len(lfp_data.timestamps))

# Choose a few channels from different regions
regions_to_sample = brain_regions[:min(5, len(brain_regions))]
channels = []
for region in regions_to_sample:
    if len(electrodes_by_region[region]) > 0:
        channel_idx = electrodes_by_region[region][len(electrodes_by_region[region])//2]
        # Make sure channel is in range of the actual data
        if channel_idx < lfp_data.data.shape[1]:
            channels.append(channel_idx)

# Plot power spectrum for selected channels
for channel_idx in channels:
    # Get the data
    region = electrodes_df.iloc[channel_idx]['location'] if channel_idx < len(electrodes_df) else 'Unknown'
    lfp_segment = lfp_data.data[start_idx:end_idx, channel_idx]
    
    # Apply Hanning window to reduce edge effects
    lfp_segment = lfp_segment * np.hanning(len(lfp_segment))
    
    # Calculate power spectrum
    f, Pxx = signal.welch(lfp_segment, fs=sample_rate, nperseg=int(sample_rate * 4), 
                          scaling='density', detrend='linear')
    
    # Plot (limit to physiological range 0-250 Hz)
    mask = (f >= 0) & (f <= 250)
    plt.semilogy(f[mask], Pxx[mask], label=f"Channel {channel_idx} ({region})")

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (μV²/Hz)')
plt.title(f'LFP Power Spectrum (60s segment from middle of recording)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.savefig('explore/lfp_power_spectrum.png')
plt.close()
print("Plotted LFP power spectrum: explore/lfp_power_spectrum.png")

print("\nAnalysis complete - see output plots in the explore/ directory")