# This script explores LFP (Local Field Potential) data from one of the subjects
# to understand neural activity patterns

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb

# Save all plots to file instead of displaying
plt.ioff()

# Load an ecephys file (probe-specific NWB file with LFP data)
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
print(f"Loading NWB file from {url}")

# Open as a remote file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic info about the file
print(f"Session ID: {nwb.session_id}")
print(f"Session description: {nwb.session_description}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject age: {nwb.subject.age}")
print(f"Subject species: {nwb.subject.species}")

# List available electrodes
print("\nElectrode information:")
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Total number of electrodes: {len(electrodes_df)}")
print(f"Brain regions represented: {electrodes_df['location'].unique()}")
print(f"Electrodes per brain region:")
print(electrodes_df['location'].value_counts())

# Get LFP data
lfp_data = nwb.acquisition['probe_0_lfp_data']
print(f"\nLFP data shape: {lfp_data.data.shape}")  # [timestamps, channels]
print(f"LFP units: {lfp_data.unit}")
print(f"LFP sampling rate: {nwb.electrode_groups['probeA'].lfp_sampling_rate} Hz")
print(f"LFP duration: {lfp_data.data.shape[0] / nwb.electrode_groups['probeA'].lfp_sampling_rate:.2f} seconds")

# Plot example LFP traces from different brain regions
# We'll sample a subset of the data to make it manageable
sample_rate = 625  # Hz (from electrode_groups['probeA'].lfp_sampling_rate)
duration_sec = 10  # sample 10 seconds
n_samples = int(duration_sec * sample_rate)

# Get a random starting point that's at least 10 seconds from the end
max_start_idx = lfp_data.data.shape[0] - n_samples
start_idx = np.random.randint(0, max_start_idx)

# Get a subset of the timestamps
timestamps = lfp_data.timestamps[start_idx:start_idx + n_samples]
t_seconds = timestamps - timestamps[0]  # make zero-based

print(f"\nPlotting {duration_sec} seconds of LFP data starting from index {start_idx}")

# Get unique brain regions and sample one electrode from each region
unique_regions = electrodes_df['location'].unique()

# To avoid plots being too crowded, limit to a sample of regions if there are many
if len(unique_regions) > 5:
    np.random.seed(42)  # for reproducibility
    region_sample = np.random.choice(unique_regions, size=5, replace=False)
else:
    region_sample = unique_regions

fig, axs = plt.subplots(len(region_sample), 1, figsize=(12, 10), sharex=True)
if len(region_sample) == 1:
    axs = [axs]  # Make it iterable if there's only one subplot
    
for i, region in enumerate(region_sample):
    # Get electrode indices for this region
    region_electrodes = electrodes_df[electrodes_df['location'] == region].index.tolist()
    
    if not region_electrodes:  # Skip if no electrodes in this region
        continue
        
    # Pick the first electrode from this region
    electrode_idx = region_electrodes[0]
    # Extract LFP data for this electrode
    lfp_trace = lfp_data.data[start_idx:start_idx + n_samples, electrode_idx]
    
    # Plot
    axs[i].plot(t_seconds, lfp_trace)
    axs[i].set_ylabel(f'LFP ({lfp_data.unit})')
    axs[i].set_title(f'Region: {region}, Electrode: {electrode_idx}')
    
axs[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('explore/lfp_traces_by_region.png')
plt.close()

# Plot a heatmap of LFP activity across all channels for a short period
n_channels_to_plot = min(60, lfp_data.data.shape[1])  # Limit to 60 channels for visibility
plt.figure(figsize=(12, 8))

# Get a short time window
short_time = 2  # 2 seconds
short_samples = int(short_time * sample_rate)
lfp_short = lfp_data.data[start_idx:start_idx + short_samples, :n_channels_to_plot]

# Normalize data for better visualization
lfp_norm = (lfp_short - np.mean(lfp_short)) / np.std(lfp_short)
plt.imshow(lfp_norm.T, aspect='auto', interpolation='none', 
           extent=[0, short_time, n_channels_to_plot, 0])
plt.colorbar(label='Normalized LFP')
plt.xlabel('Time (seconds)')
plt.ylabel('Channel Number')
plt.title('LFP Activity Heatmap Across Channels')
plt.savefig('explore/lfp_heatmap.png')
plt.close()

# Calculate and plot power spectral density for selected channels
from scipy import signal

# We'll analyze PSD for one channel from each brain region
plt.figure(figsize=(12, 8))
for i, region in enumerate(region_sample):
    # Get electrode indices for this region
    region_electrodes = electrodes_df[electrodes_df['location'] == region].index.tolist()
    
    if not region_electrodes:  # Skip if no electrodes in this region
        continue
        
    # Pick the first electrode from this region
    electrode_idx = region_electrodes[0]
    
    # Get a longer segment for frequency analysis
    freq_duration_sec = 30  # 30 seconds will give better frequency resolution
    n_freq_samples = int(freq_duration_sec * sample_rate)
    
    # Make sure we don't exceed data bounds
    freq_start_idx = min(start_idx, max_start_idx - n_freq_samples)
    lfp_segment = lfp_data.data[freq_start_idx:freq_start_idx + n_freq_samples, electrode_idx]
    
    # Calculate PSD using Welch's method
    f, Pxx = signal.welch(lfp_segment, fs=sample_rate, nperseg=1024)
    
    # Plot only frequencies up to 100 Hz (typical for LFP analysis)
    mask = f <= 100
    plt.semilogy(f[mask], Pxx[mask], label=f'Region: {region}')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density ($V^2/Hz$)')
plt.title('Power Spectral Density of LFP Signals')
plt.legend()
plt.grid(True)
plt.savefig('explore/lfp_psd.png')
plt.close()

# Try to estimate the timing of stimulus presentation events
# This will help us understand if we can correlate neural activity with stimuli
try:
    # Check if stimulus information is available
    if hasattr(nwb, 'intervals') and len(nwb.intervals) > 0:
        # Print the names of available intervals
        print("\nAvailable intervals:")
        for interval_name in nwb.intervals:
            print(f"- {interval_name}")
            
            # Try to get some example stimulus times
            try:
                interval_df = nwb.intervals[interval_name].to_dataframe()
                if len(interval_df) > 0:
                    print(f"  - Number of intervals: {len(interval_df)}")
                    print(f"  - First 3 start times: {interval_df['start_time'].iloc[:3].values}")
                    print(f"  - First 3 stop times: {interval_df['stop_time'].iloc[:3].values}")
                    
                    # Plot a histogram of interval durations
                    durations = interval_df['stop_time'] - interval_df['start_time']
                    plt.figure(figsize=(10, 6))
                    plt.hist(durations, bins=30)
                    plt.xlabel('Interval Duration (seconds)')
                    plt.ylabel('Count')
                    plt.title(f'Distribution of {interval_name} Durations')
                    plt.savefig(f'explore/stimulus_durations_{interval_name.replace(" ", "_")}.png')
                    plt.close()
                
                    # Only process one interval for brevity
                    break
            except Exception as e:
                print(f"  - Error reading interval data: {e}")
except Exception as e:
    print(f"Error accessing interval data: {e}")
    
print("\nCompleted LFP data exploration")