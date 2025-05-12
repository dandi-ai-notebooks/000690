# This script explores neural responses to visual stimuli
# We'll analyze how neural activity relates to stimulus presentations

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

# Create output directory for plots
os.makedirs('explore', exist_ok=True)

# Load the main NWB file to get stimulus timing information
print("Loading main NWB file for stimulus information...")
main_url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(main_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
main_nwb = io.read()

# Load one of the ecephys files for neural data
print("Loading probe-0 ecephys NWB file...")
ecephys_url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file_ecephys = remfile.File(ecephys_url)
h5_file_ecephys = h5py.File(remote_file_ecephys)
io_ecephys = pynwb.NWBHDF5IO(file=h5_file_ecephys)
ecephys_nwb = io_ecephys.read()

# Get the LFP data
lfp_data = ecephys_nwb.acquisition['probe_0_lfp_data']
print(f"LFP data shape: {lfp_data.data.shape}")
print(f"LFP sampling rate: 625 Hz (assumed from probe documentation)")

# Check available stimulus presentation intervals
stim_intervals = [k for k in main_nwb.intervals.keys() if 'presentations' in k]
print(f"\nFound {len(stim_intervals)} stimulus presentation intervals")
for i, stim in enumerate(stim_intervals[:5]):  # Just show first 5
    print(f"{i+1}. {stim}")

# Select one stimulus type to analyze
# Looking for a stimulus with a reasonable number of presentations
selected_stim = None
for stim_name in stim_intervals:
    try:
        stim_data = main_nwb.intervals[stim_name]
        start_times = stim_data.start_time[:]
        if 100 <= len(start_times) <= 1000:  # A manageable number
            selected_stim = stim_name
            print(f"\nSelected stimulus: {selected_stim} with {len(start_times)} presentations")
            break
    except Exception:
        continue

# If no suitable stimulus found, pick the first one
if selected_stim is None:
    selected_stim = stim_intervals[0]
    stim_data = main_nwb.intervals[selected_stim]
    start_times = stim_data.start_time[:]
    print(f"\nFalling back to: {selected_stim} with {len(start_times)} presentations")

# Get stimulus presentation times
stim_data = main_nwb.intervals[selected_stim]
stimulus_starts = stim_data.start_time[:]
stimulus_stops = stim_data.stop_time[:]
stimulus_durations = stimulus_stops - stimulus_starts
print(f"Average stimulus duration: {np.mean(stimulus_durations):.3f} s")

# Function to extract LFP segments around stimulus onsets
def extract_lfp_around_events(lfp_data, event_times, pre_time=0.5, post_time=1.5, sampling_rate=625):
    """
    Extract LFP segments around specified events
    
    Parameters:
    -----------
    lfp_data : ndarray
        LFP data matrix (time x channels)
    event_times : array-like
        List of event times in seconds
    pre_time : float
        Time before event in seconds
    post_time : float
        Time after event in seconds
    sampling_rate : float
        LFP sampling rate in Hz
        
    Returns:
    --------
    segments : ndarray
        LFP segments (events x time x channels)
    time_axis : ndarray
        Time axis relative to event onset
    """
    pre_samples = int(pre_time * sampling_rate)
    post_samples = int(post_time * sampling_rate)
    segment_length = pre_samples + post_samples
    
    # Initialize array for segments
    num_events = len(event_times)
    num_channels = lfp_data.data.shape[1]
    segments = np.zeros((num_events, segment_length, num_channels))
    
    # Convert event times to sample indices
    lfp_timestamps = lfp_data.timestamps[:]
    total_samples = len(lfp_timestamps)
    
    # Extract segments for each event
    valid_segments = 0
    for i, event_time in enumerate(event_times):
        # Find closest timestamp
        event_idx = np.argmin(np.abs(lfp_timestamps - event_time))
        
        # Check if we have enough samples before and after
        if event_idx >= pre_samples and event_idx + post_samples < total_samples:
            start_idx = event_idx - pre_samples
            end_idx = event_idx + post_samples
            
            # Extract LFP segment
            segments[valid_segments] = lfp_data.data[start_idx:end_idx, :]
            valid_segments += 1
    
    # Trim array to valid segments
    segments = segments[:valid_segments]
    
    # Create time axis relative to event onset
    time_axis = np.linspace(-pre_time, post_time, segment_length)
    
    print(f"Extracted {valid_segments} valid segments out of {num_events} events")
    return segments, time_axis

# Extract LFP segments around stimulus onsets
# Use just the first 100 stimulus presentations to keep computation manageable
num_stimuli_to_use = min(100, len(stimulus_starts))
segments, time_axis = extract_lfp_around_events(
    lfp_data, 
    stimulus_starts[:num_stimuli_to_use], 
    pre_time=0.5,   # 500 ms before stimulus
    post_time=1.5,  # 1500 ms after stimulus
    sampling_rate=625
)

# Select a subset of channels to analyze
num_channels_to_plot = 5
channel_indices = np.linspace(0, lfp_data.data.shape[1]-1, num_channels_to_plot, dtype=int)

# Plot average LFP response for selected channels
plt.figure(figsize=(12, 8))
for i, ch_idx in enumerate(channel_indices):
    # Calculate mean and SEM across events
    mean_response = np.mean(segments[:, :, ch_idx], axis=0)
    sem_response = np.std(segments[:, :, ch_idx], axis=0) / np.sqrt(segments.shape[0])
    
    # Plot with shaded error region
    offset = i * 100  # Offset for visualization
    plt.plot(time_axis, mean_response + offset, label=f'Channel {ch_idx}')
    plt.fill_between(
        time_axis, 
        mean_response - sem_response + offset, 
        mean_response + sem_response + offset, 
        alpha=0.3
    )

plt.axvline(x=0, color='k', linestyle='--', label='Stimulus onset')
plt.axvline(x=np.mean(stimulus_durations), color='r', linestyle='--', label='Avg stimulus offset')
plt.title(f'Average LFP Response to {selected_stim.split("_")[0]} Stimulus')
plt.xlabel('Time relative to stimulus onset (s)')
plt.ylabel('LFP amplitude (μV) + offset')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('explore/average_lfp_response.png')
plt.close()

# Calculate time-frequency representation of the response
# Select one channel for time-frequency analysis
tf_channel = channel_indices[0]

# Initialize array for time-frequency results
freq_range = (1, 100)  # 1-100 Hz
nperseg = 128
noverlap = nperseg // 2
freqs = np.fft.rfftfreq(nperseg, 1/625)
freqs = freqs[(freqs >= freq_range[0]) & (freqs <= freq_range[1])]
freq_indices = np.where((np.fft.rfftfreq(nperseg, 1/625) >= freq_range[0]) & 
                       (np.fft.rfftfreq(nperseg, 1/625) <= freq_range[1]))[0]

# Calculate spectrograms for each trial and average
all_specs = []
for i in range(segments.shape[0]):
    f, t, Sxx = signal.spectrogram(
        segments[i, :, tf_channel], 
        fs=625, 
        nperseg=nperseg, 
        noverlap=noverlap
    )
    # Extract relevant frequencies
    Sxx_subset = Sxx[freq_indices, :]
    all_specs.append(Sxx_subset)

# Average spectrograms across trials
avg_spec = np.mean(all_specs, axis=0)
# Convert to dB
avg_spec_db = 10 * np.log10(avg_spec)

# Plot time-frequency representation
plt.figure(figsize=(12, 8))
plt.pcolormesh(
    time_axis[nperseg//2:-nperseg//2:nperseg-noverlap], 
    freqs, 
    avg_spec_db, 
    shading='gouraud'
)
plt.colorbar(label='Power (dB)')
plt.axvline(x=0, color='k', linestyle='--', label='Stimulus onset')
plt.axvline(x=np.mean(stimulus_durations), color='r', linestyle='--', label='Avg stimulus offset')
plt.title(f'Average Time-Frequency Response to {selected_stim.split("_")[0]} Stimulus (Ch {tf_channel})')
plt.xlabel('Time relative to stimulus onset (s)')
plt.ylabel('Frequency (Hz)')
plt.ylim(1, 100)
plt.legend()
plt.savefig('explore/time_frequency_response.png')
plt.close()

# Calculate power in specific frequency bands over time
bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 80)
}

# Function to calculate band power from spectrogram
def get_band_power(spec, freqs, band_range):
    """Extract power in a specific frequency band from spectrogram"""
    band_mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    return np.mean(spec[band_mask, :], axis=0)

# Calculate power for each band
plt.figure(figsize=(12, 8))
for band_name, band_range in bands.items():
    band_power = get_band_power(avg_spec, freqs, band_range)
    # Normalize to baseline
    baseline = np.mean(band_power[:int(0.5/(time_axis[1]-time_axis[0]))])
    norm_power = band_power / baseline
    
    # Plot
    plt.plot(
        time_axis[nperseg//2:-nperseg//2:nperseg-noverlap], 
        norm_power, 
        label=f'{band_name} ({band_range[0]}-{band_range[1]} Hz)'
    )

plt.axvline(x=0, color='k', linestyle='--')
plt.axvline(x=np.mean(stimulus_durations), color='r', linestyle='--')
plt.axhline(y=1, color='gray', linestyle='-', alpha=0.5)
plt.title(f'Frequency Band Power Relative to Baseline (Ch {tf_channel})')
plt.xlabel('Time relative to stimulus onset (s)')
plt.ylabel('Normalized power')
plt.legend()
plt.grid(True)
plt.savefig('explore/band_power_response.png')
plt.close()

print("Neural response exploration completed. Check explore directory for plots.")