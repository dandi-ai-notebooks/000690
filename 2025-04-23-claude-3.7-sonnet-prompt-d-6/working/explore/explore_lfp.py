"""
This script explores the Local Field Potential (LFP) data from Dandiset 000690.
It loads an LFP file for one probe and analyzes the signals in relation to visual stimuli.
"""

import numpy as np
import h5py
import remfile
import pynwb
import matplotlib.pyplot as plt
import pandas as pd
import time

# Set the start time to measure execution
start_time = time.time()

# URL of the LFP file (probe 0)
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"

print("Loading NWB file from URL...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"File loaded in {time.time() - start_time:.2f} seconds")
print(f"Dandiset identifier: {nwb.identifier}")
print(f"Session description: {nwb.session_description}")

# Get information about electrodes
print("\nExtracting electrode information...")
electrodes_df = nwb.electrodes.to_dataframe()
print(f"Total number of electrodes: {len(electrodes_df)}")

# Print some information about the electrodes
print("\nElectrode locations:")
location_counts = electrodes_df['location'].value_counts()
print(location_counts)

# Get the LFP data
print("\nExtracting LFP data...")
lfp_data = nwb.acquisition['probe_0_lfp_data']
print(f"LFP data dimensions: {lfp_data.data.shape}")
print(f"LFP sampling rate: {nwb.electrode_groups['probeA'].lfp_sampling_rate} Hz")
print(f"LFP unit: {lfp_data.unit}")
print(f"Number of time points: {len(lfp_data.timestamps)}")
print(f"Number of channels: {lfp_data.data.shape[1]}")

# Get a short segment of LFP data for visualization (5 seconds)
# Starting from 100 second mark to avoid initial artifacts
start_time_sec = 100
duration_sec = 5
sampling_rate = nwb.electrode_groups['probeA'].lfp_sampling_rate

start_idx = int(start_time_sec * sampling_rate)
end_idx = start_idx + int(duration_sec * sampling_rate)

print(f"\nExtracting LFP segment from {start_time_sec}s to {start_time_sec + duration_sec}s...")
try:
    lfp_segment = lfp_data.data[start_idx:end_idx, :]
    timestamps_segment = lfp_data.timestamps[start_idx:end_idx]
    
    print(f"Segment dimensions: {lfp_segment.shape}")
    
    # Plot LFP traces for a few channels
    channels_to_plot = 5  # Number of channels to plot
    channel_indices = np.linspace(0, lfp_data.data.shape[1]-1, channels_to_plot, dtype=int)
    
    plt.figure(figsize=(12, 10))
    for i, ch_idx in enumerate(channel_indices):
        # Offset each channel for better visualization
        offset = i * 200  # microvolts
        plt.plot(timestamps_segment, lfp_segment[:, ch_idx] + offset, linewidth=1, label=f"Channel {ch_idx}")
    
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV) + offset')
    plt.title(f'LFP Traces from Probe 0 ({channels_to_plot} channels)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lfp_traces.png')
    print("LFP traces plot saved to 'lfp_traces.png'")
    
    # Now let's do a time-frequency analysis for one channel
    from scipy import signal
    
    # Select a central channel for analysis
    central_channel = lfp_data.data.shape[1] // 2
    print(f"\nPerforming time-frequency analysis for channel {central_channel}...")
    
    # Get a longer segment for better frequency resolution (30 seconds)
    long_duration_sec = 30
    long_end_idx = start_idx + int(long_duration_sec * sampling_rate)
    lfp_long_segment = lfp_data.data[start_idx:long_end_idx, central_channel]
    timestamps_long_segment = lfp_data.timestamps[start_idx:long_end_idx]
    
    # Calculate the spectrogram
    nperseg = int(sampling_rate)  # 1-second window
    noverlap = nperseg // 2        # 50% overlap
    
    f, t, Sxx = signal.spectrogram(lfp_long_segment, fs=sampling_rate, 
                                  nperseg=nperseg, noverlap=noverlap,
                                  scaling='density')
    
    # Plot the spectrogram (log scale for better visualization)
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(t + start_time_sec, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(f'LFP Spectrogram (Channel {central_channel})')
    plt.ylim(0, 100)  # Limit to 0-100 Hz
    plt.colorbar(label='Power/Frequency [dB/Hz]')
    plt.tight_layout()
    plt.savefig('lfp_spectrogram.png')
    print("Spectrogram saved to 'lfp_spectrogram.png'")
    
    # Let's also calculate and plot average power in different frequency bands
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 80)
    }
    
    plt.figure(figsize=(12, 6))
    for band_name, (fmin, fmax) in bands.items():
        # Find frequency indices within the band
        idx_band = np.logical_and(f >= fmin, f <= fmax)
        # Calculate mean power within the band
        band_power = np.mean(Sxx[idx_band, :], axis=0)
        # Plot the band power as a function of time
        plt.plot(t + start_time_sec, 10 * np.log10(band_power), linewidth=2, label=f"{band_name} ({fmin}-{fmax} Hz)")
    
    plt.xlabel('Time [s]')
    plt.ylabel('Power [dB]')
    plt.title('LFP Power in Different Frequency Bands')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('lfp_frequency_bands.png')
    print("Frequency band power plot saved to 'lfp_frequency_bands.png'")
    
except Exception as e:
    print(f"Error processing LFP data: {e}")

# Try to correlate with stimulus presentations
try:
    # URL of the main file that contains stimulus information
    main_url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
    
    print("\nLoading main NWB file to get stimulus timing information...")
    main_remote_file = remfile.File(main_url)
    main_h5_file = h5py.File(main_remote_file)
    main_io = pynwb.NWBHDF5IO(file=main_h5_file)
    main_nwb = main_io.read()
    
    # Get stimulus presentations for a specific type
    stimulus_type = "SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations"
    stim_presentations = main_nwb.intervals[stimulus_type].to_dataframe()
    
    # Select a 30-second window containing stimulus presentations
    # Look for presentations within our analysis window
    window_start = start_time_sec
    window_end = start_time_sec + long_duration_sec
    
    presentations_in_window = stim_presentations[
        (stim_presentations['start_time'] >= window_start) & 
        (stim_presentations['start_time'] <= window_end)
    ]
    
    print(f"\nFound {len(presentations_in_window)} {stimulus_type} presentations in the analysis window")
    
    if len(presentations_in_window) > 0:
        # Plot the stimulus timing along with the band powers
        plt.figure(figsize=(12, 8))
        
        # Plot the band powers
        for band_name, (fmin, fmax) in bands.items():
            idx_band = np.logical_and(f >= fmin, f <= fmax)
            band_power = np.mean(Sxx[idx_band, :], axis=0)
            plt.plot(t + start_time_sec, 10 * np.log10(band_power), linewidth=2, label=f"{band_name} ({fmin}-{fmax} Hz)")
        
        # Add markers for stimulus presentations
        for _, pres in presentations_in_window.iloc[:100].iterrows():  # Limit to first 100 for clarity
            plt.axvline(x=pres['start_time'], color='r', alpha=0.3, linewidth=0.5)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Power [dB]')
        plt.title(f'LFP Power with {stimulus_type} Presentations')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('lfp_power_with_stimuli.png')
        print("LFP power with stimuli plot saved to 'lfp_power_with_stimuli.png'")
    
        # Create an event-related spectral analysis (average response to stimulus)
        # Take a window of +/- 0.5 seconds around each stimulus
        pre_stim = 0.5  # seconds before stimulus
        post_stim = 1.5  # seconds after stimulus
        
        # Convert to sampling rate
        pre_samples = int(pre_stim * sampling_rate)
        post_samples = int(post_stim * sampling_rate)
        window_samples = pre_samples + post_samples
        
        # Initialize matrix to hold spectrograms for each stimulus
        n_stimuli = min(50, len(presentations_in_window))  # Limit to 50 stimuli
        
        # Prepare time and frequency arrays for plotting
        window_time = np.linspace(-pre_stim, post_stim, window_samples)
        
        # Extract timestamps of first few stimulus presentations
        stim_times = presentations_in_window['start_time'].iloc[:n_stimuli].values
        
        # Plot the average LFP trace around stimulus onset
        plt.figure(figsize=(10, 6))
        
        # Calculate average LFP response to stimulus
        avg_response = np.zeros(window_samples)
        n_valid = 0
        
        for stim_time in stim_times:
            # Find the closest timestamp index
            closest_idx = np.argmin(np.abs(lfp_data.timestamps[:] - stim_time))
            
            # Extract LFP segment around the stimulus
            if closest_idx >= pre_samples and closest_idx + post_samples < len(lfp_data.timestamps):
                segment = lfp_data.data[closest_idx - pre_samples:closest_idx + post_samples, central_channel]
                avg_response += segment
                n_valid += 1
        
        if n_valid > 0:
            avg_response /= n_valid
            plt.plot(window_time, avg_response)
            plt.axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')
            plt.xlabel('Time relative to stimulus onset [s]')
            plt.ylabel('Amplitude [µV]')
            plt.title(f'Average LFP Response to {stimulus_type} (n={n_valid})')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig('average_lfp_response.png')
            print(f"Average LFP response plot saved (from {n_valid} presentations)")
        else:
            print("No valid stimulus presentations found for averaging")
    
except Exception as e:
    print(f"Error correlating with stimulus presentations: {e}")

print(f"\nScript completed in {time.time() - start_time:.2f} seconds")