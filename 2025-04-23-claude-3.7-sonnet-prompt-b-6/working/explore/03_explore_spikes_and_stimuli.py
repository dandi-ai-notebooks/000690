# This script explores spike data and stimulus information from the Dandiset
# to understand neural activity in relation to presented visual stimuli

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb
import pandas as pd
from matplotlib.cm import get_cmap

# Save plots to file instead of displaying
plt.ioff()

# Load a main NWB file (containing spikes and stimulus info)
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading main NWB file from {url}")

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

# Check for units (spike) data
print("\nUnits (spike) information:")
if hasattr(nwb, 'units') and nwb.units is not None:
    units_df = nwb.units.to_dataframe()
    print(f"Total number of units: {len(units_df)}")
    if 'location' in units_df.columns:
        print("Units per brain region:")
        print(units_df['location'].value_counts())
    elif 'peak_channel_id' in units_df.columns:
        print(f"Units listed by peak_channel_id")
    
    # Print some unit statistics
    if 'firing_rate' in units_df.columns:
        print(f"\nFiring rate stats:")
        print(f"Mean: {units_df['firing_rate'].mean():.2f} Hz")
        print(f"Median: {units_df['firing_rate'].median():.2f} Hz")
        print(f"Min: {units_df['firing_rate'].min():.2f} Hz")
        print(f"Max: {units_df['firing_rate'].max():.2f} Hz")
    
    # Plot histogram of firing rates
    if 'firing_rate' in units_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(units_df['firing_rate'], bins=30)
        plt.xlabel('Firing Rate (Hz)')
        plt.ylabel('Number of Units')
        plt.title('Distribution of Unit Firing Rates')
        plt.savefig('explore/firing_rate_distribution.png')
        plt.close()
    
    # Get a list of some spike times 
    print("\nExtracting some example spike times...")
    # Sample a few random units to examine their spike timing
    n_units_to_sample = min(5, len(units_df))
    sampled_units = np.random.choice(units_df.index.values, size=n_units_to_sample, replace=False)
    
    for unit_id in sampled_units:
        spike_times = nwb.units['spike_times'][unit_id]
        if len(spike_times) > 0:
            print(f"Unit {unit_id}: {len(spike_times)} spikes, first few times: {spike_times[:5]}")
        else:
            print(f"Unit {unit_id}: No spikes recorded")
else:
    print("No units data found in this file")

# Check for stimulus information
print("\nStimulus information:")
stimulus_intervals = []
stimulus_names = []

# Look for intervals data (stimulus presentations)
if hasattr(nwb, 'intervals') and len(nwb.intervals) > 0:
    print("\nAvailable intervals (stimulus presentations):")
    for interval_name in nwb.intervals:
        print(f"- {interval_name}")
        
        # Get interval data as DataFrame
        try:
            interval_df = nwb.intervals[interval_name].to_dataframe()
            n_intervals = len(interval_df)
            stimulus_intervals.append(interval_df)
            stimulus_names.append(interval_name)
            
            if n_intervals > 0:
                print(f"  - Number of intervals: {n_intervals}")
                print(f"  - Duration range: {interval_df['stop_time'].min() - interval_df['start_time'].min():.2f} - {interval_df['stop_time'].max() - interval_df['start_time'].max():.2f} seconds")
                
                # Calculate the distribution of stimulus durations
                durations = interval_df['stop_time'] - interval_df['start_time']
                print(f"  - Mean duration: {durations.mean():.4f} seconds")
                print(f"  - Median duration: {durations.median():.4f} seconds")
                
                # Plot duration histogram
                plt.figure(figsize=(10, 6))
                plt.hist(durations, bins=30)
                plt.xlabel('Duration (seconds)')
                plt.ylabel('Count')
                plt.title(f'Distribution of {interval_name} Durations')
                plt.savefig(f'explore/stimulus_durations_{interval_name[:20]}.png')
                plt.close()
                
        except Exception as e:
            print(f"  - Error reading interval data: {e}")
else:
    print("No stimulus intervals found in this file")

# Check if there's stimulus template data (actual visual stimuli)
if hasattr(nwb, 'stimulus_template') and len(nwb.stimulus_template) > 0:
    print("\nAvailable stimulus templates:")
    for template_name in nwb.stimulus_template:
        template = nwb.stimulus_template[template_name]
        print(f"- {template_name}: shape {template.data.shape}, dtype {template.data.dtype}")
        
        # Try to visualize a frame from one of the stimuli
        try:
            # Take a middle frame to visualize
            if len(template.data.shape) == 3:  # [height, width, frames]
                middle_frame_idx = template.data.shape[2] // 2
                frame = template.data[:, :, middle_frame_idx]
                plt.figure(figsize=(10, 6))
                plt.imshow(frame, cmap='gray')
                plt.colorbar()
                plt.title(f'Middle Frame of {template_name}')
                plt.savefig(f'explore/stimulus_frame_{template_name[:20]}.png')
                plt.close()
            elif len(template.data.shape) == 4:  # [height, width, frames, channels]
                middle_frame_idx = template.data.shape[2] // 2
                frame = template.data[:, :, middle_frame_idx, :]
                plt.figure(figsize=(10, 6))
                plt.imshow(frame)
                plt.title(f'Middle Frame of {template_name}')
                plt.savefig(f'explore/stimulus_frame_{template_name[:20]}.png')
                plt.close()
                
            print(f"  - Saved middle frame visualization")
        except Exception as e:
            print(f"  - Could not visualize template: {e}")
else:
    print("No stimulus templates found in this file")

# If we have both unit data and stimulus information, try to plot a simple PSTH (Peri-Stimulus Time Histogram)
if hasattr(nwb, 'units') and nwb.units is not None and len(stimulus_intervals) > 0:
    print("\nCreating simple PSTH for a sample unit...")
    
    # Choose a random unit with sufficient spikes
    unit_spike_counts = {i: len(nwb.units['spike_times'][i]) for i in range(len(nwb.units['spike_times']))}
    sorted_units = sorted(unit_spike_counts.items(), key=lambda x: x[1], reverse=True)
    
    if sorted_units and sorted_units[0][1] > 100:  # Check for at least 100 spikes
        unit_id = sorted_units[0][0]
        print(f"Selected unit {unit_id} with {sorted_units[0][1]} spikes for PSTH analysis")
        
        # Get spike times for this unit
        spike_times = nwb.units['spike_times'][unit_id]
        
        # Use the most common stimulus type (most intervals)
        stim_idx = np.argmax([len(df) for df in stimulus_intervals])
        stim_name = stimulus_names[stim_idx]
        stim_df = stimulus_intervals[stim_idx]
        print(f"Using stimulus: {stim_name} with {len(stim_df)} presentations")
        
        # Limit to first 50 stimulus presentations for speed
        max_stim = min(50, len(stim_df))
        stim_df = stim_df.iloc[:max_stim]
        
        # Parameters for PSTH
        window_size = 1.0  # seconds
        bin_size = 0.05    # seconds
        bins = np.arange(-window_size, window_size + bin_size, bin_size)
        bin_centers = bins[:-1] + bin_size/2
        all_counts = np.zeros((len(stim_df), len(bins)-1))
        
        # Compute PSTH
        for i, (_, stim) in enumerate(stim_df.iterrows()):
            stim_start = stim['start_time']
            
            # Find spikes relative to stimulus onset
            relative_spike_times = spike_times - stim_start
            
            # Filter spikes within our window
            mask = (relative_spike_times >= -window_size) & (relative_spike_times <= window_size)
            windowed_spikes = relative_spike_times[mask]
            
            # Count spikes in bins
            counts, _ = np.histogram(windowed_spikes, bins)
            all_counts[i, :] = counts
        
        # Average across all stimulus presentations
        mean_counts = all_counts.mean(axis=0) / bin_size  # Convert to Hz
        
        # Plot PSTH
        plt.figure(figsize=(10, 6))
        plt.bar(bin_centers, mean_counts, width=bin_size, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')
        plt.xlabel('Time relative to stimulus onset (seconds)')
        plt.ylabel('Firing Rate (Hz)')
        plt.title(f'PSTH for Unit {unit_id} - Stimulus: {stim_name}')
        plt.legend()
        plt.savefig('explore/psth_example.png')
        plt.close()
        
        print("PSTH analysis complete")
        
        # Create a plot showing spike times for the unit aligned to multiple stimulus onsets
        plt.figure(figsize=(12, 8))
        
        # Show first 20 stimulus presentations
        n_to_show = min(20, len(stim_df))
        
        for i in range(n_to_show):
            stim_start = stim_df.iloc[i]['start_time']
            relative_spike_times = spike_times - stim_start
            
            # Get spikes within window
            mask = (relative_spike_times >= -window_size) & (relative_spike_times <= window_size)
            windowed_spikes = relative_spike_times[mask]
            
            # Plot raster
            plt.fill_between([0, stim_df.iloc[i]['stop_time'] - stim_start], 
                            i - 0.4, i + 0.4, color='lightgray', alpha=0.3)
            plt.scatter(windowed_spikes, np.ones_like(windowed_spikes) * i, 
                      marker='|', color='black', s=20)
        
        plt.axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')
        plt.xlabel('Time relative to stimulus onset (seconds)')
        plt.ylabel('Stimulus Presentation #')
        plt.xlim([-window_size, window_size])
        plt.title(f'Spike Raster for Unit {unit_id} - Stimulus: {stim_name}')
        plt.legend()
        plt.savefig('explore/spike_raster.png')
        plt.close()
        
        print("Spike raster plot created")
    else:
        print("No units with sufficient spikes found for PSTH analysis")
        
print("\nCompleted spike and stimulus exploration")