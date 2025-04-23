"""
This script explores neural responses to natural movie stimuli in Dandiset 000690.
It loads the NWB files, examines the movie stimulus information, and analyzes neural
responses to natural movie presentations.
"""

import numpy as np
import h5py
import remfile
import pynwb
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy import signal

# Set the start time to measure execution
start_time = time.time()

# URL of the main file that contains stimulus information
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"

print("Loading NWB file from URL...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"File loaded in {time.time() - start_time:.2f} seconds")
print(f"Dandiset identifier: {nwb.identifier}")
print(f"Session date: {nwb.session_start_time}")
print(f"Session description: {nwb.session_description}")

# Check for natural movie stimulus presentations
print("\nLooking for natural movie stimulus presentations...")
natural_movie_intervals = []
for interval_name in nwb.intervals:
    if 'natmovie' in interval_name:
        natural_movie_intervals.append(interval_name)
        try:
            n_presentations = len(nwb.intervals[interval_name])
            print(f"- {interval_name}: {n_presentations} presentations")
        except Exception as e:
            print(f"- {interval_name}: Error accessing presentation count")

if not natural_movie_intervals:
    print("No natural movie stimulus presentations found.")
    exit()

# Choose a natural movie stimulus for analysis
selected_movie = natural_movie_intervals[0]
print(f"\nAnalyzing movie stimulus: {selected_movie}")

# Get the stimulus presentation times
movie_presentations = nwb.intervals[selected_movie].to_dataframe()
print(f"Number of presentations: {len(movie_presentations)}")
print("Presentation duration statistics:")
durations = movie_presentations['stop_time'] - movie_presentations['start_time']
print(f"  Mean duration: {durations.mean():.6f}s")
print(f"  Min duration: {durations.min():.6f}s")
print(f"  Max duration: {durations.max():.6f}s")

# Plot the first 20 movie presentation durations
plt.figure(figsize=(12, 6))
plt.bar(range(min(20, len(durations))), durations.iloc[:20])
plt.xlabel('Presentation Index')
plt.ylabel('Duration (seconds)')
plt.title(f'First 20 {selected_movie} Presentation Durations')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('movie_presentation_durations.png')
print("Movie presentation durations plot saved to 'movie_presentation_durations.png'")

# Now load the image file to see what's in the actual movie
print("\nAttempting to get information about the movie content...")
# Get URL of image file
image_url = "https://api.dandiarchive.org/api/assets/cbc64387-19b9-494a-a8fa-04d3207f7ffb/download/"

try:
    print("Loading image NWB file...")
    image_remote_file = remfile.File(image_url)
    image_h5_file = h5py.File(image_remote_file)
    image_io = pynwb.NWBHDF5IO(file=image_h5_file)
    image_nwb = image_io.read()
    
    print("Checking for movie templates...")
    # See if the selected movie exists in the stimulus_template
    if selected_movie in image_nwb.stimulus_template:
        movie_template = image_nwb.stimulus_template[selected_movie]
        print(f"Movie template found: {selected_movie}")
        print(f"Shape: {movie_template.data.shape}")
        print(f"Data type: {movie_template.data.dtype}")
        
        # Try to get one frame of the movie to visualize
        try:
            # Grab the middle frame to avoid potential issues with first/last frames
            try:
                middle_frame_idx = movie_template.data.shape[2] // 2
                frame = np.array(movie_template.data[:, :, middle_frame_idx, :]).copy() if len(movie_template.data.shape) == 4 else np.array(movie_template.data[:, :, middle_frame_idx]).copy()
                
                plt.figure(figsize=(10, 8))
                if len(movie_template.data.shape) == 4:  # Color movie (HWFC)
                    plt.imshow(frame)
                else:  # Grayscale movie (HWF)
                    plt.imshow(frame, cmap='gray')
                plt.title(f'Sample Frame from {selected_movie}')
                plt.axis('off')
                plt.savefig('movie_sample_frame.png')
                print(f"Sample frame from movie saved to 'movie_sample_frame.png'")
            except Exception as e:
                print(f"Error extracting middle frame: {e}")
                
                # Try the first frame instead
                try:
                    print("Attempting to extract first frame...")
                    first_frame = np.array(movie_template.data[:, :, 0, :]).copy() if len(movie_template.data.shape) == 4 else np.array(movie_template.data[:, :, 0]).copy()
                    
                    plt.figure(figsize=(10, 8))
                    if len(movie_template.data.shape) == 4:  # Color movie (HWFC)
                        plt.imshow(first_frame)
                    else:  # Grayscale movie (HWF)
                        plt.imshow(first_frame, cmap='gray')
                    plt.title(f'First Frame from {selected_movie}')
                    plt.axis('off')
                    plt.savefig('movie_first_frame.png')
                    print(f"First frame from movie saved to 'movie_first_frame.png'")
                except Exception as e2:
                    print(f"Error extracting first frame: {e2}")
        except Exception as e:
            print(f"Error visualizing movie frames: {e}")
    else:
        print(f"Movie template not found for {selected_movie}")
except Exception as e:
    print(f"Error accessing movie content: {e}")

# Now analyze neural responses to the movie presentations
# First, get information about the units
print("\nExtracting unit information...")
units_df = nwb.units.to_dataframe()
print(f"Total number of units: {len(units_df)}")

# Print quality distribution
print("\nUnit quality distribution:")
quality_counts = units_df['quality'].value_counts()
print(quality_counts)

# Get good quality units only
good_units = units_df[units_df['quality'] == 'good'].copy()
print(f"Number of good quality units: {len(good_units)}")

# Get the responses of a few high-firing units to movie presentations
sorted_units = good_units.sort_values('firing_rate', ascending=False)
top_n_units = 3  # Number of top firing units to analyze
units_to_analyze = sorted_units.head(top_n_units).index

print(f"\nAnalyzing movie responses for top {top_n_units} units...")

# Get a subset of movie presentations to analyze (first 10)
presentations_to_analyze = min(10, len(movie_presentations))
movie_subset = movie_presentations.head(presentations_to_analyze)

# Define time window for analysis
pre_stim = 0.5  # seconds before stimulus
post_stim = 2.0  # seconds after stimulus

# Create plots for each unit
for unit_idx, unit_id in enumerate(units_to_analyze):
    unit_info = sorted_units.loc[unit_id]
    print(f"\nAnalyzing unit {unit_id}, firing rate: {unit_info['firing_rate']:.2f} Hz")
    
    # Get spike times for this unit
    spike_times = nwb.units[unit_id]['spike_times']
    print(f"Number of spikes: {len(spike_times)}")
    
    # Create figure for raster and PSTH for this unit
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Collect all spikes for PSTH
    all_aligned_spikes = []
    
    # Plot raster for each presentation
    for pres_idx, (idx, pres) in enumerate(movie_subset.iterrows()):
        stim_start = pres['start_time']
        stim_end = pres['stop_time']
        
        # Find spikes within the window
        window_start = stim_start - pre_stim
        window_end = stim_start + post_stim
        mask = (spike_times >= window_start) & (spike_times <= window_end)
        spikes_in_window = spike_times[mask]
        
        # Align spikes to stimulus onset
        aligned_spikes = spikes_in_window - stim_start
        all_aligned_spikes.extend(aligned_spikes)
        
        # Plot raster
        axes[0].eventplot(aligned_spikes, colors='k', linelengths=0.5, 
                         lineoffsets=pres_idx+1)
    
    # Add stimulus duration line
    axes[0].axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')
    avg_duration = durations.mean()
    axes[0].axvline(x=avg_duration, color='g', linestyle='--', label='Avg. Stimulus Offset')
    axes[0].set_title(f'Unit {unit_id} Response to {selected_movie}')
    axes[0].set_ylabel('Presentation')
    axes[0].legend()
    
    # Plot PSTH
    bin_width = 0.05  # 50 ms bins
    bins = np.arange(-pre_stim, post_stim + bin_width, bin_width)
    psth, edges = np.histogram(all_aligned_spikes, bins=bins)
    psth = psth / presentations_to_analyze / bin_width  # Convert to Hz
    
    axes[1].bar(edges[:-1], psth, width=bin_width, alpha=0.7)
    axes[1].axvline(x=0, color='r', linestyle='--')
    axes[1].axvline(x=avg_duration, color='g', linestyle='--')
    axes[1].set_xlabel('Time from stimulus onset (s)')
    axes[1].set_ylabel('Firing rate (Hz)')
    
    plt.tight_layout()
    plt.savefig(f'unit_{unit_id}_movie_response.png')
    print(f"Response plot saved to 'unit_{unit_id}_movie_response.png'")

print(f"\nScript completed in {time.time() - start_time:.2f} seconds")