# This script explores the eye tracking data from the Dandiset 000690
# We want to understand the eye tracking data structure and visualize some example data

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create the output directory for plots
os.makedirs('explore', exist_ok=True)

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the file
print(f"Session ID: {nwb.session_id}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject info: {nwb.subject.subject_id}, {nwb.subject.species}, Age: {nwb.subject.age}")

# Access eye tracking data
print("\nExploring eye tracking data:")
eye_tracking = nwb.acquisition['EyeTracking']
print(f"Eye tracking data type: {type(eye_tracking)}")
print(f"Available spatial series: {list(eye_tracking.spatial_series.keys())}")

# Get pupil data
pupil_tracking = eye_tracking.pupil_tracking
print(f"\nPupil tracking data shape: {pupil_tracking.data.shape}")
print(f"Timestamps shape: {pupil_tracking.timestamps.shape if hasattr(pupil_tracking.timestamps, 'shape') else 'N/A'}")

# Plot a sample of pupil position data
sample_size = 1000  # Number of data points to plot
plt.figure(figsize=(10, 6))
plt.plot(pupil_tracking.data[:sample_size, 0], pupil_tracking.data[:sample_size, 1], 'b-', alpha=0.7)
plt.scatter(pupil_tracking.data[:sample_size, 0], pupil_tracking.data[:sample_size, 1], c='r', s=5)
plt.title('Pupil Position (first 1000 samples)')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.grid(True)
plt.savefig('explore/pupil_position.png')
plt.close()

# Plot pupil area over time
plt.figure(figsize=(12, 6))
plt.plot(pupil_tracking.area[:sample_size])
plt.title('Pupil Area (first 1000 samples)')
plt.xlabel('Sample index')
plt.ylabel('Area')
plt.grid(True)
plt.savefig('explore/pupil_area.png')
plt.close()

# Check for blink data
if hasattr(eye_tracking, 'likely_blink'):
    blinks = eye_tracking.likely_blink.data[:sample_size]
    plt.figure(figsize=(12, 6))
    plt.plot(blinks)
    plt.title('Blink Detection (first 1000 samples, 1 = blink)')
    plt.xlabel('Sample index')
    plt.ylabel('Blink detected')
    plt.grid(True)
    plt.savefig('explore/blink_detection.png')
    plt.close()
    print(f"Blink data shape: {eye_tracking.likely_blink.data.shape}")
    print(f"Number of blinks in sample: {np.sum(blinks)}")
else:
    print("No blink detection data found")

# Get information about available stimuli
print("\nExploring stimuli information:")
if hasattr(nwb, 'intervals'):
    stim_intervals = [k for k in nwb.intervals.keys() if 'presentations' in k]
    print(f"Found {len(stim_intervals)} stimulus presentation intervals")
    for i, stim in enumerate(stim_intervals[:5]):  # Just show first 5
        print(f"{i+1}. {stim}")
else:
    print("No intervals found in the dataset")

# Look at one stimulus presentation interval in detail
if len(stim_intervals) > 0:
    example_stim = nwb.intervals[stim_intervals[0]]
    print(f"\nDetailed view of {stim_intervals[0]}:")
    print(f"Description: {example_stim.description}")
    print(f"Column names: {example_stim.colnames}")
    
    # Get some statistics about when stimuli were presented
    start_times = example_stim.start_time[:]
    stop_times = example_stim.stop_time[:]
    durations = stop_times - start_times
    
    print(f"Number of presentations: {len(start_times)}")
    print(f"Average presentation duration: {np.mean(durations):.6f} seconds")
    print(f"Total presentation time: {np.sum(durations):.2f} seconds")
    
    # Plot stimulus presentation start times
    plt.figure(figsize=(12, 6))
    plt.plot(start_times[:1000], np.ones(min(1000, len(start_times))), '|', markersize=10)
    plt.title(f'First 1000 Stimulus Presentation Start Times: {stim_intervals[0]}')
    plt.xlabel('Time (seconds)')
    plt.yticks([])
    plt.grid(True, axis='x')
    plt.savefig('explore/stimulus_times.png')
    plt.close()
    
    # Plot distribution of stimulus durations
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50)
    plt.title('Distribution of Stimulus Presentation Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('explore/stimulus_durations.png')
    plt.close()

print("Exploration completed. Check the explore directory for plots.")