# This script explores the visual stimuli used in the experiments
# We want to understand the types of stimuli, their timing, and visualize examples

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory for plots
os.makedirs('explore', exist_ok=True)

# Load the image NWB file
url = "https://api.dandiarchive.org/api/assets/cbc64387-19b9-494a-a8fa-04d3207f7ffb/download/"
print(f"Loading image NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the file
print(f"Session ID: {nwb.session_id}")
print(f"Session description: {nwb.session_description}")

# Get information about the stimulus templates
stim_templates = nwb.stimulus_template
print(f"\nStimulus templates found: {len(stim_templates.keys())}")

# List all stimulus types
stim_types = list(stim_templates.keys())
print("All stimulus types:")
for i, stim in enumerate(stim_types):
    print(f"{i+1}. {stim}")

# Get stimulus presentation intervals
stim_intervals = [k for k in nwb.intervals.keys() if 'presentations' in k]
print(f"\nFound {len(stim_intervals)} stimulus presentation intervals")

# Create a summary of the stimulus presentations
total_presentations = 0
presentation_counts = {}

for stim_name in stim_intervals:
    stim_data = nwb.intervals[stim_name]
    count = len(stim_data.start_time[:])
    total_presentations += count
    presentation_counts[stim_name] = count

# Plot the distribution of stimulus presentations
plt.figure(figsize=(15, 8))
names = list(presentation_counts.keys())
values = list(presentation_counts.values())
# Sort by count
sorted_indices = np.argsort(values)[::-1]  # Descending order
names = [names[i] for i in sorted_indices]
values = [values[i] for i in sorted_indices]

# Use abbreviated names for better readability
abbrev_names = [name.split('_')[0] for name in names]

plt.bar(abbrev_names, values)
plt.title('Number of Presentations by Stimulus Type')
plt.xlabel('Stimulus Type')
plt.ylabel('Number of Presentations')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('explore/stimulus_presentation_counts.png')
plt.close()

print(f"\nTotal stimulus presentations: {total_presentations}")

# Function to visualize a stimulus
def visualize_stimulus(stim_name, frame_indices=None):
    if stim_name not in stim_templates:
        print(f"Stimulus {stim_name} not found in templates")
        return
    
    stim_template = stim_templates[stim_name]
    
    # Get information about the stimulus
    print(f"\nStimulus: {stim_name}")
    print(f"Data shape: {stim_template.data.shape}")
    print(f"Frame rate: {stim_template.rate} Hz")
    
    # If frame indices not provided, use evenly spaced frames
    if frame_indices is None:
        num_frames = min(4, stim_template.data.shape[2])  # Max 4 frames
        frame_indices = np.linspace(0, stim_template.data.shape[2]-1, num_frames, dtype=int)
    
    # Plot the frames
    fig, axes = plt.subplots(1, len(frame_indices), figsize=(15, 5))
    if len(frame_indices) == 1:
        axes = [axes]  # Make iterable for single frame
        
    for i, frame_idx in enumerate(frame_indices):
        if len(stim_template.data.shape) == 4:  # Color images (H, W, T, C)
            frame = stim_template.data[:, :, frame_idx, :]
            axes[i].imshow(frame)
        else:  # Grayscale images (H, W, T)
            frame = stim_template.data[:, :, frame_idx]
            axes[i].imshow(frame, cmap='gray')
        axes[i].set_title(f"Frame {frame_idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'explore/stimulus_{stim_name.split("_")[0]}_frames.png')
    plt.close()
    
    return frame_indices

# Visualize a few different types of stimuli
# Natural movie stimulus
nat_movie_stim = [s for s in stim_types if 'natmovie' in s]
if nat_movie_stim:
    frames = visualize_stimulus(nat_movie_stim[0])  # First natural movie

# Simple bar stimulus (SAC = standard bar)
sac_stim = [s for s in stim_types if 'SAC_Wd15_Vel2' in s]
if sac_stim:
    frames = visualize_stimulus(sac_stim[0], [0, 60, 120, 180])  # Different frames

# Special stimulus (Disco or Green)
special_stim = [s for s in stim_types if 'Disco' in s or 'Green' in s]
if special_stim:
    frames = visualize_stimulus(special_stim[0])

# Analyze stimulus timing for one example
example_stim_name = sac_stim[0] if sac_stim else stim_intervals[0]
stim_data = nwb.intervals[example_stim_name]

# Calculate inter-stimulus intervals
start_times = stim_data.start_time[:]
stop_times = stim_data.stop_time[:]
durations = stop_times - start_times

if len(start_times) > 1:
    isis = np.diff(start_times)  # Inter-stimulus intervals
    
    plt.figure(figsize=(12, 6))
    plt.hist(isis, bins=50)
    plt.title(f'Inter-stimulus Intervals for {example_stim_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('explore/inter_stimulus_intervals.png')
    plt.close()
    
    print(f"\nStimulus timing for {example_stim_name}:")
    print(f"Average stimulus duration: {np.mean(durations):.6f} s")
    print(f"Average inter-stimulus interval: {np.mean(isis):.6f} s")
    print(f"Total experiment time spanned: {np.max(stop_times) - np.min(start_times):.2f} s")

# Try to get additional metadata about stimuli if available
example_stim_data = nwb.intervals[example_stim_name]
print("\nStimulus metadata columns:")
for column in example_stim_data.colnames:
    print(f"- {column}")

# Try to analyze specific parameter variations in the stimuli
try:
    if 'contrast' in example_stim_data.colnames:
        contrasts = example_stim_data.contrast[:]
        unique_contrasts = np.unique(contrasts)
        print(f"\nUnique contrast values in {example_stim_name}: {unique_contrasts}")
        
    if 'orientation' in example_stim_data.colnames:
        orientations = example_stim_data.orientation[:]
        unique_orientations = np.unique(orientations)
        print(f"Unique orientation values: {unique_orientations}")
        
    if 'size' in example_stim_data.colnames:
        sizes = example_stim_data.size[:]
        unique_sizes = np.unique(sizes)
        print(f"Unique size values: {unique_sizes}")
except Exception as e:
    print(f"Error accessing metadata: {e}")

print("Visual stimuli exploration completed. Check explore directory for plots.")