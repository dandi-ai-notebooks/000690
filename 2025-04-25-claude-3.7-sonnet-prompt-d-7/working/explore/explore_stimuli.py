"""
This script explores the visual stimuli information from the image NWB file to understand:
1. The structure of stimulus data
2. The different types of stimuli presented
3. Visual representation of sample stimuli
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Save figures with standard style (not seaborn, since we're plotting images)
plt.style.use('default')

# Set the URL for the image data
url = "https://api.dandiarchive.org/api/assets/cbc64387-19b9-494a-a8fa-04d3207f7ffb/download/"

# Load the data
print("Loading data from remote NWB file...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("NWB file loaded. Exploring stimulus data...")

# Look at the stimulus template section
stimulus_templates = nwb.stimulus_template
print(f"Available stimulus templates: {list(stimulus_templates.keys())}")

# Look at intervals for presentations
interval_names = list(nwb.intervals.keys())
print(f"\nNumber of stimulus presentation intervals: {len(interval_names)}")
print(f"Sample of interval names: {interval_names[:5]}")

# Group stimulus names by pattern to understand categories
stimulus_types = {}
for name in interval_names:
    if "_presentations" not in name:
        continue
    
    base_name = name.replace("_presentations", "")
    parts = base_name.split("_")
    
    # Group by first part of name (stimulus type)
    stim_type = parts[0]
    if stim_type not in stimulus_types:
        stimulus_types[stim_type] = []
    
    stimulus_types[stim_type].append(name)

# Print stimulus categories
print("\nStimulus categories:")
for stim_type, intervals in stimulus_types.items():
    print(f"- {stim_type}: {len(intervals)} variants")

# Plot number of variants per stimulus type
plt.figure(figsize=(12, 6))
counts = [len(intervals) for stim_type, intervals in stimulus_types.items()]
plt.bar(stimulus_types.keys(), counts)
plt.xlabel('Stimulus Type')
plt.ylabel('Number of Variants')
plt.title('Number of Stimulus Variants per Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('explore/stimulus_type_counts.png', dpi=300, bbox_inches='tight')
plt.close()

# Select a few example stimuli to visualize
example_stimuli = [
    "SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",  # Standard bar stimulus
    "Disco2SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",  # Disco bar stimulus
    "natmovie_EagleSwooping1_540x960Full_584x460Active_presentations"  # Natural movie
]

# Check if the natural movie key exists, if not select another one
if example_stimuli[2] not in nwb.intervals:
    # Find a natural movie that exists
    for key in nwb.intervals.keys():
        if key.startswith("natmovie_"):
            example_stimuli[2] = key
            break

# Plot sample frames from each stimulus
for stim_name in example_stimuli:
    # Get stimulus template
    if stim_name in stimulus_templates:
        stim_template = stimulus_templates[stim_name]
        print(f"\nStimulus {stim_name}:")
        print(f"Shape: {stim_template.data.shape}")
        print(f"Rate: {stim_template.rate} Hz")
        
        # Determine if it's a RGB stimulus or grayscale
        is_rgb = len(stim_template.data.shape) == 4

        # Get a few frames to display
        if is_rgb:
            # For RGB, get frames at intervals
            n_frames = stim_template.data.shape[2]
            frame_indices = np.linspace(0, n_frames-1, 4, dtype=int)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for i, frame_idx in enumerate(frame_indices):
                frame = stim_template.data[:, :, frame_idx, :]
                axes[i].imshow(frame)
                axes[i].set_title(f"Frame {frame_idx}")
                axes[i].axis('off')
        else:
            # For grayscale, get frames at intervals
            n_frames = stim_template.data.shape[2]  
            frame_indices = np.linspace(0, n_frames-1, 4, dtype=int)
            
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for i, frame_idx in enumerate(frame_indices):
                frame = stim_template.data[:, :, frame_idx]
                axes[i].imshow(frame, cmap='gray')
                axes[i].set_title(f"Frame {frame_idx}")
                axes[i].axis('off')
                
        plt.suptitle(f"{stim_name.replace('_presentations', '')}")
        plt.tight_layout()
        plt.savefig(f'explore/stimulus_{stim_name.replace("_presentations", "")}_frames.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print(f"Stimulus template not found for {stim_name}")

# Get presentation intervals for a sample stimulus
presentation_df = None
for stim_name in example_stimuli:
    if stim_name in nwb.intervals:
        presentation_df = nwb.intervals[stim_name].to_dataframe()
        break

if presentation_df is not None:
    print("\nSample of stimulus presentation intervals:")
    print(presentation_df.head())
    
    # Plot the presentation times
    plt.figure(figsize=(12, 6))
    plt.scatter(presentation_df['start_time'], np.zeros_like(presentation_df['start_time']), 
                alpha=0.5, marker='|', s=80)
    plt.xlabel('Time (s)')
    plt.yticks([])
    plt.title(f'Presentation Times for {stim_name.replace("_presentations", "")}')
    plt.xlim(presentation_df['start_time'].min(), 
             min(presentation_df['start_time'].min() + 300, presentation_df['start_time'].max()))
    plt.grid(True, axis='x')
    plt.savefig('explore/stimulus_presentation_times.png', dpi=300, bbox_inches='tight')
    plt.close()
        
    # If columns include stimulus parameters, show distributions
    param_columns = ['contrast', 'size', 'orientation', 'opacity']
    available_params = [col for col in param_columns if col in presentation_df.columns]
    
    if available_params:
        fig, axes = plt.subplots(len(available_params), 1, figsize=(10, 3*len(available_params)))
        if len(available_params) == 1:
            axes = [axes]  # Make it iterable for single parameter
            
        for i, param in enumerate(available_params):
            if presentation_df[param].nunique() < 10:  # Categorical
                presentation_df[param].value_counts().plot(kind='bar', ax=axes[i])
            else:  # Continuous
                axes[i].hist(presentation_df[param], bins=20)
            
            axes[i].set_title(f'Distribution of {param}')
            axes[i].set_xlabel(param)
            axes[i].set_ylabel('Count')
            
        plt.tight_layout()
        plt.savefig('explore/stimulus_parameters.png', dpi=300, bbox_inches='tight')
        plt.close()

print("Script completed successfully!")