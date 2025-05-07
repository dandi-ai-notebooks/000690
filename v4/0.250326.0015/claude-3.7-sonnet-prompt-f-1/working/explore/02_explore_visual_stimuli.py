"""
This script explores the visual stimuli used in the experiment,
focusing on the stimulus presentations and their properties.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
from collections import defaultdict

# Load the image NWB file which contains the actual stimulus images
url = "https://api.dandiarchive.org/api/assets/cbc64387-19b9-494a-a8fa-04d3207f7ffb/download/"
print(f"Loading NWB file from URL: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print stimulus information
print("Available stimuli in stimulus_template:")
for stim_name, stim in nwb.stimulus_template.items():
    if hasattr(stim, 'data'):
        shape = stim.data.shape
        print(f"  {stim_name}: Shape {shape}, Rate {stim.rate} Hz")

# Get a list of stimulus types from the intervals
print("\nStimulus presentation intervals:")
for interval_name, interval in nwb.intervals.items():
    if 'presentations' in interval_name:
        df = interval.to_dataframe()
        print(f"  {interval_name}: {len(df)} presentations")
        if 'stimulus_name' in df.columns:
            unique_stim_names = df['stimulus_name'].unique()
            print(f"    Unique stimulus names: {unique_stim_names}")

# Function to save a single frame from a stimulus
def save_stimulus_frame(stim_name, frame_idx=0, output_path=None):
    stim = nwb.stimulus_template[stim_name]
    
    # Handle different dimensional structures
    if len(stim.data.shape) == 3:  # height, width, frames
        frame = stim.data[:, :, frame_idx]
        is_color = False
    elif len(stim.data.shape) == 4:  # height, width, frames, color
        frame = stim.data[:, :, frame_idx, :]
        is_color = True
    else:
        print(f"Unexpected stimulus shape: {stim.data.shape}")
        return
    
    plt.figure(figsize=(10, 6))
    if is_color:
        plt.imshow(frame)
    else:
        plt.imshow(frame, cmap='gray')
    plt.title(f"{stim_name} - Frame {frame_idx}")
    plt.axis('off')
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved frame to {output_path}")
    plt.close()

# Function to create and save a small GIF animation
def create_stimulus_animation(stim_name, num_frames=10, start_frame=0, output_path=None):
    stim = nwb.stimulus_template[stim_name]
    
    # Determine dimensions
    if len(stim.data.shape) == 3:  # height, width, frames
        is_color = False
        total_frames = stim.data.shape[2]
    elif len(stim.data.shape) == 4:  # height, width, frames, color
        is_color = True
        total_frames = stim.data.shape[2]
    else:
        print(f"Unexpected stimulus shape: {stim.data.shape}")
        return
    
    # Limit number of frames to what's available
    if start_frame + num_frames > total_frames:
        num_frames = total_frames - start_frame
    
    # Create animation in memory
    frames = []
    for i in range(start_frame, start_frame + num_frames):
        if is_color:
            frames.append(stim.data[:, :, i, :])
        else:
            frames.append(stim.data[:, :, i])
    
    # Return first frame to create a still image
    return frames[0]

# Save an example frame from different stimulus types
# Simple bar stimulus
print("\nSaving example frames...")
save_stimulus_frame('SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations', frame_idx=0, 
                    output_path='explore/simple_bar_stimulus_frame.png')

# "Disco" bar stimulus
save_stimulus_frame('Disco2SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations', frame_idx=0,
                    output_path='explore/disco_bar_stimulus_frame.png')

# Natural movie stimulus
save_stimulus_frame('natmovie_EagleSwooping1_540x960Full_584x460Active_presentations', frame_idx=0,
                     output_path='explore/natural_movie_frame.png')

# Get information about stimulus parameters
print("\nExtracting stimulus parameter information...")
stimulus_info = defaultdict(list)

# Sample a few stimulus presentation intervals to extract parameter info
for interval_name in ['SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations', 
                      'SAC_Wd45_Vel2_Bndry1_Cntst0_loop_presentations',
                      'SAC_Wd15_Vel8_Bndry1_Cntst0_loop_presentations']:
    if interval_name in nwb.intervals:
        interval = nwb.intervals[interval_name]
        df = interval.to_dataframe()
        
        # Get the first row to extract parameters
        if len(df) > 0:
            first_row = df.iloc[0]
            for param in ['size', 'contrast', 'orientation', 'color', 'opacity']:
                if param in first_row:
                    stimulus_info[param].append(f"{interval_name}: {first_row[param]}")

# Print parameter information
for param, values in stimulus_info.items():
    print(f"\n{param.capitalize()} values:")
    for value in values:
        print(f"  {value}")

print("\nExploration of visual stimuli complete!")