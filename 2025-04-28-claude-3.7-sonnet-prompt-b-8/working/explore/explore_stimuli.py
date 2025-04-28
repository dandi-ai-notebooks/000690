# Script to explore the visual stimuli data in the Dandiset
# This examines the types of stimuli, their presentation timing, and sample images

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import time

# Configure matplotlib to save rather than display
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

# Function to load a session file and get stimulus information
def get_stimulus_info(session_asset_id):
    print(f"Loading session NWB file...")
    url = f"https://api.dandiarchive.org/api/assets/{session_asset_id}/download/"
    
    try:
        remote_file = remfile.File(url)
        h5_file = h5py.File(remote_file)
        io = pynwb.NWBHDF5IO(file=h5_file)
        nwb = io.read()
        
        print(f"Session ID: {nwb.session_id}")
        
        # Get stimulus presentation intervals
        if hasattr(nwb, 'intervals'):
            stim_intervals = [k for k in nwb.intervals.keys() if "presentations" in k]
            print(f"\nFound {len(stim_intervals)} stimulus presentation intervals:")
            for i, interval in enumerate(stim_intervals[:10]):  # Print first 10
                print(f"{i+1}. {interval}")
            if len(stim_intervals) > 10:
                print(f"... and {len(stim_intervals) - 10} more")
            
            # Get detailed info about one example stimulus presentation
            if len(stim_intervals) > 0:
                sample_interval = stim_intervals[0]
                interval_df = nwb.intervals[sample_interval].to_dataframe()
                print(f"\nExample stimulus '{sample_interval}' has {len(interval_df)} presentations")
                print(f"Start time range: {interval_df['start_time'].min():.2f} - {interval_df['start_time'].max():.2f} seconds")
                print(f"Duration range: {(interval_df['stop_time'] - interval_df['start_time']).mean():.4f} ± {(interval_df['stop_time'] - interval_df['start_time']).std():.4f} seconds")
                
                # Plot some basic presentation time information
                plt.figure(figsize=(15, 6))
                # Get first N presentations to avoid very large plots
                max_stim = min(100, len(interval_df))
                plt.eventplot(interval_df['start_time'].values[:max_stim], lineoffsets=0.5, linelengths=0.5, linewidths=2)
                plt.xlabel('Time (seconds)')
                plt.ylabel('Presentation')
                plt.title(f"{sample_interval} First {max_stim} Presentations")
                plt.savefig(f'explore/stim_timing_example.png')
                plt.close()
        
        return nwb, stim_intervals
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return None, []

# Use the session file we've already been examining
session_asset_id = "fbcd4fe5-7107-41b2-b154-b67f783f23dc"
nwb, stim_intervals = get_stimulus_info(session_asset_id)

print(f"Session ID: {nwb.session_id}")

# Get the list of stimulus keys
stimulus_templates = list(nwb.stimulus_template.keys())
print(f"\nFound {len(stimulus_templates)} stimulus templates")
for i, stim in enumerate(stimulus_templates[:10]):  # Print first 10
    print(f"{i+1}. {stim}")
if len(stimulus_templates) > 10:
    print(f"... and {len(stimulus_templates) - 10} more")

# Get the list of presentation intervals from nwb.intervals
presentation_intervals = [k for k in nwb.intervals.keys() if "presentations" in k]
print(f"\nFound {len(presentation_intervals)} stimulus presentation intervals")
for i, interval in enumerate(presentation_intervals[:10]):  # Print first 10
    print(f"{i+1}. {interval}")
if len(presentation_intervals) > 10:
    print(f"... and {len(presentation_intervals) - 10} more")

# Examine a few different types of stimuli
stimulus_types_to_examine = [
    'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations',  # Standard grating
    'natmovie_EagleSwooping1_540x960Full_584x460Active_presentations',  # Natural movie
    'Disco2SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations',  # Disco Bar
    'Ring_Wd15_Vel2_Bndry1_Cntst0_loop_presentations'  # Ring pattern
]

# For each stimulus type, get presentation times and plot
plt.figure(figsize=(15, 10))
for i, stim_type in enumerate(stimulus_types_to_examine):
    if stim_type in nwb.intervals:
        # Get presentation dataframe
        df = nwb.intervals[stim_type].to_dataframe()
        
        # Extract start and stop times
        start_times = df['start_time'].values
        durations = df['stop_time'].values - df['start_time'].values
        
        # Sort by start time just in case
        sort_idx = np.argsort(start_times)
        start_times = start_times[sort_idx]
        durations = durations[sort_idx]
        
        # Plot presentation times
        plt.subplot(len(stimulus_types_to_examine), 1, i+1)
        plt.eventplot(start_times, lineoffsets=0.5, linelengths=0.5, 
                      linewidths=2, color='blue')
        plt.title(f"{stim_type} ({len(start_times)} presentations)")
        plt.xlabel('Time (seconds)')
        plt.ylabel('Presentations')
        
        # Print some statistics
        print(f"\n{stim_type}:")
        print(f"  Number of presentations: {len(start_times)}")
        print(f"  Total presentation time: {np.sum(durations):.2f} seconds")
        print(f"  Average presentation duration: {np.mean(durations):.4f} seconds")
        
        # Check if there are stimulus_block values (different sections of the stimulus)
        if 'stimulus_block' in df.columns:
            num_blocks = len(df['stimulus_block'].unique())
            print(f"  Number of stimulus blocks: {num_blocks}")
    else:
        print(f"\nStimulus type {stim_type} not found in intervals")

plt.tight_layout()
plt.savefig('explore/stimulus_presentations.png')
plt.close()

# Function to display a frame from a stimulus
def display_stimulus_frame(stim_name, frame_idx=0):
    if stim_name in nwb.stimulus_template:
        stim = nwb.stimulus_template[stim_name]
        
        # Get the shape to determine if it's grayscale or RGB
        if len(stim.data.shape) == 3:
            # Grayscale stimulus (height, width, frames)
            frame = stim.data[:, :, frame_idx]
            cmap = 'gray'
            is_rgb = False
        else:
            # Color stimulus (height, width, frames, channels)
            frame = stim.data[:, :, frame_idx, :]
            cmap = None
            is_rgb = True
        
        print(f"\n{stim_name}:")
        print(f"  Shape: {stim.data.shape}")
        print(f"  Frame size: {frame.shape}")
        
        plt.figure(figsize=(8, 8))
        if is_rgb:
            plt.imshow(frame)
        else:
            plt.imshow(frame, cmap=cmap)
        plt.title(f"{stim_name} - Frame {frame_idx}")
        plt.colorbar()
        plt.savefig(f'explore/{stim_name.split("_")[0]}_frame{frame_idx}.png')
        plt.close()
        
        return frame
    else:
        print(f"Stimulus {stim_name} not found in stimulus_template")
        return None

# Display a frame from each type of stimulus
for stim_type in stimulus_types_to_examine:
    # Extract the stimulus template name from the presentation name
    template_name = stim_type.replace('_presentations', '')
    # For some stims, show multiple frames to see the pattern
    frames_to_show = [0]
    if "SAC" in template_name or "natmovie" in template_name:
        frames_to_show += [60, 120]  # Show a few more frames for these types
    
    for frame_idx in frames_to_show:
        display_stimulus_frame(template_name, frame_idx)

# Display a sequence of frames to understand the stimulus motion
def display_stimulus_sequence(stim_name, start_frame=0, num_frames=8, step=30):
    if stim_name in nwb.stimulus_template:
        stim = nwb.stimulus_template[stim_name]
        
        # Determine if it's grayscale or RGB
        if len(stim.data.shape) == 3:
            # Grayscale stimulus (height, width, frames)
            is_rgb = False
        else:
            # Color stimulus (height, width, frames, channels)
            is_rgb = True
        
        # Create a grid of frames
        rows = int(np.ceil(num_frames / 4))
        cols = min(num_frames, 4)
        
        plt.figure(figsize=(cols*4, rows*4))
        for i in range(num_frames):
            frame_idx = start_frame + i * step
            if frame_idx >= stim.data.shape[2 if is_rgb else 2]:
                break
                
            plt.subplot(rows, cols, i+1)
            if is_rgb:
                plt.imshow(stim.data[:, :, frame_idx, :])
            else:
                plt.imshow(stim.data[:, :, frame_idx], cmap='gray')
            plt.title(f"Frame {frame_idx}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'explore/{stim_name.split("_")[0]}_sequence.png')
        plt.close()
        
        print(f"\nSaved sequence of {num_frames} frames from {stim_name}")
    else:
        print(f"Stimulus {stim_name} not found in stimulus_template")

# Show sequences for selected stimuli
display_stimulus_sequence("SAC_Wd15_Vel2_Bndry1_Cntst0_loop", num_frames=8, step=30)
display_stimulus_sequence("natmovie_EagleSwooping1_540x960Full_584x460Active", num_frames=8, step=20)

# Compare different stimulus variants
sac_variants = [name for name in stimulus_templates if name.startswith("SAC_") and "loop" in name]
print(f"\nFound {len(sac_variants)} SAC stimulus variants")

if len(sac_variants) > 1:
    # Compare first frame of different SAC variants
    plt.figure(figsize=(15, 15))
    num_variants = min(9, len(sac_variants))
    rows = int(np.ceil(np.sqrt(num_variants)))
    cols = int(np.ceil(num_variants / rows))
    
    for i, variant in enumerate(sac_variants[:num_variants]):
        stim = nwb.stimulus_template[variant]
        
        plt.subplot(rows, cols, i+1)
        if len(stim.data.shape) == 3:
            # Grayscale
            plt.imshow(stim.data[:, :, 0], cmap='gray')
        else:
            # Color
            plt.imshow(stim.data[:, :, 0, :])
        plt.title(variant.split('_')[1:4])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('explore/sac_variants_comparison.png')
    plt.close()
    
    # Extract key parameters from the names
    params = []
    for variant in sac_variants:
        parts = variant.split('_')
        width = parts[1] if len(parts) > 1 else 'N/A'  # Wd15, Wd45
        vel = parts[2] if len(parts) > 2 else 'N/A'    # Vel2, Vel8
        bndry = parts[3] if len(parts) > 3 else 'N/A'  # Bndry1, Bndry2, Bndry3
        cntst = parts[4] if len(parts) > 4 else 'N/A'  # Cntst0, Cntst1
        params.append({
            'name': variant,
            'width': width,
            'velocity': vel,
            'boundary': bndry,
            'contrast': cntst
        })
    
    # Print the parameter variations
    param_df = pd.DataFrame(params)
    print("\nParameter variations in SAC stimuli:")
    for col in ['width', 'velocity', 'boundary', 'contrast']:
        if col in param_df.columns:
            unique_vals = param_df[col].unique()
            print(f"  {col}: {', '.join(unique_vals)}")

print("\nAnalysis complete - see output plots in the explore/ directory")