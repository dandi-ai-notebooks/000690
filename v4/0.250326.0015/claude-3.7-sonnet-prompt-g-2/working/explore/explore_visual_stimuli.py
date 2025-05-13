"""
Explore visual stimuli from the Vision2Hippocampus project.
This script visualizes sample frames from different stimulus types used in the experiment.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load the image NWB file
url = "https://api.dandiarchive.org/api/assets/cbc64387-19b9-494a-a8fa-04d3207f7ffb/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get stimulus template information
stim_templates = nwb.stimulus_template

# Print available stimulus templates
print("Available stimulus templates:")
for name in stim_templates.keys():
    stim = stim_templates[name]
    if hasattr(stim, 'data'):
        shape = stim.data.shape
        print(f"- {name}: shape {shape}, rate {stim.rate} Hz")

# Dictionary to categorize the stimuli
stimulus_categories = {
    'simple_oriented': ['SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations',  # Standard bar
                        'SAC_Wd45_Vel2_Bndry1_Cntst0_loop_presentations',  # Wide bar
                        'SAC_Wd15_Vel8_Bndry1_Cntst0_loop_presentations',  # Fast bar
                        'SAC_Wd15_Vel2_Bndry1_Cntst1_loop_presentations'], # Contrast bar
    'complex_shape': ['Disk_Wd15_Vel2_Bndry1_Cntst0_loop_presentations',
                     'Ring_Wd15_Vel2_Bndry1_Cntst0_loop_presentations'],
    'boundary_variations': ['SAC_Wd15_Vel2_Bndry2_Cntst0_loop_presentations',
                           'SAC_Wd15_Vel2_Bndry3_Cntst0_loop_presentations',
                           'SAC_Wd15_Vel2_Bndry2_Cntst0_oneway_presentations'],
    'natural_movies': ['natmovie_EagleSwooping1_540x960Full_584x460Active_presentations',
                      'natmovie_EagleSwooping2_540x960Full_584x460Active_presentations',
                      'natmovie_SnakeOnARoad_540x960Full_584x460Active_presentations',
                      'natmovie_CricketsOnARock_540x960Full_584x460Active_presentations',
                      'natmovie_Squirreland3Mice_540x960Full_584x460Active_presentations']
}

# Function to sample and plot frames from different stimulus types
def plot_stimulus_examples(stim_templates, stimulus_keys, title, num_frames=3, figsize=(15, 10), filename=None):
    """
    Plot example frames from different stimulus types
    
    Parameters:
    -----------
    stim_templates : pynwb object
        Stimulus templates from NWB file
    stimulus_keys : list
        List of stimulus keys to plot
    title : str
        Title for the plot
    num_frames : int
        Number of frames to plot for each stimulus
    figsize : tuple
        Figure size
    filename : str
        If provided, save the figure to this filename
    """
    n_stims = len(stimulus_keys)
    
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(n_stims, num_frames, figure=fig)
    
    for i, stim_key in enumerate(stimulus_keys):
        if stim_key not in stim_templates:
            print(f"Warning: {stim_key} not found in stimulus templates")
            continue
            
        stim = stim_templates[stim_key]
        
        # Determine frame indices to plot
        if hasattr(stim, 'data'):
            if len(stim.data.shape) == 4:  # Color images [width, height, frames, channels]
                frame_indices = np.linspace(0, stim.data.shape[2]-1, num_frames, dtype=int)
                for j, frame_idx in enumerate(frame_indices):
                    ax = fig.add_subplot(gs[i, j])
                    ax.imshow(stim.data[:, :, frame_idx, :])
                    ax.set_title(f"Frame {frame_idx}")
                    ax.axis('off')
            elif len(stim.data.shape) == 3:  # Grayscale images [width, height, frames]
                frame_indices = np.linspace(0, stim.data.shape[2]-1, num_frames, dtype=int)
                for j, frame_idx in enumerate(frame_indices):
                    ax = fig.add_subplot(gs[i, j])
                    ax.imshow(stim.data[:, :, frame_idx], cmap='gray')
                    ax.set_title(f"Frame {frame_idx}")
                    ax.axis('off')
            else:
                print(f"Unexpected shape for {stim_key}: {stim.data.shape}")
                continue
            
            # Add row label
            if j == 0:
                ax.set_ylabel(stim_key.split('_')[0], fontsize=12)
                
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"Saved figure to {filename}")
    
# Plot examples from each stimulus category
for category, stim_keys in stimulus_categories.items():
    # Filter for keys that actually exist in the data
    existing_keys = [k for k in stim_keys if k in stim_templates.keys()]
    
    if existing_keys:
        # Take first 3 keys at most
        keys_to_plot = existing_keys[:3]
        plot_stimulus_examples(
            stim_templates, 
            keys_to_plot, 
            f"{category.replace('_', ' ').title()} Stimuli",
            filename=f"explore/stimuli_{category}.png"
        )

# Display all stimulus presentations for a given subject
print("\nStimulus presentation intervals:")
intervals = nwb.intervals

# Count presentation intervals by stimulus type
presentation_counts = {}
for name in intervals.keys():
    if "_presentations" in name:
        interval = intervals[name]
        count = len(interval.start_time[:])
        presentation_counts[name] = count

print("\nNumber of stimulus presentations:")
for name, count in presentation_counts.items():
    print(f"- {name}: {count} presentations")

print("\nDone! Stimulus visualizations saved to explore directory.")