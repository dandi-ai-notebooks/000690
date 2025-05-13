# This script explores the visual stimuli in the image NWB file

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set seaborn style for static plots (not for image plotting)
import seaborn as sns
sns.set_theme()

print("Loading image NWB file...")
url = "https://api.dandiarchive.org/api/assets/cbc64387-19b9-494a-a8fa-04d3207f7ffb/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract information about available stimuli
print("Available stimulus templates:")
for key in nwb.stimulus_template.keys():
    template = nwb.stimulus_template[key]
    if hasattr(template, 'data'):
        shape = template.data.shape
        print(f"  {key}: shape={shape}")

# Choose one stimulus to analyze
stim_keys = list(nwb.stimulus_template.keys())
selected_stim_key = 'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations'

if selected_stim_key in nwb.stimulus_template:
    stim_template = nwb.stimulus_template[selected_stim_key]
    print(f"\nExtracting frames from {selected_stim_key}")
    
    # Get information about the stimulus
    stim_shape = stim_template.data.shape
    print(f"Stimulus shape: {stim_shape}")
    
    # Extract a few frames from the stimulus
    num_frames = min(5, stim_shape[2])
    interval = stim_shape[2] // num_frames
    
    plt.figure(figsize=(15, 3))
    
    for i in range(num_frames):
        frame_idx = i * interval
        frame = stim_template.data[:, :, frame_idx]
        
        plt.subplot(1, num_frames, i+1)
        plt.imshow(frame, cmap='gray')
        plt.title(f"Frame {frame_idx}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('explore/stimulus_frames.png')
    
    # Extract presentation times for this stimulus
    if selected_stim_key in nwb.intervals:
        stim_presentations = nwb.intervals[selected_stim_key]
        start_times = stim_presentations.start_time[:]
        stop_times = stim_presentations.stop_time[:]
        
        # Plot the start times of stimulus presentations
        plt.figure(figsize=(10, 4))
        plt.hist(start_times, bins=50)
        plt.title(f'Distribution of Stimulus Presentation Start Times')
        plt.xlabel('Time (s)')
        plt.ylabel('Count')
        plt.savefig('explore/stimulus_timing.png')
        
        # Calculate statistics 
        durations = stop_times - start_times
        mean_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        print(f"\nStimulus presentation statistics:")
        print(f"  Number of presentations: {len(start_times)}")
        print(f"  Mean duration: {mean_duration:.4f}s")
        print(f"  Standard deviation of duration: {std_duration:.4f}s")
        print(f"  Time range: {np.min(start_times):.2f}s - {np.max(stop_times):.2f}s")
    
    # Try to create a GIF of the stimulus if data shape is reasonable
    if len(stim_shape) >= 3 and stim_shape[2] <= 30:  # Limit to max 30 frames
        print(f"\nCreating animation of stimulus...")
        fig, ax = plt.figure(figsize=(5, 5)), plt.subplot(111)
        
        frames = min(30, stim_shape[2])
        
        # Initialize with first frame
        if len(stim_shape) == 3:  # Grayscale
            im = ax.imshow(stim_template.data[:, :, 0], cmap='gray')
        else:  # Color
            im = ax.imshow(stim_template.data[:, :, 0, :])
        
        ax.axis('off')
        plt.tight_layout()
        
        # Function to update the frame
        def update(frame):
            if len(stim_shape) == 3:  # Grayscale
                im.set_array(stim_template.data[:, :, frame])
            else:  # Color
                im.set_array(stim_template.data[:, :, frame, :])
            return [im]
        
        # Create the animation
        ani = FuncAnimation(fig, update, frames=frames, blit=True)
        ani.save('explore/stimulus_animation.gif', writer='pillow', fps=10)
        plt.close()
    else:
        print(f"Stimulus has too many frames ({stim_shape[2]}) for animation.")
        
    # Look at nature movie stimulus if available
    nature_movie_keys = [k for k in nwb.stimulus_template.keys() if 'natmovie' in k]
    
    if nature_movie_keys:
        selected_movie_key = nature_movie_keys[0]
        movie_template = nwb.stimulus_template[selected_movie_key]
        movie_shape = movie_template.data.shape
        
        print(f"\nFound nature movie stimulus: {selected_movie_key}")
        print(f"Movie shape: {movie_shape}")
        
        # Extract a few frames from the movie
        num_frames = min(3, movie_shape[2])
        interval = movie_shape[2] // num_frames
        
        plt.figure(figsize=(15, 5))
        
        for i in range(num_frames):
            frame_idx = i * interval
            
            plt.subplot(1, num_frames, i+1)
            # Check if movie is grayscale or color
            if len(movie_shape) == 3:  # Grayscale
                plt.imshow(movie_template.data[:, :, frame_idx], cmap='gray')
            else:  # Color 
                plt.imshow(movie_template.data[:, :, frame_idx, :])
            
            plt.title(f"Frame {frame_idx}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('explore/nature_movie_frames.png')

print("\nAnalysis complete. See output images in explore directory.")