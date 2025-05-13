# This script explores the running speed data and how it correlates with stimulus presentations

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Set seaborn style
import seaborn as sns
sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract running speed data
running_speed_data = nwb.processing['running'].data_interfaces['running_speed']
running_speeds = running_speed_data.data[:]
running_timestamps = running_speed_data.timestamps[:]

# Sample a subset of the data (10% of the data)
sample_size = len(running_speeds) // 10
sample_indices = np.linspace(0, len(running_speeds)-1, sample_size, dtype=int)
sampled_speeds = running_speeds[sample_indices]
sampled_timestamps = running_timestamps[sample_indices]

# Plot running speed over a portion of the session
plt.figure(figsize=(12, 6))
plt.plot(sampled_timestamps, sampled_speeds)
plt.title('Running Speed vs Time (Sampled)')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.savefig('explore/running_speed.png')

# Get stimulus presentation information for one stimulus type
stim_key = 'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations'
if stim_key in nwb.intervals:
    stim_presentations = nwb.intervals[stim_key]
    start_times = stim_presentations.start_time[:]
    stop_times = stim_presentations.stop_time[:]
    
    # Plot stimulus timing with running speed for a time window
    time_window_start = start_times[0]
    time_window_end = max(time_window_start + 60, stop_times[min(len(stop_times)-1, 100)])
    
    mask = (sampled_timestamps >= time_window_start) & (sampled_timestamps <= time_window_end)
    window_times = sampled_timestamps[mask]
    window_speeds = sampled_speeds[mask]
    
    # Find stimulus presentations in this window
    stim_mask = (start_times >= time_window_start) & (start_times <= time_window_end)
    window_stim_starts = start_times[stim_mask]
    window_stim_stops = stop_times[stim_mask]
    
    plt.figure(figsize=(12, 6))
    plt.plot(window_times, window_speeds, label='Running Speed')
    
    # Add stimulus presentation periods
    for start, stop in zip(window_stim_starts, window_stim_stops):
        plt.axvspan(start, stop, color='red', alpha=0.2)
    
    plt.title(f'Running Speed with {stim_key} Stimulus Presentations')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (cm/s)')
    plt.legend()
    plt.savefig('explore/running_speed_with_stim.png')

print("Analysis complete. See output images in explore directory.")