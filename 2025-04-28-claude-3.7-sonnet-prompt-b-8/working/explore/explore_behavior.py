# Script to explore the behavior data in the Dandiset
# This examines running wheel data and eye tracking to understand mouse behavior

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# Configure matplotlib to save
import seaborn as sns
sns.set_theme()
plt.rcParams['figure.figsize'] = (12, 8)

# Load the session NWB file
print("Loading session NWB file...")
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Session ID: {nwb.session_id}")

# Explore the running wheel data
print("\nExploring running wheel data:")
running_speed = nwb.processing['running'].data_interfaces['running_speed']
print(f"Running speed data shape: {running_speed.data.shape}")
print(f"Running speed timestamps shape: {running_speed.timestamps.shape}")

# Get some basic stats
speed_data = running_speed.data[:]
timestamps = running_speed.timestamps[:]

print(f"Duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
print(f"Average running speed: {np.mean(speed_data):.2f} cm/s")
print(f"Maximum running speed: {np.max(speed_data):.2f} cm/s")
print(f"Time spent running (speed > 5 cm/s): {np.sum(speed_data > 5) / len(speed_data) * 100:.1f}%")

# Plot the running speed over time
plt.figure(figsize=(15, 5))
plt.plot(timestamps, speed_data)
plt.xlabel('Time (s)')
plt.ylabel('Running Speed (cm/s)')
plt.title('Mouse Running Speed Over Time')
plt.grid(True, alpha=0.3)
plt.savefig('explore/running_speed.png')
plt.close()

# Histogram of running speeds
plt.figure(figsize=(10, 6))
plt.hist(speed_data, bins=50, density=True, alpha=0.7)
plt.xlabel('Running Speed (cm/s)')
plt.ylabel('Density')
plt.title('Distribution of Running Speeds')
plt.grid(True, alpha=0.3)
plt.savefig('explore/running_speed_histogram.png')
plt.close()

# Analyze running during stimuli
# Select a few stimulus types to analyze
stimulus_types = [
    'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations',
    'natmovie_EagleSwooping1_540x960Full_584x460Active_presentations'
]

def analyze_running_during_stimulus(stim_name, running_speed_series, speed_timestamps, max_presentations=None):
    """Analyze running behavior during a specific stimulus."""
    print(f"\nAnalyzing running during {stim_name}")
    
    if stim_name not in nwb.intervals:
        print(f"  Stimulus {stim_name} not found in intervals")
        return None
    
    # Get presentation times
    stim_df = nwb.intervals[stim_name].to_dataframe()
    
    if max_presentations is not None:
        stim_df = stim_df.iloc[:max_presentations]
    
    print(f"  Analyzing {len(stim_df)} stimulus presentations")
    
    # Define the pre/post time window
    pre_time = 1.0  # seconds before stimulus
    post_time = 3.0  # seconds after stimulus
    
    # Initialize arrays to store results
    all_speed_profiles = []
    speed_before = []
    speed_during = []
    
    # Process each stimulus presentation
    for _, stim in stim_df.iterrows():
        stim_start = stim['start_time']
        stim_end = stim['stop_time']
        stim_duration = stim_end - stim_start
        
        # Find the closest timestamp indices
        start_idx = np.searchsorted(speed_timestamps, stim_start - pre_time)
        end_idx = np.searchsorted(speed_timestamps, stim_end + post_time)
        
        if start_idx >= end_idx or start_idx >= len(speed_timestamps) or end_idx >= len(speed_timestamps):
            continue
        
        # Get speed data around this stimulus
        speeds = running_speed_series[start_idx:end_idx]
        times = speed_timestamps[start_idx:end_idx]
        
        # Calculate time relative to stimulus start
        rel_times = times - stim_start
        
        # Save the speed profile for this presentation
        all_speed_profiles.append((rel_times, speeds))
        
        # Calculate average speed before and during stimulus
        before_mask = (rel_times >= -pre_time) & (rel_times < 0)
        during_mask = (rel_times >= 0) & (rel_times < stim_duration)
        
        if np.any(before_mask) and np.any(during_mask):
            speed_before.append(np.mean(speeds[before_mask]))
            speed_during.append(np.mean(speeds[during_mask]))
    
    if not all_speed_profiles:
        print("  No valid stimulus presentations found")
        return None
    
    # Create a plot showing running speed aligned to stimulus onset
    plt.figure(figsize=(12, 6))
    
    # Plot individual trials (light gray)
    for rel_times, speeds in all_speed_profiles[:50]:  # Limit to 50 trials for clarity
        plt.plot(rel_times, speeds, color='gray', alpha=0.1)
    
    # Calculate and plot the average (blue)
    # First need to interpolate onto a common time base
    common_times = np.linspace(-pre_time, post_time, 500)
    interp_speeds = []
    
    for rel_times, speeds in all_speed_profiles:
        if rel_times[0] <= -pre_time and rel_times[-1] >= post_time:
            # Only use trials with full time range
            interp_speed = np.interp(common_times, rel_times, speeds)
            interp_speeds.append(interp_speed)
    
    if interp_speeds:
        avg_speed = np.mean(interp_speeds, axis=0)
        sem_speed = stats.sem(interp_speeds, axis=0)
        
        plt.plot(common_times, avg_speed, color='blue', linewidth=2, label='Mean')
        plt.fill_between(common_times, avg_speed - sem_speed, avg_speed + sem_speed, 
                         color='blue', alpha=0.2, label='SEM')
    
    # Add stimulus period indicator
    plt.axvspan(0, min(3.0, np.mean(stim_df['stop_time'] - stim_df['start_time'])), 
                color='red', alpha=0.1, label='Stimulus')
    
    plt.axvline(x=0, color='red', linestyle='--', label='Stimulus Onset')
    plt.xlabel('Time Relative to Stimulus Onset (s)')
    plt.ylabel('Running Speed (cm/s)')
    plt.title(f'Running Speed During {stim_name.split("_")[0]} Stimulus')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    stim_short_name = stim_name.split("_")[0]
    plt.savefig(f'explore/running_during_{stim_short_name}.png')
    plt.close()
    
    # Statistical analysis
    if speed_before and speed_during:
        speed_before = np.array(speed_before)
        speed_during = np.array(speed_during)
        
        # Paired t-test to see if there's a significant change
        t_stat, p_val = stats.ttest_rel(speed_before, speed_during)
        
        mean_before = np.mean(speed_before)
        mean_during = np.mean(speed_during)
        
        print("  Statistical analysis of running behavior:")
        print(f"  Mean speed before stimulus: {mean_before:.2f} cm/s")
        print(f"  Mean speed during stimulus: {mean_during:.2f} cm/s")
        print(f"  Change: {mean_during - mean_before:.2f} cm/s ({(mean_during - mean_before) / mean_before * 100:.1f}%)")
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        
        if p_val < 0.05:
            effect = "increase" if mean_during > mean_before else "decrease"
            print(f"  Result: Significant {effect} in running speed during stimulus (p<0.05)")
        else:
            print(f"  Result: No significant change in running speed during stimulus (p>0.05)")
    
    return {
        'speed_before': speed_before,
        'speed_during': speed_during,
        'speed_profiles': all_speed_profiles
    }

# Analyze running during each stimulus type
results = {}
for stim_type in stimulus_types:
    results[stim_type] = analyze_running_during_stimulus(
        stim_type, 
        running_speed.data[:], 
        running_speed.timestamps[:],
        max_presentations=200
    )

# Analyze pupil data (if available)
print("\nExploring eye tracking data:")
if 'EyeTracking' in nwb.acquisition:
    eye_tracking = nwb.acquisition['EyeTracking']
    pupil_tracking = eye_tracking.pupil_tracking
    
    print(f"Pupil tracking data shape: {pupil_tracking.data.shape}")
    print(f"Pupil tracking timestamps shape: {pupil_tracking.timestamps.shape}")
    
    pupil_data = pupil_tracking.area[:]
    pupil_timestamps = pupil_tracking.timestamps[:]
    
    # Basic stats
    print(f"Mean pupil area: {np.mean(pupil_data):.4f}")
    print(f"Std pupil area: {np.std(pupil_data):.4f}")
    
    # Handle NaN or inf values
    valid_mask = ~np.isnan(pupil_data) & ~np.isinf(pupil_data)
    valid_pupil = pupil_data[valid_mask]
    valid_timestamps = pupil_timestamps[valid_mask]
    
    # Plot pupil size over time (first 500 seconds for clarity)
    plt.figure(figsize=(15, 5))
    max_time = 500  # seconds
    time_mask = valid_timestamps < max_time
    plt.plot(valid_timestamps[time_mask], valid_pupil[time_mask])
    plt.xlabel('Time (s)')
    plt.ylabel('Pupil Area')
    plt.title('Pupil Size Over Time (First 500 seconds)')
    plt.grid(True, alpha=0.3)
    plt.savefig('explore/pupil_size.png')
    plt.close()
    
    # Create a histogram of pupil sizes
    plt.figure(figsize=(10, 6))
    plt.hist(valid_pupil, bins=50, alpha=0.7, density=True)
    plt.xlabel('Pupil Area')
    plt.ylabel('Density')
    plt.title('Distribution of Pupil Sizes')
    plt.grid(True, alpha=0.3)
    plt.savefig('explore/pupil_histogram.png')
    plt.close()
    
    # Check correlation between pupil size and running speed
    # First, interpolate pupil data to match running speed timestamps
    running_timestamps = running_speed.timestamps[:]
    interp_pupil = np.interp(
        running_timestamps, 
        valid_timestamps,
        valid_pupil
    )
    
    # Calculate correlation
    mask = (~np.isnan(interp_pupil)) & (~np.isnan(running_speed.data[:]))
    running_corr = np.corrcoef(interp_pupil[mask], running_speed.data[:][mask])[0, 1]
    
    print(f"Correlation between pupil size and running speed: {running_corr:.3f}")
    
    # Scatter plot of pupil size vs running speed (subsample for clarity)
    plt.figure(figsize=(10, 10))
    subsample = 5000
    if len(mask) > subsample:
        indices = np.random.choice(np.where(mask)[0], subsample, replace=False)
    else:
        indices = np.where(mask)[0]
    
    plt.scatter(
        interp_pupil[indices], 
        running_speed.data[:][indices], 
        alpha=0.3,
        s=3
    )
    plt.xlabel('Pupil Area')
    plt.ylabel('Running Speed (cm/s)')
    plt.title(f'Pupil Size vs Running Speed (r={running_corr:.3f})')
    plt.grid(True, alpha=0.3)
    plt.savefig('explore/pupil_vs_running.png')
    plt.close()
else:
    print("No eye tracking data found in this file")

print("\nAnalysis complete - see output plots in the explore/ directory")