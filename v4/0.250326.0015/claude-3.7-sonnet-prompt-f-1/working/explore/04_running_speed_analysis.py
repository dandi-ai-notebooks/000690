"""
This script explores the running speed data and its potential correlation with neural activity.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy import stats

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from URL: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract running speed data
print("\nExtracting running speed data...")
if 'running' in nwb.processing:
    running_module = nwb.processing['running']
    if 'running_speed' in running_module.data_interfaces:
        running_speed = running_module.data_interfaces['running_speed']
        
        # Extract running speed values and timestamps (just a sample to avoid memory issues)
        # Take a limited sample to avoid memory issues
        max_samples = 10000
        speed_data = running_speed.data[:max_samples]
        speed_timestamps = running_speed.timestamps[:max_samples]
        
        print(f"Running speed data shape: {speed_data.shape}")
        print(f"Running speed timestamps shape: {speed_timestamps.shape}")
        
        # Basic statistics
        print(f"Mean running speed: {np.mean(speed_data):.2f} cm/s")
        print(f"Median running speed: {np.median(speed_data):.2f} cm/s")
        print(f"Min running speed: {np.min(speed_data):.2f} cm/s")
        print(f"Max running speed: {np.max(speed_data):.2f} cm/s")
        
        # Plot running speed over time
        plt.figure(figsize=(12, 6))
        plt.plot(speed_timestamps, speed_data, 'b-', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Running Speed (cm/s)')
        plt.title('Mouse Running Speed Over Time')
        plt.grid(True, alpha=0.3)
        plt.savefig('explore/running_speed_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot histogram of running speeds
        plt.figure(figsize=(10, 6))
        plt.hist(speed_data, bins=50, alpha=0.7, color='g')
        plt.xlabel('Running Speed (cm/s)')
        plt.ylabel('Count')
        plt.title('Distribution of Running Speeds')
        plt.grid(True, alpha=0.3)
        plt.savefig('explore/running_speed_histogram.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Now, try to correlate running speed with neural firing rates
        print("\nAnalyzing correlation between running speed and neural activity...")
        
        # Get units data
        units_firing_rates = {}
        
        # Limit to a small number of neurons to avoid memory issues
        num_units = 20
        
        # Select time windows for analysis based on speed timestamps
        time_window_size = 5.0  # seconds
        num_windows = 5
        window_step = (speed_timestamps[-1] - speed_timestamps[0]) / (num_windows + 1)
        
        # Extract spike counts in each window for each unit
        windows = []
        window_speeds = []
        unit_counts = defaultdict(list)
        
        for i in range(num_windows):
            window_start = speed_timestamps[0] + i * window_step
            window_end = window_start + time_window_size
            window = (window_start, window_end)
            windows.append(window)
            
            # Find average running speed in this window
            in_window = (speed_timestamps >= window_start) & (speed_timestamps < window_end)
            window_speed = np.mean(speed_data[in_window]) if np.any(in_window) else 0
            window_speeds.append(window_speed)
            
            # Count spikes for each unit in this window
            for unit_id in range(num_units):
                if unit_id < len(nwb.units):
                    spike_times = nwb.units['spike_times'][unit_id]
                    spike_count = np.sum((spike_times >= window_start) & (spike_times < window_end))
                    unit_counts[unit_id].append(spike_count)
        
        # Convert spike counts to firing rates
        window_firing_rates = {}
        for unit_id, counts in unit_counts.items():
            window_firing_rates[unit_id] = [count / time_window_size for count in counts]
            
        # Calculate correlation between running speed and firing rate for each unit
        correlations = {}
        p_values = {}
        
        for unit_id, rates in window_firing_rates.items():
            if len(rates) > 0:  # Ensure we have data
                corr, p_value = stats.pearsonr(window_speeds, rates)
                correlations[unit_id] = corr
                p_values[unit_id] = p_value
        
        # Print correlation results
        print("\nCorrelations between running speed and firing rates:")
        for unit_id, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
            p = p_values[unit_id]
            print(f"Unit {unit_id}: Correlation = {corr:.3f}, p-value = {p:.3f}")
        
        # Create a scatter plot for the unit with the strongest correlation
        if correlations:
            strongest_unit = max(correlations.items(), key=lambda x: abs(x[1]))[0]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(window_speeds, window_firing_rates[strongest_unit], alpha=0.7, s=50)
            
            # Add regression line
            m, b = np.polyfit(window_speeds, window_firing_rates[strongest_unit], 1)
            plt.plot(window_speeds, m * np.array(window_speeds) + b, 'r-')
            
            plt.xlabel('Running Speed (cm/s)')
            plt.ylabel('Firing Rate (spikes/s)')
            plt.title(f'Correlation Between Running Speed and Neural Activity (Unit {strongest_unit})')
            plt.grid(True, alpha=0.3)
            plt.savefig('explore/running_speed_correlation.png', dpi=150, bbox_inches='tight')
            plt.close()
    else:
        print("No running_speed data interface found.")
else:
    print("No running module found in processing.")

print("\nRunning speed analysis complete!")
io.close()