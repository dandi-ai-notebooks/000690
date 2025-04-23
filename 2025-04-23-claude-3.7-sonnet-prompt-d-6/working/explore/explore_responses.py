"""
This script explores the relationship between neuronal activity and visual stimuli
in Dandiset 000690. It loads the main NWB file and extracts spike times for selected
units, aligning them with the stimulus presentation intervals.
"""

import numpy as np
import h5py
import remfile
import pynwb
import matplotlib.pyplot as plt
import pandas as pd
import time

# Set the start time to measure execution
start_time = time.time()

# URL of the main file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"

print("Loading NWB file from URL...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"File loaded in {time.time() - start_time:.2f} seconds")
print(f"Dandiset identifier: {nwb.identifier}")
print(f"Session date: {nwb.session_start_time}")
print(f"Session description: {nwb.session_description}")

# Get information about the units
print("\nExtracting unit information...")
units_df = nwb.units.to_dataframe()
print(f"Total number of units: {len(units_df)}")

# Print quality distribution
print("\nUnit quality distribution:")
quality_counts = units_df['quality'].value_counts()
print(quality_counts)

# Get good quality units only
good_units = units_df[units_df['quality'] == 'good'].copy()
print(f"Number of good quality units: {len(good_units)}")

# Print basic stats for good units
print("\nGood units statistics:")
print(f"Mean firing rate: {good_units['firing_rate'].mean():.2f} Hz")
print(f"Median firing rate: {good_units['firing_rate'].median():.2f} Hz")
print(f"Max firing rate: {good_units['firing_rate'].max():.2f} Hz")
print(f"Min firing rate: {good_units['firing_rate'].min():.2f} Hz")

# Get stimulus presentations for the first stimulus
print("\nAvailable stimulus presentations:")
for interval_name in nwb.intervals:
    try:
        n_presentations = len(nwb.intervals[interval_name])
        print(f"- {interval_name}: {n_presentations} presentations")
    except Exception as e:
        print(f"- {interval_name}: Error accessing presentation count")

# Select a stimulus type for analysis
stimulus_type = "SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations"
print(f"\nAnalyzing stimulus: {stimulus_type}")

# Get the stimulus presentation dataframe
try:
    stim_presentations = nwb.intervals[stimulus_type].to_dataframe()
    print(f"Number of presentations: {len(stim_presentations)}")
    print("First few presentations:")
    print(stim_presentations.head())
    
    # Plot stimulus start and stop times for first few presentations
    plt.figure(figsize=(10, 6))
    for i in range(min(20, len(stim_presentations))):
        plt.plot([stim_presentations.iloc[i]['start_time'], stim_presentations.iloc[i]['stop_time']], 
                [i, i], 'b-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Presentation Index')
    plt.title(f'First 20 {stimulus_type} Presentations')
    plt.tight_layout()
    plt.savefig('stimulus_presentations.png')
    
    # Analyze a sample unit response to stimulus
    # Select a unit with high firing rate for better visualization
    sorted_units = good_units.sort_values('firing_rate', ascending=False)
    sample_unit_id = sorted_units.index[0]
    sample_unit = sorted_units.loc[sample_unit_id]
    
    print(f"\nAnalyzing sample unit {sample_unit_id}")
    print(f"Firing rate: {sample_unit['firing_rate']:.2f} Hz")
    
    # Get spike times for this unit
    spike_times = nwb.units[sample_unit_id]['spike_times']
    print(f"Number of spikes: {len(spike_times)}")
    
    # Align spikes with stimulus presentations
    # Take the first 10 presentations for analysis
    n_presentations = min(10, len(stim_presentations))
    window_pre = 0.5  # time before stimulus onset (seconds)
    window_post = 2.0  # time after stimulus onset (seconds)
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_presentations):
        stim_start = stim_presentations.iloc[i]['start_time']
        stim_end = stim_presentations.iloc[i]['stop_time']
        
        # Find spikes within the window around stimulus start
        spikes_in_window = spike_times[
            (spike_times >= stim_start - window_pre) & 
            (spike_times <= stim_start + window_post)
        ]
        
        # Plot raster
        if len(spikes_in_window) > 0:
            plt.subplot(n_presentations, 1, i+1)
            plt.vlines(spikes_in_window - stim_start, -0.5, 0.5, colors='k')
            plt.axvline(x=0, color='r', linestyle='--')  # Stimulus onset
            plt.axvline(x=stim_end-stim_start, color='g', linestyle='--')  # Stimulus offset
            plt.xlim(-window_pre, window_post)
            plt.ylabel(f'Trial {i+1}')
            
    plt.xlabel('Time from stimulus onset (s)')
    plt.suptitle(f'Unit {sample_unit_id} response to {stimulus_type}')
    plt.tight_layout()
    plt.savefig('unit_response_raster.png')
    
    # Create a PSTH for all trials combined
    all_spikes = []
    
    for i in range(min(100, len(stim_presentations))):
        stim_start = stim_presentations.iloc[i]['start_time']
        
        # Find spikes within the window around stimulus start
        spikes_in_window = spike_times[
            (spike_times >= stim_start - window_pre) & 
            (spike_times <= stim_start + window_post)
        ]
        
        # Add spikes to list, aligned to stimulus onset
        all_spikes.extend(spikes_in_window - stim_start)
    
    # Create PSTH
    plt.figure(figsize=(10, 6))
    plt.hist(all_spikes, bins=100, range=(-window_pre, window_post), density=False)
    plt.axvline(x=0, color='r', linestyle='--')  # Stimulus onset
    plt.axvline(x=stim_presentations.iloc[0]['stop_time'] - stim_presentations.iloc[0]['start_time'], 
               color='g', linestyle='--')  # Stimulus offset
    plt.xlabel('Time from stimulus onset (s)')
    plt.ylabel('Spike Count')
    plt.title(f'PSTH for Unit {sample_unit_id} response to {stimulus_type}')
    plt.tight_layout()
    plt.savefig('unit_psth.png')
    
except Exception as e:
    print(f"Error analyzing stimulus presentations: {e}")

print(f"\nScript completed in {time.time() - start_time:.2f} seconds")