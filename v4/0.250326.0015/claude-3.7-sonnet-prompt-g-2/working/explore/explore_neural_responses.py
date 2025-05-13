"""
Explore neural responses to visual stimuli in the Vision2Hippocampus project.
This script analyzes how neural units respond to different visual stimuli and 
displays spike rasters and firing rates for selected units.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the main NWB file
print("Loading main NWB file...")
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about units
units_df = nwb.units.to_dataframe()
print(f"Total number of units: {len(units_df)}")
print(f"Unit quality counts: {units_df['quality'].value_counts().to_dict()}")

# Get location information for each unit
print("\nGetting unit locations...")
unit_locations = {}
electrode_df = nwb.electrodes.to_dataframe()

# Map electrode IDs to brain regions
electrode_to_location = {i: loc for i, loc in zip(electrode_df.index, electrode_df['location'])}

# Get peak channel for each unit (if available)
if 'peak_channel_id' in units_df.columns:
    for unit_id, peak_channel in zip(units_df.index, units_df['peak_channel_id']):
        if peak_channel in electrode_to_location:
            unit_locations[unit_id] = electrode_to_location[peak_channel]
        else:
            unit_locations[unit_id] = "unknown"

# Count units per brain region
units_per_region = defaultdict(int)
if unit_locations:
    for location in unit_locations.values():
        units_per_region[location] += 1
    
    print("\nUnits per brain region:")
    for region, count in sorted(units_per_region.items(), key=lambda x: x[1], reverse=True):
        print(f"{region}: {count} units")
else:
    print("Could not determine unit locations based on available data")

# Get stimulus presentation intervals
print("\nGetting stimulus presentation intervals...")
intervals = nwb.intervals

# Select a few stimulus types to analyze
stim_types = [
    'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations',  # Simple oriented bar
    'Disk_Wd15_Vel2_Bndry1_Cntst0_loop_presentations',  # Complex shape
    'natmovie_EagleSwooping1_540x960Full_584x460Active_presentations'  # Natural movie
]

# Function to analyze unit responses to stimuli
def analyze_unit_responses(unit_id, stimulus_intervals, window=[-0.5, 1.0]):
    """
    Analyze how a specific unit responds to given stimulus presentations
    
    Parameters:
    -----------
    unit_id : int
        ID of the unit to analyze
    stimulus_intervals : TimeIntervals
        Intervals of stimulus presentations
    window : list
        Time window around stimulus onset [start, end] in seconds
    
    Returns:
    --------
    tuple
        (psth, raster) - peristimulus time histogram and spike raster
    """
    unit_spike_times = units_df.loc[unit_id, 'spike_times']
    
    if len(unit_spike_times) == 0:
        print(f"Unit {unit_id} has no spike times")
        return None, None
    
    # Get stimulus onset times
    stim_onsets = stimulus_intervals.start_time[:]
    
    # Take at most 100 presentations for efficiency
    if len(stim_onsets) > 100:
        indices = np.random.choice(len(stim_onsets), 100, replace=False)
        stim_onsets = stim_onsets[indices]
    
    # Prepare binning for PSTH
    bin_edges = np.linspace(window[0], window[1], 50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    all_counts = []
    raster_data = []
    
    # Compute spike counts around each stimulus onset
    for onset in stim_onsets:
        # Find spikes in the window around this stimulus
        window_spikes = unit_spike_times[
            (unit_spike_times >= onset + window[0]) & 
            (unit_spike_times <= onset + window[1])
        ] - onset  # Align to stimulus onset
        
        # Store raster data
        raster_data.append(window_spikes)
        
        # Bin spikes for PSTH
        counts, _ = np.histogram(window_spikes, bins=bin_edges)
        all_counts.append(counts)
    
    # Average across trials
    mean_counts = np.mean(all_counts, axis=0)
    
    # Convert to firing rate (spikes/s)
    bin_width = bin_edges[1] - bin_edges[0]
    firing_rate = mean_counts / bin_width
    
    return (bin_centers, firing_rate), raster_data

# Select some good units to analyze
if 'quality' in units_df.columns:
    # Select a few good units that have sufficient firing rates
    good_units = units_df[(units_df['quality'] == 'good') & 
                          (units_df['firing_rate'] > 1)].index.tolist()[:10]
else:
    # If no quality metric, just select units with higher firing rates
    good_units = units_df.sort_values('firing_rate', ascending=False).index.tolist()[:10]

print(f"\nSelected {len(good_units)} units for analysis")

# For each selected unit, plot responses to different stimuli
for unit_idx, unit_id in enumerate(good_units[:5]):  # Limit to 5 units for this exploration
    unit_info = units_df.loc[unit_id]
    unit_location = unit_locations.get(unit_id, "unknown") if unit_locations else "unknown"
    
    print(f"\nAnalyzing unit {unit_id} (region: {unit_location}, firing rate: {unit_info.get('firing_rate', 'unknown')} Hz)")
    
    # Create a figure with subplots for each stimulus type
    fig, axes = plt.subplots(len(stim_types), 2, figsize=(12, 4*len(stim_types)), 
                            gridspec_kw={'width_ratios': [3, 1]})
    
    for i, stim_name in enumerate(stim_types):
        if stim_name not in intervals:
            print(f"  Stimulus {stim_name} not found in intervals")
            continue
            
        print(f"  Analyzing responses to {stim_name}")
        
        # Get stimulus intervals
        stim_intervals = intervals[stim_name]
        
        # Analyze unit responses
        psth_data, raster_data = analyze_unit_responses(unit_id, stim_intervals)
        
        if psth_data is None:
            print(f"  No response data available for unit {unit_id} to stimulus {stim_name}")
            continue
        
        # Plot PSTH (firing rate)
        bin_centers, firing_rate = psth_data
        ax1 = axes[i, 0]
        ax1.bar(bin_centers, firing_rate, width=(bin_centers[1] - bin_centers[0]), 
                alpha=0.7, color='blue')
        ax1.axvline(x=0, color='red', linestyle='--', label='Stimulus onset')
        ax1.set_xlabel('Time from stimulus onset (s)')
        ax1.set_ylabel('Firing rate (spikes/s)')
        ax1.set_title(f'Response to {stim_name.split("_")[0]}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot raster
        ax2 = axes[i, 1]
        for j, spikes in enumerate(raster_data):
            ax2.plot(spikes, np.ones_like(spikes) * j, '|', color='black', markersize=4)
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_xlabel('Time from stimulus onset (s)')
        ax2.set_ylabel('Trial')
        ax2.set_title('Spike raster')
        ax2.set_ylim(-1, len(raster_data))
        
    plt.tight_layout()
    plt.savefig(f'explore/unit_{unit_id}_responses.png')
    plt.close()
    print(f"  Saved response plots to explore/unit_{unit_id}_responses.png")

print("\nAnalysis complete!")