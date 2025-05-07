"""
This script analyzes neural responses to specific stimuli by extracting spike times 
around stimulus presentation events, without loading full stimulus data.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Load the main NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from URL: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get basic unit (neuron) information
print("\nExtracting basic unit information...")
# Get total number of units
num_units = len(nwb.units.id[:])
print(f"Total number of units: {num_units}")

# Function to find stimulus presentations in a time window
def get_stimulus_presentations(stim_name, start_time=0, end_time=60):
    """Get stimulus presentations within a given time window."""
    if stim_name not in nwb.intervals:
        print(f"Stimulus {stim_name} not found in intervals.")
        return []
    
    interval = nwb.intervals[stim_name]
    # Extract only the presentations in the time window to avoid memory issues
    start_times = interval.start_time[:]
    stop_times = interval.stop_time[:]
    
    presentations = []
    for i, (start, stop) in enumerate(zip(start_times, stop_times)):
        if start >= start_time and start <= end_time:
            presentations.append({
                'index': i,
                'start_time': start,
                'stop_time': stop,
                'duration': stop - start
            })
            
            # Only collect a few for analysis to avoid memory issues
            if len(presentations) >= 10:
                break
                
    print(f"Found {len(presentations)} presentations of {stim_name} in time window.")
    return presentations

# Function to get spikes for specific units in a time window
def get_unit_spikes(unit_ids, start_time, end_time):
    """Get spike times for specific units within a given time window."""
    unit_spikes = {}
    
    for unit_id in unit_ids:
        # Access spike times for this unit
        spike_times = nwb.units['spike_times'][unit_id]
        
        # Filter to the time window
        in_window = (spike_times >= start_time) & (spike_times <= end_time)
        window_spikes = spike_times[in_window]
        
        unit_spikes[unit_id] = window_spikes
    
    return unit_spikes

# Function to create a raster plot
def create_raster_plot(unit_spikes, stimulus_times, window_before=1.0, window_after=2.0, title="Neural Response"):
    """Create a raster plot for neural activity aligned to stimulus onset times."""
    plt.figure(figsize=(10, 8))
    
    # Track trial and unit position for the plot
    trial_positions = []
    unit_positions = []
    spike_times_aligned = []
    
    # For each stimulus presentation
    for trial_idx, stim_time in enumerate(stimulus_times):
        stim_start = stim_time['start_time']
        
        # For each unit
        for unit_idx, (unit_id, spikes) in enumerate(unit_spikes.items()):
            # Find spikes in the window around stimulus onset
            window_mask = (spikes >= (stim_start - window_before)) & (spikes <= (stim_start + window_after))
            stim_spikes = spikes[window_mask]
            
            # Align spike times to stimulus onset
            aligned_spikes = stim_spikes - stim_start
            
            # Store information for plotting
            trial_positions.extend([trial_idx] * len(aligned_spikes))
            unit_positions.extend([unit_idx] * len(aligned_spikes))
            spike_times_aligned.extend(aligned_spikes)
    
    # Create raster plot
    plt.scatter(spike_times_aligned, trial_positions, s=2, color='k', marker='|')
    
    # Add stimulus onset line
    plt.axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')
    
    # Add labels
    plt.xlabel('Time (s) relative to stimulus onset')
    plt.ylabel('Presentation #')
    plt.title(title)
    plt.xlim([-window_before, window_after])
    plt.ylim([-1, len(stimulus_times)])
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.savefig(f'explore/{title.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()

# Function to create PSTH (Peri-Stimulus Time Histogram)
def create_psth(unit_spikes, stimulus_times, window_before=1.0, window_after=2.0, bin_size=0.05, title="PSTH"):
    """Create a PSTH for neural activity aligned to stimulus onset times."""
    plt.figure(figsize=(10, 6))
    
    # Create time bins
    bins = np.arange(-window_before, window_after + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    
    # Initialize counts
    all_counts = np.zeros((len(unit_spikes), len(bins) - 1))
    
    # For each stimulus presentation
    for stim_time in stimulus_times:
        stim_start = stim_time['start_time']
        
        # For each unit
        for unit_idx, (unit_id, spikes) in enumerate(unit_spikes.items()):
            # Find spikes in the window around stimulus onset
            window_mask = (spikes >= (stim_start - window_before)) & (spikes <= (stim_start + window_after))
            stim_spikes = spikes[window_mask]
            
            # Align spike times to stimulus onset
            aligned_spikes = stim_spikes - stim_start
            
            # Count spikes in each bin
            counts, _ = np.histogram(aligned_spikes, bins=bins)
            all_counts[unit_idx] += counts
    
    # Calculate average firing rate across all units
    avg_counts = np.mean(all_counts, axis=0)
    
    # Convert to firing rate (spikes/sec)
    firing_rate = avg_counts / (bin_size * len(stimulus_times))
    
    # Plot PSTH
    plt.bar(bin_centers, firing_rate, width=bin_size, alpha=0.7, color='b')
    
    # Add stimulus onset line
    plt.axvline(x=0, color='r', linestyle='--', label='Stimulus Onset')
    
    # Add labels
    plt.xlabel('Time (s) relative to stimulus onset')
    plt.ylabel('Firing Rate (spikes/sec)')
    plt.title(title)
    plt.xlim([-window_before, window_after])
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    plt.savefig(f'explore/{title.replace(" ", "_").lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()

# First, try to get information about a stimulus type
print("\nAnalyzing neural responses to visual stimuli...")
# Try with a simple bar stimulus
stim_name = 'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations'

# Get stimulus presentations in a short time window to limit data
presentations = get_stimulus_presentations(stim_name, start_time=100, end_time=200)

if presentations:
    print("\nFound stimulus presentations, analyzing neural responses...")
    # Get a small sample of units
    num_sample_units = 20
    unit_ids = list(range(min(num_sample_units, num_units)))
    
    print(f"Analyzing responses for {len(unit_ids)} units...")
    
    # Get spikes around all presentations
    start_time = presentations[0]['start_time'] - 5  # 5 seconds before first presentation
    end_time = presentations[-1]['stop_time'] + 5    # 5 seconds after last presentation
    
    # Get spikes for each unit in this time window
    unit_spikes = get_unit_spikes(unit_ids, start_time, end_time)
    
    # Create visualization
    print("Creating raster plot...")
    create_raster_plot(unit_spikes, presentations, window_before=0.5, window_after=1.0, 
                      title=f"Neural Response to {stim_name[:10]}")
    
    print("Creating PSTH plot...")
    create_psth(unit_spikes, presentations, window_before=0.5, window_after=1.0, bin_size=0.05,
               title=f"PSTH for {stim_name[:10]}")
    
else:
    print("No presentations found. Trying a different stimulus...")
    # Try with a natural movie stimulus
    stim_name = 'natmovie_EagleSwooping1_540x960Full_584x460Active_presentations'
    presentations = get_stimulus_presentations(stim_name, start_time=500, end_time=600)
    
    if presentations:
        # Get a small sample of units
        num_sample_units = 20
        unit_ids = list(range(min(num_sample_units, num_units)))
        
        print(f"Analyzing responses for {len(unit_ids)} units...")
        
        # Get spikes around all presentations
        start_time = presentations[0]['start_time'] - 5
        end_time = presentations[-1]['stop_time'] + 5
        
        # Get spikes for each unit in this time window
        unit_spikes = get_unit_spikes(unit_ids, start_time, end_time)
        
        # Create visualization
        print("Creating raster plot...")
        create_raster_plot(unit_spikes, presentations, window_before=0.5, window_after=1.0, 
                          title=f"Neural Response to {stim_name[:10]}")
        
        print("Creating PSTH plot...")
        create_psth(unit_spikes, presentations, window_before=0.5, window_after=1.0, bin_size=0.05,
                   title=f"PSTH for {stim_name[:10]}")

print("\nNeural response analysis complete!")
io.close()