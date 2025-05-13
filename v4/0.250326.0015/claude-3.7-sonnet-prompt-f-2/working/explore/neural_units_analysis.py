# This script explores neural units data from the ecephys recording

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Set seaborn style
import seaborn as sns
sns.set_theme()

print("Loading main NWB file for units information...")
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Extract units data
units = nwb.units
unit_ids = units.id[:]
firing_rates = units['firing_rate'][:]
qualities = units['quality'][:]

# Examine only 'good' units if that information is available
if 'good' in np.unique(qualities):
    good_mask = qualities == 'good'
    good_unit_ids = unit_ids[good_mask]
    good_firing_rates = firing_rates[good_mask]
    print(f"Found {len(good_unit_ids)} good units out of {len(unit_ids)} total units.")
else:
    # If quality is not categorized as 'good', use all units
    good_unit_ids = unit_ids
    good_firing_rates = firing_rates
    print(f"No 'good' quality label found. Using all {len(unit_ids)} units.")

# Plot histogram of firing rates
plt.figure(figsize=(10, 6))
plt.hist(good_firing_rates, bins=50)
plt.title('Distribution of Firing Rates')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Units')
plt.savefig('explore/firing_rate_distribution.png')

# Get information about probes
probe_ids = np.unique(units['peak_channel_id'][:])
print(f"Found {len(probe_ids)} unique probe IDs.")

# Analysis of spike times for a sample unit
if len(good_unit_ids) > 0:
    # Select a unit with relatively high firing rate for analysis
    sorted_indices = np.argsort(good_firing_rates)
    high_fr_unit_idx = sorted_indices[-5]  # 5th highest firing rate
    selected_unit_id = good_unit_ids[high_fr_unit_idx]
    selected_unit_idx = np.where(unit_ids == selected_unit_id)[0][0]
    
    unit_spike_times = units.spike_times_index[selected_unit_idx]
    
    # If the spike times are available
    if len(unit_spike_times) > 0:
        print(f"Selected unit {selected_unit_id} has {len(unit_spike_times)} spikes.")
        
        # Get a sample of the spike times (first 1000 or all if fewer)
        sample_size = min(1000, len(unit_spike_times))
        sample_spike_times = unit_spike_times[:sample_size]
        
        # Plot spike raster
        plt.figure(figsize=(12, 4))
        plt.plot(sample_spike_times, np.ones_like(sample_spike_times), '|', markersize=10)
        plt.title(f'Spike Raster for Unit {selected_unit_id} (First {sample_size} Spikes)')
        plt.xlabel('Time (s)')
        plt.yticks([])
        plt.savefig('explore/spike_raster.png')
        
        # Compute and plot inter-spike intervals (ISIs)
        isis = np.diff(unit_spike_times)
        
        plt.figure(figsize=(10, 6))
        plt.hist(isis, bins=50, range=(0, 0.2))  # Focus on ISIs up to 200 ms
        plt.title(f'Inter-Spike Interval Distribution for Unit {selected_unit_id}')
        plt.xlabel('Inter-Spike Interval (s)')
        plt.ylabel('Count')
        plt.savefig('explore/isi_distribution.png')
    else:
        print("No spike times available for the selected unit.")

# Try to correlate neural activity with stimulus
# Get stimulus presentation information for one stimulus type
stim_key = 'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations'
if stim_key in nwb.intervals:
    stim_presentations = nwb.intervals[stim_key]
    start_times = stim_presentations.start_time[:]
    stop_times = stim_presentations.stop_time[:]
    
    # Select the first few stimulus presentations
    num_stim = min(5, len(start_times))
    stim_starts = start_times[:num_stim]
    stim_stops = stop_times[:num_stim]
    
    # Select a unit to analyze around stimulus presentation
    if len(good_unit_ids) > 0 and len(unit_spike_times) > 0:
        # For each stimulus presentation, count spikes
        for i in range(num_stim):
            # Get spikes during and around this stimulus
            window_start = max(0, stim_starts[i] - 1.0)  # 1 second before stimulus
            window_stop = stim_stops[i] + 1.0  # 1 second after stimulus
            
            spike_mask = (unit_spike_times >= window_start) & (unit_spike_times <= window_stop)
            window_spikes = unit_spike_times[spike_mask]
            
            # Create a histogram of spikes around stimulus
            plt.figure(figsize=(10, 4))
            
            # Mark stimulus period
            plt.axvspan(stim_starts[i], stim_stops[i], color='red', alpha=0.2, label='Stimulus')
            
            # Plot spike raster
            plt.plot(window_spikes, np.ones_like(window_spikes), '|', markersize=12, color='black')
            
            plt.title(f'Spikes Around Stimulus Presentation {i+1}')
            plt.xlabel('Time (s)')
            plt.yticks([])
            plt.legend()
            plt.xlim(window_start, window_stop)
            plt.savefig(f'explore/spikes_around_stimulus_{i+1}.png')
            
        print(f"Generated plots for {num_stim} stimulus presentations.")

print("Analysis complete. See output images in explore directory.")