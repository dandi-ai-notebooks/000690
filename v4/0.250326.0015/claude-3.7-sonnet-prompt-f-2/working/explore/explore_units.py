# This script explores the spiking activity (units) from the dataset
# We want to analyze spike properties and how they relate to visual stimuli

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory for plots
os.makedirs('explore', exist_ok=True)

# Load the main NWB file which contains the units data
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the file
print(f"Session ID: {nwb.session_id}")
print(f"Subject: {nwb.subject.subject_id}, {nwb.subject.species}")

# Get information about units (sorted spikes)
units = nwb.units
print(f"\nFound {len(units.id)} sorted units")
print(f"Available unit metadata: {units.colnames}")

# Extract key unit properties
unit_ids = units.id[:]
firing_rates = units.firing_rate[:] if 'firing_rate' in units.colnames else None
qualities = units.quality[:] if 'quality' in units.colnames else None
waveform_durations = units.waveform_duration[:] if 'waveform_duration' in units.colnames else None

# Get electrode information
electrodes = nwb.electrodes

# Try to get brain regions for units
brain_regions = []
try:
    if 'peak_channel_id' in units.colnames:
        peak_channels = units.peak_channel_id[:]
        for channel_id in peak_channels:
            # Find matching electrode
            if channel_id < len(electrodes.location):
                brain_regions.append(electrodes.location[channel_id])
            else:
                brain_regions.append('unknown')
    else:
        brain_regions = ['unknown'] * len(unit_ids)
except Exception as e:
    print(f"Could not get brain regions: {e}")
    brain_regions = ['unknown'] * len(unit_ids)

# Count units by brain region
unique_regions = np.unique(brain_regions)
region_counts = {}
for region in unique_regions:
    region_counts[region] = np.sum(np.array(brain_regions) == region)

# Plot unit counts by brain region
plt.figure(figsize=(12, 8))
regions = list(region_counts.keys())
counts = list(region_counts.values())
# Sort by count
sorted_indices = np.argsort(counts)[::-1]  # Descending order
regions = [regions[i] for i in sorted_indices]
counts = [counts[i] for i in sorted_indices]

plt.bar(regions, counts)
plt.title('Number of Units by Brain Region')
plt.xlabel('Brain Region')
plt.ylabel('Number of Units')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('explore/units_by_region.png')
plt.close()

# Plot distribution of firing rates if available
if firing_rates is not None:
    plt.figure(figsize=(10, 6))
    plt.hist(firing_rates, bins=50)
    plt.title('Distribution of Firing Rates')
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('explore/firing_rate_distribution.png')
    plt.close()
    
    print(f"\nFiring rate statistics:")
    print(f"Mean firing rate: {np.mean(firing_rates):.2f} Hz")
    print(f"Median firing rate: {np.median(firing_rates):.2f} Hz")
    print(f"Min firing rate: {np.min(firing_rates):.2f} Hz")
    print(f"Max firing rate: {np.max(firing_rates):.2f} Hz")

# Plot distribution of waveform durations if available
if waveform_durations is not None:
    plt.figure(figsize=(10, 6))
    plt.hist(waveform_durations, bins=50)
    plt.title('Distribution of Waveform Durations')
    plt.xlabel('Waveform Duration (ms)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('explore/waveform_duration_distribution.png')
    plt.close()

# Function to create a raster plot of spikes for selected units
def plot_spike_raster(units, unit_indices, time_window, title):
    """
    Create a raster plot for selected units
    
    Parameters:
    -----------
    units : pynwb.misc.Units
        The units object containing spike times
    unit_indices : list
        Indices of units to plot
    time_window : tuple
        (start_time, end_time) to plot in seconds
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 8))
    
    for i, unit_idx in enumerate(unit_indices):
        # Get spike times for this unit
        spike_times = units.spike_times_index[unit_idx]
        
        # Filter to time window
        mask = (spike_times >= time_window[0]) & (spike_times <= time_window[1])
        visible_spikes = spike_times[mask]
        
        # Plot spikes as dots
        plt.plot(visible_spikes, np.ones_like(visible_spikes)*i, 'k|', markersize=4)
    
    plt.yticks(np.arange(len(unit_indices)), unit_indices)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Unit Index')
    plt.grid(True, axis='x')
    return plt.gcf()

# Select a subset of units to plot
n_units_to_plot = 20
# Try to select units from regions of interest (if available), otherwise use random selection
if 'unknown' not in unique_regions:
    # Find the region with the most units
    top_region = regions[0]
    units_from_region = np.where(np.array(brain_regions) == top_region)[0]
    
    if len(units_from_region) >= n_units_to_plot:
        unit_indices = units_from_region[:n_units_to_plot]
    else:
        # Need to add units from other regions
        remaining_units = n_units_to_plot - len(units_from_region)
        other_units = np.where(np.array(brain_regions) != top_region)[0][:remaining_units]
        unit_indices = np.concatenate([units_from_region, other_units])
else:
    # Random selection
    unit_indices = np.random.choice(len(unit_ids), min(n_units_to_plot, len(unit_ids)), replace=False)

# Get a time window that shows some interesting activity
time_start = 100  # Starting 100 seconds into the recording
time_span = 5    # 5-second window

# Create raster plot
fig = plot_spike_raster(units, unit_indices, (time_start, time_start + time_span), 
                        f'Spike Raster Plot ({time_span} seconds)')
plt.savefig('explore/spike_raster.png')
plt.close()

# Try to plot response of units to stimuli
# Get stimulus presentation times for one example stimulus
stim_intervals = [k for k in nwb.intervals.keys() if 'presentations' in k]
if stim_intervals:
    selected_stim = stim_intervals[0]
    for stim_name in stim_intervals:
        # Prefer a simple stimulus (like a bar) if possible
        if 'SAC' in stim_name or 'bar' in stim_name.lower():
            selected_stim = stim_name
            break
    
    print(f"\nAnalyzing spike responses to {selected_stim}")
    stim_data = nwb.intervals[selected_stim]
    stim_times = stim_data.start_time[:]
    
    # Use a subset of stimulus presentations
    n_stim_to_use = min(50, len(stim_times))
    stim_times_subset = stim_times[:n_stim_to_use]
    
    # Create a peri-stimulus time histogram (PSTH) for selected units
    def compute_psth(spike_times, event_times, pre_time=0.5, post_time=1.5, bin_size=0.025):
        """Compute Peri-Stimulus Time Histogram"""
        edges = np.arange(-pre_time, post_time + bin_size, bin_size)
        centers = edges[:-1] + bin_size/2
        hist = np.zeros((len(event_times), len(centers)))
        
        for i, event_time in enumerate(event_times):
            # Align spikes to this event
            aligned_times = spike_times - event_time
            # Include only spikes within our window
            mask = (aligned_times >= -pre_time) & (aligned_times < post_time)
            spikes_in_window = aligned_times[mask]
            
            # Count spikes in bins
            hist[i], _ = np.histogram(spikes_in_window, bins=edges)
        
        # Average across events and convert to firing rate
        avg_psth = np.mean(hist, axis=0) / bin_size
        return avg_psth, centers
    
    # Compute PSTH for selected units
    psth_units = unit_indices[:5]  # Just use the first 5 for simplicity
    plt.figure(figsize=(12, 10))
    
    for i, unit_idx in enumerate(psth_units):
        # Get spike times for this unit
        spike_times = units.spike_times_index[unit_idx]
        
        # Compute PSTH
        psth, time_bins = compute_psth(spike_times, stim_times_subset)
        
        # Plot PSTH
        ax = plt.subplot(len(psth_units), 1, i+1)
        ax.bar(time_bins, psth, width=time_bins[1]-time_bins[0], alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Stimulus onset')
        ax.set_ylabel('Firing Rate (Hz)')
        
        # Only show x-label on bottom plot
        if i == len(psth_units)-1:
            ax.set_xlabel('Time relative to stimulus onset (s)')
        
        ax.set_title(f'Unit {unit_idx}')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('explore/psth_example_units.png')
    plt.close()
    
    # Try to calculate average waveform shapes if available
    if 'waveform_mean' in units.colnames:
        plt.figure(figsize=(12, 8))
        for i, unit_idx in enumerate(psth_units):
            try:
                # Get waveform for this unit
                waveform = units.waveform_mean_index[unit_idx]
                if waveform is not None and len(waveform) > 0:
                    # Plot waveform
                    plt.subplot(len(psth_units), 1, i+1)
                    plt.plot(waveform)
                    plt.title(f'Unit {unit_idx} Waveform')
                    plt.ylabel('Amplitude')
                    if i == len(psth_units)-1:
                        plt.xlabel('Sample')
                    plt.grid(True)
            except Exception as e:
                print(f"Could not plot waveform for unit {unit_idx}: {e}")
        
        plt.tight_layout()
        plt.savefig('explore/unit_waveforms.png')
        plt.close()

print("Units exploration completed. Check explore directory for plots.")