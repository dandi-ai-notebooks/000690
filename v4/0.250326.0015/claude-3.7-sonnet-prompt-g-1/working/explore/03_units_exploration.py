# This script explores the Units data from the session NWB file
# to understand neural responses to visual stimuli

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Load the main session NWB file
url = "https://api.dandiarchive.org/api/assets/9b14e3b4-5d3e-4121-ae5e-ced7bc92af4e/download/"
print(f"Loading main session NWB file from {url}")
print("This might take a moment as we're accessing a remote file...")

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Basic info
print(f"\nMain session file loaded.")
print(f"Session ID: {nwb.session_id}")

# Get units data
print("\nAccessing units data...")
units = nwb.units
units_df = units.to_dataframe()

# Basic statistics
print(f"Number of units (neurons): {len(units_df)}")

# Look at unit properties
print("\nUnit property statistics:")
spike_times = units.spike_times[:]
firing_rates = units_df['firing_rate'].dropna()

print(f"Mean firing rate: {firing_rates.mean():.2f} Hz")
print(f"Median firing rate: {firing_rates.median():.2f} Hz")
print(f"Min firing rate: {firing_rates.min():.2f} Hz")
print(f"Max firing rate: {firing_rates.max():.2f} Hz")

# Create a histogram of firing rates
plt.figure(figsize=(12, 6))
plt.hist(firing_rates, bins=50)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Units')
plt.title('Distribution of Neuron Firing Rates')
plt.savefig('explore/firing_rate_histogram.png', dpi=300)
print("Firing rate histogram saved to explore/firing_rate_histogram.png")

# Check quality metrics of units
if 'quality' in units_df.columns:
    quality_counts = units_df['quality'].value_counts()
    print("\nUnit quality distribution:")
    for quality, count in quality_counts.items():
        print(f"{quality}: {count} units ({100*count/len(units_df):.1f}%)")

# Plot the waveform for a few units if available
if 'waveform_mean' in units_df.columns:
    print("\nExtracting waveforms for sample units...")
    
    # Select a few units with good firing rates
    sample_units = units_df.sort_values('firing_rate', ascending=False).head(5).index
    
    plt.figure(figsize=(12, 8))
    for i, unit_id in enumerate(sample_units):
        waveform = units.waveform_mean[unit_id]
        if waveform is not None and len(waveform) > 0:
            plt.subplot(len(sample_units), 1, i+1)
            plt.plot(waveform)
            plt.title(f"Unit {unit_id} - Firing Rate: {units_df.loc[unit_id, 'firing_rate']:.2f} Hz")
            plt.ylabel('Amplitude')
            
    plt.tight_layout()
    plt.savefig('explore/sample_waveforms.png', dpi=300)
    print("Sample waveforms saved to explore/sample_waveforms.png")

# Let's explore the distribution of spikes over time for a few units
print("\nAnalyzing spike timing for high-firing units...")

# Select top firing units
top_units = units_df.sort_values('firing_rate', ascending=False).head(3).index

plt.figure(figsize=(15, 10))
for i, unit_id in enumerate(top_units):
    spike_times_unit = units.spike_times[unit_id]
    
    # Create a spike raster plot
    plt.subplot(len(top_units), 1, i+1)
    plt.plot(spike_times_unit, np.ones_like(spike_times_unit), '|', markersize=2)
    plt.title(f"Unit {unit_id} - Firing Rate: {units_df.loc[unit_id, 'firing_rate']:.2f} Hz")
    plt.ylabel('Spikes')
    
    # Only show x-label for bottom plot
    if i == len(top_units) - 1:
        plt.xlabel('Time (seconds)')
    
plt.tight_layout()
plt.savefig('explore/spike_rasters.png', dpi=300)
print("Spike rasters saved to explore/spike_rasters.png")

# Create a summary of waveform properties
if 'waveform_duration' in units_df.columns and 'waveform_halfwidth' in units_df.columns:
    plt.figure(figsize=(10, 8))
    plt.scatter(units_df['waveform_duration'], units_df['waveform_halfwidth'], 
               alpha=0.5, s=20)
    plt.xlabel('Waveform Duration (ms)')
    plt.ylabel('Waveform Half-width (ms)')
    plt.title('Waveform Properties')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('explore/waveform_properties.png', dpi=300)
    print("Waveform properties plot saved to explore/waveform_properties.png")

print("\nExploration complete!")