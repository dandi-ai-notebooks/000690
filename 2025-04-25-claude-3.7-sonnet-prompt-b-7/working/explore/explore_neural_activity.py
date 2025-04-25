"""
This script explores the neural activity data from the ecephys probe recordings.
We'll examine spiking patterns of individual units (neurons).
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
print("Connecting to DANDI archive...")
client = DandiAPIClient()
dandiset = client.get_dandiset("000690")

# Get a specific NWB file for a probe
print("Finding a probe ecephys file...")
subject = "695763"  # The subject we looked at earlier
assets = list(dandiset.get_assets())
probe_nwb_assets = [asset for asset in assets 
                    if f"sub-{subject}/" in asset.path and 
                    "_probe-" in asset.path and 
                    asset.path.endswith('_ecephys.nwb')]

# Select the first probe file
probe_nwb = probe_nwb_assets[0]
print(f"Selected probe file: {probe_nwb.path}")

# Get the URL for the asset
asset_url = f"https://api.dandiarchive.org/api/assets/{probe_nwb.identifier}/download/"

# Load the main session file to get stimulus timing information
print("Finding main session file...")
main_nwb_asset = [asset for asset in assets 
                 if f"sub-{subject}/" in asset.path and
                 not "_probe-" in asset.path and
                 not "_image.nwb" in asset.path and
                 asset.path.endswith('.nwb')][0]
main_url = f"https://api.dandiarchive.org/api/assets/{main_nwb_asset.identifier}/download/"

# Load main NWB file
print("Loading main NWB file...")
remote_main = remfile.File(main_url)
h5_file_main = h5py.File(remote_main)
io_main = pynwb.NWBHDF5IO(file=h5_file_main)
nwb_main = io_main.read()

# Get stimulus presentation intervals
print("Getting stimulus timing information...")
stimulus_intervals = {}
for interval_name, interval_obj in nwb_main.intervals.items():
    if interval_name.endswith('_presentations'):
        # Get first and last presentation times for this type
        df = interval_obj.to_dataframe()
        if len(df) > 0:
            stimulus_intervals[interval_name] = {
                'start': df['start_time'].min(),
                'end': df['stop_time'].max(),
                'count': len(df)
            }

# Print stimulus intervals
print(f"\nFound {len(stimulus_intervals)} stimulus types with timing information")
for stim_name, timing in list(stimulus_intervals.items())[:5]:  # Show just first 5
    print(f"- {stim_name}: {timing['count']} presentations, {timing['start']:.1f}s - {timing['end']:.1f}s")

# Load the probe file
print("\nLoading probe NWB file (this might take a moment)...")
remote_file = remfile.File(asset_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# --- Neural Activity Data ---
print("\n--- Neural Units Data ---")
if hasattr(nwb, 'units') and len(nwb.units) > 0:
    # Create a dataframe of units
    units_df = nwb.units.to_dataframe()
    print(f"Total number of units: {len(units_df)}")
    
    # Print basic info about quality metrics
    print("\nUnit quality metrics:")
    for metric in ['quality', 'snr', 'isi_violations', 'amplitude', 'firing_rate']:
        if metric in units_df.columns:
            print(f"- {metric}: mean={units_df[metric].mean():.2f}, min={units_df[metric].min():.2f}, max={units_df[metric].max():.2f}")
    
    # Get spike times for good quality units (if quality column exists)
    if 'quality' in units_df.columns:
        good_units = units_df[units_df['quality'] == 'good']
        print(f"\nGood quality units: {len(good_units)}")
    else:
        # If no quality column, just take first few units
        good_units = units_df.head(10)
        print("\nNo quality metric found. Using first 10 units for analysis.")
    
    # Plot firing rate histograms for a few units
    plt.figure(figsize=(12, 8))
    for i, (idx, unit) in enumerate(good_units.iloc[:5].iterrows()):
        # Get spike times
        spike_times = nwb.units['spike_times'][idx]
        
        # Create histogram of spike times
        plt.subplot(5, 1, i+1)
        plt.hist(spike_times, bins=100, density=True)
        plt.title(f"Unit {idx} - Firing Rate: {unit.get('firing_rate', 'unknown')}")
        plt.ylabel('Density')
        if i == 4:  # only add x-label to bottom plot
            plt.xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('explore/unit_spike_histograms.png')
    
    # Examine distribution of firing rates
    if 'firing_rate' in units_df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(units_df['firing_rate'], bins=50)
        plt.xlabel('Firing Rate (Hz)')
        plt.ylabel('Number of Units')
        plt.title('Distribution of Unit Firing Rates')
        plt.savefig('explore/firing_rate_distribution.png')
        
        # Print firing rate stats
        print(f"\nFiring rate statistics:")
        print(f"Mean firing rate: {units_df['firing_rate'].mean():.2f} Hz")
        print(f"Median firing rate: {units_df['firing_rate'].median():.2f} Hz")
        print(f"Min firing rate: {units_df['firing_rate'].min():.2f} Hz")
        print(f"Max firing rate: {units_df['firing_rate'].max():.2f} Hz")
        
        # Print distribution by range
        print("\nFiring rate distribution:")
        ranges = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 50), (50, float('inf'))]
        for low, high in ranges:
            count = len(units_df[(units_df['firing_rate'] >= low) & (units_df['firing_rate'] < high)])
            print(f"- {low}-{high if high != float('inf') else '+'} Hz: {count} units ({count/len(units_df)*100:.1f}%)")
else:
    print("No units data found in this file")

# Close the files
h5_file.close()
io.close()
h5_file_main.close()
io_main.close()
print("\nExploration complete!")