"""
This script explores the basic structure of the NWB file to understand the available data types,
focusing on the units (neurons) and their properties.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Set the plot style but not using seaborn as it's deprecated
plt.style.use('default')

# Load the main NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from URL: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic information about the file
print("\nBasic NWB file information:")
print(f"Session ID: {nwb.session_id}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Session description: {nwb.session_description}")
print(f"Institution: {nwb.institution}")
print(f"Stimulus notes: {nwb.stimulus_notes}")

# Look at the electrode groups (probes)
print("\nElectrode groups (probes):")
for group_name, group in nwb.electrode_groups.items():
    print(f"  {group_name}: {group.description} at {group.location}")
    print(f"    Manufacturer: {group.device.manufacturer}")
    print(f"    Sampling rate: {group.device.sampling_rate} Hz")
    if hasattr(group, 'lfp_sampling_rate'):
        print(f"    LFP sampling rate: {group.lfp_sampling_rate} Hz")

# Get units information
print("\nUnits (neurons) information:")
print(f"Number of units: {len(nwb.units.id[:])}")

# Convert to dataframe for easier analysis
units_df = nwb.units.to_dataframe()
print("\nUnits dataframe columns:")
print(units_df.columns)

# Get basic statistics on unit quality and firing rate
print("\nUnits quality distribution:")
if 'quality' in units_df.columns:
    print(units_df['quality'].value_counts())

print("\nFiring rate statistics:")
if 'firing_rate' in units_df.columns:
    print(f"Mean: {units_df['firing_rate'].mean():.2f} Hz")
    print(f"Median: {units_df['firing_rate'].median():.2f} Hz")
    print(f"Min: {units_df['firing_rate'].min():.2f} Hz")
    print(f"Max: {units_df['firing_rate'].max():.2f} Hz")

# Create a histogram of firing rates and save to a file
plt.figure(figsize=(10, 6))
if 'firing_rate' in units_df.columns:
    plt.hist(units_df['firing_rate'], bins=50, alpha=0.7)
    plt.xlabel('Firing Rate (Hz)')
    plt.ylabel('Count')
    plt.title('Distribution of Neuron Firing Rates')
    plt.grid(True, alpha=0.3)
    plt.savefig('explore/firing_rate_histogram.png', dpi=300, bbox_inches='tight')
    print("Saved firing rate histogram to 'explore/firing_rate_histogram.png'")

# Look at the available stimulus presentations
print("\nStimulus presentations:")
for interval_name, interval in nwb.intervals.items():
    if 'presentations' in interval_name:
        print(f"  {interval_name}: {len(interval.id[:])} presentations")

# Get information about the processing modules
print("\nProcessing modules:")
for module_name, module in nwb.processing.items():
    print(f"  {module_name}: {module.description}")
    print(f"    Data interfaces: {list(module.data_interfaces.keys())}")

# Close the file
io.close()
print("\nExploration complete!")