"""
This script explores the spike data from the main session NWB file to understand:
1. The structure of spike data
2. The distribution of neurons across brain regions
3. Basic firing properties
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb
import pandas as pd

# Save figures with seaborn style
import seaborn as sns
sns.set_theme()

# Set the URL for the main session data
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"

# Load the data
print("Loading data from remote NWB file...")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("NWB file loaded. Exploring spike data...")

# Get units data
units_df = nwb.units.to_dataframe()
print(f"Number of units: {len(units_df)}")
print(f"Unit table columns: {units_df.columns.tolist()}")

# Display sample of units data
print("\nSample of units data:")
print(units_df.head())

# Get electrode data to map units to brain regions
electrodes_df = nwb.electrodes.to_dataframe()
print(f"\nNumber of electrodes: {len(electrodes_df)}")

# Get spike timestamps for a few example neurons
# Select a few units with different firing rates
units_by_firing_rate = units_df.sort_values('firing_rate', ascending=False)
sample_units = units_by_firing_rate.iloc[[0, 10, 50, 100, 250]].index

# Plot firing rates distribution
plt.figure(figsize=(10, 6))
sns.histplot(units_df['firing_rate'], bins=50, kde=True)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Count')
plt.title('Distribution of Neuron Firing Rates')
plt.savefig('explore/firing_rate_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot quality metrics distribution
quality_metrics = ['isolation_distance', 'l_ratio', 'd_prime', 'snr', 'isi_violations']
fig, axes = plt.subplots(len(quality_metrics), 1, figsize=(10, 15))
for i, metric in enumerate(quality_metrics):
    if metric in units_df.columns:
        # Filter out extreme outliers
        values = units_df[metric].dropna()
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
        
        sns.histplot(filtered_values, bins=30, kde=True, ax=axes[i])
        axes[i].set_xlabel(metric)
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'Distribution of {metric}')

plt.tight_layout()
plt.savefig('explore/unit_quality_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a raster plot for sample units
# Get spike times for sample units
sample_spike_times = {}
for unit_id in sample_units:
    spike_times = units_df.loc[unit_id, 'spike_times']
    # Only keep spike times in a reasonable range to visualize
    # For example, 10 seconds of data
    sample_spike_times[unit_id] = spike_times[(spike_times >= 100) & (spike_times < 110)]

# Plot the raster
plt.figure(figsize=(15, 8))
for i, (unit_id, spike_times) in enumerate(sample_spike_times.items()):
    plt.scatter(
        spike_times, 
        np.ones_like(spike_times) * i, 
        s=10, 
        label=f"Unit {unit_id} (FR: {units_df.loc[unit_id, 'firing_rate']:.2f} Hz)"
    )

plt.xlabel('Time (s)')
plt.yticks(range(len(sample_units)), [f"Unit {id}" for id in sample_units])
plt.ylabel('Unit')
plt.title('Spike Raster Plot for Sample Units (10 sec window)')
plt.grid(True, axis='x')
plt.savefig('explore/spike_raster.png', dpi=300, bbox_inches='tight')
plt.close()

# Get waveforms for sample units
fig, axes = plt.subplots(len(sample_units), 1, figsize=(10, 12))
for i, unit_id in enumerate(sample_units):
    if 'waveform_mean' in units_df.columns:
        waveform = units_df.loc[unit_id, 'waveform_mean']
        if waveform is not None and len(waveform) > 0:
            axes[i].plot(waveform)
            axes[i].set_title(f"Unit {unit_id} Waveform")
            axes[i].set_xlabel('Samples')
            axes[i].set_ylabel('Amplitude (V)')
        else:
            axes[i].text(0.5, 0.5, 'No waveform data available', 
                         horizontalalignment='center',
                         verticalalignment='center',
                         transform=axes[i].transAxes)
    else:
        axes[i].text(0.5, 0.5, 'No waveform data available', 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=axes[i].transAxes)

plt.tight_layout()
plt.savefig('explore/unit_waveforms.png', dpi=300, bbox_inches='tight')
plt.close()

print("Script completed successfully!")