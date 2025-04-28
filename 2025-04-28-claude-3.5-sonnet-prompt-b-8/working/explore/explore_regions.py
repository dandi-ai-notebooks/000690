import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

print("Loading NWB file...")

# Load the file
url = "https://api.dandiarchive.org/api/assets/ecaed1ec-a8b5-4fe7-87c1-baf68cfa900f/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the LFP data and electrodes info
probe_0_lfp_data = nwb.acquisition["probe_0_lfp_data"]
electrodes_df = nwb.electrodes.to_dataframe()

# Print electrode count and data shape
print(f"\nTotal electrodes in dataframe: {len(electrodes_df)}")
print(f"LFP data shape: {probe_0_lfp_data.data.shape}")

# Group electrode indices by region, ensuring indices are within bounds
region_electrodes = defaultdict(list)
for idx, row in electrodes_df.iterrows():
    if row['location'] != 'none' and idx < probe_0_lfp_data.data.shape[1]:
        region_electrodes[row['location']].append(idx)

# Load a 2-second segment of data
start_idx = 0
n_samples = 1250  # 2 seconds at 625 Hz
data = probe_0_lfp_data.data[start_idx:start_idx+n_samples, :]
timestamps = probe_0_lfp_data.timestamps[start_idx:start_idx+n_samples]

# Create plot showing average LFP by region
plt.figure(figsize=(15, 12))

# Sort regions to group hippocampal and visual areas
regions_order = ['CA1', 'CA3', 'DG', 'VISl1', 'VISl2/3', 'VISl4', 'VISl5', 'VISl6']
for i, region in enumerate(regions_order):
    if region in region_electrodes and len(region_electrodes[region]) > 0:
        # Calculate mean LFP for the region
        region_data = data[:, region_electrodes[region]]
        mean_lfp = np.mean(region_data, axis=1)
        
        # Plot with offset for visibility
        offset = i * 0.0004
        plt.plot(timestamps, mean_lfp + offset, label=f'{region} (n={len(region_electrodes[region])})')

plt.xlabel('Time (seconds)')
plt.ylabel('LFP (V)')
plt.title('Average LFP by Brain Region')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('explore/region_lfp_comparison.png')
plt.close()

# Print some statistics about the regions
print("\nNumber of electrodes by region:")
for region in regions_order:
    if region in region_electrodes:
        print(f"{region}: {len(region_electrodes[region])} electrodes")

# Calculate power in different frequency bands by region
from scipy import signal

def calculate_band_power(data, fs, band):
    freqs, psd = signal.welch(data, fs, nperseg=fs)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.mean(psd[idx])

bands = {
    'Delta': (1, 4),
    'Theta': (4, 12),
    'Beta': (12, 30),
    'Gamma': (30, 100)
}

print("\nRelative band power by region:")
for region in regions_order:
    if region in region_electrodes and len(region_electrodes[region]) > 0:
        print(f"\n{region}:")
        region_data = data[:, region_electrodes[region]]
        mean_lfp = np.mean(region_data, axis=1)
        
        # Calculate total power across all bands
        total_power = sum(calculate_band_power(mean_lfp, 625, band) for band in bands.values())
        
        # Calculate relative power in each band
        for band_name, band_range in bands.items():
            power = calculate_band_power(mean_lfp, 625, band_range)
            rel_power = power / total_power * 100
            print(f"{band_name}: {rel_power:.1f}%")

print("\nAnalysis complete!")