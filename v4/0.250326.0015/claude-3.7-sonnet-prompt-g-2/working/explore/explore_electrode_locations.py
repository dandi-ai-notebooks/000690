"""
Explore electrode locations and probe information for the Dandiset.
This script visualizes the electrode locations in 3D space to understand the probe configuration.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the main NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode table as dataframe
electrodes_df = nwb.electrodes.to_dataframe()

# Print basic information
print(f"Total number of electrodes: {len(electrodes_df)}")
print(f"Available probe groups: {electrodes_df['group_name'].unique()}")
print(f"Electrode locations: {electrodes_df['location'].unique()}")

# Create a 3D plot of electrode positions
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Dictionary to map group names to colors
colors = {'probeA': 'red', 'probeB': 'blue', 'probeE': 'green', 'probeF': 'purple'}

for group_name in electrodes_df['group_name'].unique():
    group_df = electrodes_df[electrodes_df['group_name'] == group_name]
    ax.scatter(
        group_df['x'], 
        group_df['y'], 
        group_df['z'], 
        c=colors.get(group_name, 'black'),
        label=group_name,
        alpha=0.7
    )

ax.set_xlabel('X (posterior +)')
ax.set_ylabel('Y (inferior +)')
ax.set_zlabel('Z (right +)')
ax.legend()
ax.set_title('3D Electrode Locations')

# Save the figure
plt.savefig('explore/electrode_locations_3d.png')

# Plot electrodes by probe in 2D (X-Y and X-Z projections)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

for group_name in electrodes_df['group_name'].unique():
    group_df = electrodes_df[electrodes_df['group_name'] == group_name]
    
    # X-Y projection (top view)
    axes[0].scatter(
        group_df['x'], 
        group_df['y'],
        c=colors.get(group_name, 'black'),
        label=group_name,
        alpha=0.7
    )
    
    # X-Z projection (side view)
    axes[1].scatter(
        group_df['x'], 
        group_df['z'],
        c=colors.get(group_name, 'black'),
        label=group_name,
        alpha=0.7
    )

axes[0].set_xlabel('X (posterior +)')
axes[0].set_ylabel('Y (inferior +)')
axes[0].set_title('X-Y Projection of Electrode Locations (Top View)')
axes[0].legend()

axes[1].set_xlabel('X (posterior +)')
axes[1].set_ylabel('Z (right +)')
axes[1].set_title('X-Z Projection of Electrode Locations (Side View)')
axes[1].legend()

plt.tight_layout()
plt.savefig('explore/electrode_locations_2d.png')

# Print electrode count by brain region
print("\nElectrode count by location:")
location_counts = electrodes_df['location'].value_counts()
for location, count in location_counts.items():
    print(f"{location}: {count} electrodes")

# Print information about each probe
print("\nProbe information:")
for probe_name in ['probeA', 'probeB', 'probeE', 'probeF']:
    probe = nwb.electrode_groups[probe_name]
    print(f"{probe_name}:")
    print(f"  Description: {probe.description}")
    print(f"  Location: {probe.location}")
    print(f"  Has LFP data: {probe.has_lfp_data}")
    print(f"  LFP sampling rate: {probe.lfp_sampling_rate} Hz")
    print(f"  Probe device sampling rate: {probe.device.sampling_rate} Hz")
    print(f"  Manufacturer: {probe.device.manufacturer}")
    print()

print("Done! Electrode location plots saved to explore/electrode_locations_3d.png and explore/electrode_locations_2d.png")