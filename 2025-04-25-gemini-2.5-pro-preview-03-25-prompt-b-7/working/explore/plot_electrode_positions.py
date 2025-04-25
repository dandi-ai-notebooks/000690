# Explore electrode metadata: Plot the spatial positions of electrodes on the probe.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

# Get electrodes table as DataFrame
electrodes_df = nwb.electrodes.to_dataframe()

# Plot electrode positions
plt.figure(figsize=(6, 10))
sns.scatterplot(data=electrodes_df, x='probe_horizontal_position', y='probe_vertical_position', hue='location', s=50) # Color by location
plt.title(f'Electrode Positions on Probe {nwb.electrode_groups["probeA"].probe_id}')
plt.xlabel('Horizontal Position (microns)')
plt.ylabel('Vertical Position (microns)')
plt.legend(title='Brain Region', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

# Save plot
plt.savefig("explore/electrode_positions.png")

print("Saved plot to explore/electrode_positions.png")

# Close resources
io.close()
remote_file.close()