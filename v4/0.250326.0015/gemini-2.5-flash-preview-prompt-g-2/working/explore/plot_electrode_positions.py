# Script to visualize electrode positions from NWB file

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode data as a DataFrame
electrodes_df = nwb.electrodes.to_dataframe()

# Plot electrode positions
sns.set_theme()
plt.figure(figsize=(8, 6))
sns.scatterplot(data=electrodes_df, x='probe_horizontal_position', y='probe_vertical_position')
plt.xlabel('Horizontal Position (microns)')
plt.ylabel('Vertical Position (microns)')
plt.title('Electrode Positions on Probe')
plt.gca().invert_yaxis() # Invert y-axis so 0 is at the top
plt.savefig('explore/electrode_positions.png')
# plt.show() # Do not show the plot to avoid hanging

io.close() # Close the NWB file