# Explore spike data by creating a raster plot
# This script loads spike times for a few selected units
# from the NWB file and creates a raster plot.
# It saves the plot to explore/raster_plot.png.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, 'r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Access units table
units_df = nwb.units.to_dataframe()

# Select a few units to plot (e.g., first 5 units)
num_units_to_plot = 5
selected_unit_ids = units_df.index[:num_units_to_plot]

spike_times_list = []
for unit_id in selected_unit_ids:
    # Get spike times for the current unit
    # The spike_times are stored as an index into nwb.units['spike_times'].data
    # For a VectorIndex, we access the data of the target VectorData, then slice by the index.
    unit_spike_indices = nwb.units['spike_times'].get(unit_id).data[:]
    spike_times_list.append(unit_spike_indices)


# Create raster plot
sns.set_theme()
fig, ax = plt.subplots(figsize=(12, 6))
event_colors = sns.color_palette("husl", num_units_to_plot)

for i, unit_id in enumerate(selected_unit_ids):
    times = spike_times_list[i]
    # Plot only a subset of spikes if there are too many, e.g., first 1000 spikes per unit
    # and only within a certain time range for better visualization.
    times_subset = times[(times >= 0) & (times <= 300)][:1000] # Spikes up to 300s, max 1000 spikes
    ax.eventplot(times_subset, lineoffsets=i + 1, colors=[event_colors[i]], linelengths=0.8)

ax.set_yticks(np.arange(num_units_to_plot) + 1)
ax.set_yticklabels([f'Unit {uid}' for uid in selected_unit_ids])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Unit ID')
ax.set_title(f'Raster Plot (First {num_units_to_plot} units, 0-300s, max 1000 spikes/unit)')
plt.tight_layout()

# Save the plot
plt.savefig('explore/raster_plot.png')

print(f"Plot saved to explore/raster_plot.png")

# Close the HDF5 file and the NWBHDF5IO object
h5_file.close()
io.close()
print("NWB file processing complete.")