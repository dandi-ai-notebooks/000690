# explore/plot_spike_raster.py
# Goal: Visualize spike times for a subset of units as a raster plot.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use seaborn styling
sns.set_theme()

print("Loading NWB file remotely...")
# Define the URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"

# Use remfile to access the remote file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, mode='r')

# Use pynwb to read the NWB file structure
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()
print("NWB file loaded.")

# Access the units table
units_table = nwb.units

# Select a subset of units (e.g., first 10)
num_units_to_plot = 10
if len(units_table.id[:]) >= num_units_to_plot:
    unit_indices_to_plot = range(num_units_to_plot)
    unit_ids_to_plot = units_table.id[unit_indices_to_plot]
    print(f"Selected first {num_units_to_plot} units with IDs: {list(unit_ids_to_plot)}")
else:
    unit_indices_to_plot = range(len(units_table.id[:]))
    unit_ids_to_plot = units_table.id[:]
    print(f"Selected all {len(unit_ids_to_plot)} units.")


# Load spike times for the selected units
spike_times_list = []
print("Loading spike times for selected units...")
for i in unit_indices_to_plot:
    # access spike times using the row index i
    # spike_times_index gives the start/end indices into the main spike_times data vector
    # Access the scalar value within the potentially array-like index structure
    start_index_val = units_table['spike_times_index'][i-1][0] if i > 0 else 0
    end_index_val = units_table['spike_times_index'][i][0]
    # Ensure indices are standard Python integers for h5py slicing
    start_index = int(start_index_val)
    end_index = int(end_index_val)
    # slice the spike_times dataset
    spike_times = units_table['spike_times'].data[start_index:end_index]
    spike_times_list.append(spike_times)
print("Spike times loaded.")

# Create the raster plot
print("Generating raster plot...")
fig, ax = plt.subplots(figsize=(12, 6))
# Use eventplot to create the raster plot
ax.eventplot(spike_times_list, color='black', linelengths=0.75)

ax.set_yticks(range(len(unit_ids_to_plot)))
ax.set_yticklabels(unit_ids_to_plot)
ax.set_ylabel('Unit ID')
ax.set_xlabel('Time (s)')
ax.set_title(f'Spike Raster Plot (First {len(unit_ids_to_plot)} Units)')
plt.tight_layout()

# Save the plot
output_path = "explore/spike_raster.png"
print(f"Saving plot to {output_path}")
plt.savefig(output_path)
plt.close(fig) # Close the figure

# Close the file handles
io.close()
h5_file.close()
print("Script finished.")