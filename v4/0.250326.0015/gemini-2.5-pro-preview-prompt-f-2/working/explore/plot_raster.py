# Explore spike data by creating a raster plot (revised)
# This script loads spike times for a few selected units
# from the NWB file and creates a raster plot.
# It avoids loading the entire units table into a DataFrame.
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

# Select a few units to plot
num_units_to_plot = 5
all_unit_ids = nwb.units.id[:] # Get all unit IDs
selected_unit_ids = all_unit_ids[:num_units_to_plot]

spike_times_list = []
actual_unit_ids_for_plot = [] # Store IDs of units that actually have spike times

for unit_id in selected_unit_ids:
    try:
        # Access spike times for the current unit
        # For a VectorIndex, .get(id) is used if the index is non-integer or if you want to be explicit.
        # .data[:] then accesses the actual spike times array.
        unit_spike_data = nwb.units['spike_times'].get(int(unit_id)) # unit_id from nwb.units.id[:] might be int64
        if unit_spike_data is not None:
             times = np.array(unit_spike_data.data[:]) # Convert memoryview to numpy array
             spike_times_list.append(times)
             actual_unit_ids_for_plot.append(unit_id)
        else:
            print(f"Warning: No spike time data found for unit_id {unit_id}")
    except Exception as e:
        print(f"Error accessing spike times for unit_id {unit_id}: {e}")

if not spike_times_list:
    print("No spike data found for the selected units. Exiting.")
    h5_file.close()
    io.close()
    exit()

# Update num_units_to_plot based on successfully loaded units
num_units_to_plot = len(spike_times_list)


# Create raster plot
sns.set_theme()
fig, ax = plt.subplots(figsize=(12, 6))

# Use a consistent color palette if desired, or let eventplot choose
# event_colors = sns.color_palette("husl", num_units_to_plot)

for i in range(num_units_to_plot):
    times = spike_times_list[i]
    # Plot only spikes within a certain time range for better visualization, e.g., 0-60 seconds
    time_range_max = 60  # seconds
    times_in_range = times[(times >= 0) & (times <= time_range_max)]
    
    # Further subsample if still too many spikes for clarity
    max_spikes_to_plot_per_unit = 1000
    if len(times_in_range) > max_spikes_to_plot_per_unit:
        indices = np.random.choice(len(times_in_range), max_spikes_to_plot_per_unit, replace=False)
        indices.sort() # Keep time order
        times_subset = times_in_range[indices]
    else:
        times_subset = times_in_range
        
    if len(times_subset) > 0:
      ax.eventplot(times_subset, lineoffsets=i + 1, linelengths=0.8) # removed color for simplicity
    else:
      print(f"No spikes in range [0, {time_range_max}s] for unit {actual_unit_ids_for_plot[i]}")


ax.set_yticks(np.arange(num_units_to_plot) + 1)
ax.set_yticklabels([f'Unit {uid}' for uid in actual_unit_ids_for_plot])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Unit ID')
ax.set_title(f'Raster Plot ({num_units_to_plot} units, 0-{time_range_max}s, max {max_spikes_to_plot_per_unit} spikes/unit)')
ax.set_xlim(0, time_range_max) # Set x-axis limits for clarity
plt.tight_layout()

# Save the plot
plt.savefig('explore/raster_plot.png')
print(f"Plot saved to explore/raster_plot.png")

# Close the HDF5 file and the NWBHDF5IO object
h5_file.close()
io.close()
print("NWB file processing complete.")