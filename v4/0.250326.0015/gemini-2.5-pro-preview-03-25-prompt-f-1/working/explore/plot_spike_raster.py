# This script explores the spike times data from the NWB file's units table.
# It loads spike times for a few units and generates a raster plot.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply seaborn theme
sns.set_theme()

print("Starting spike raster plot generation script...")

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from: {url}")
remote_f = None
h5_f = None
io = None
try:
    remote_f = remfile.File(url)
    h5_f = h5py.File(remote_f, 'r')
    with pynwb.NWBHDF5IO(file=h5_f, mode='r', load_namespaces=True) as nwb_io:
        nwb = nwb_io.read()
        print("NWB file loaded successfully.")

        if nwb.units is None or len(nwb.units.id[:]) == 0:
            print("No units data found in the NWB file.")
        else:
            units_df = nwb.units.to_dataframe()
            print(f"Total units found: {len(units_df)}")
            
            # Select a few units for plotting, e.g., first 10, or units with most spikes
            # For simplicity, let's try to select units that have a decent number of spikes.
            # We need to access spike_times using the VectorIndex interface
            
            num_units_to_plot = 10
            selected_unit_indices = []
            selected_unit_ids = []
            spike_times_list = []
            
            # Iterate through units to find some with spikes
            # The nwb.units.spike_times is a VectorIndex.
            # nwb.units.spike_times_index tells us where each unit's spikes start.
            # nwb.units.spike_times.data contains all spike times concatenated.
            
            valid_unit_counter = 0
            # Iterating by index because unit_id might not be sequential or start from 0
            for i in range(len(nwb.units.id[:])): 
                if valid_unit_counter >= num_units_to_plot:
                    break
                
                # nwb.units.spike_times is a VectorIndex object.
                # nwb.units.spike_times.data provides access to all concatenated spike times.
                # nwb.units.spike_times.index provides access to the boundary indices for each unit.
                
                # Keep these as HDF5 dataset references or VectorData objects initially
                all_spike_times_ds = nwb.units.spike_times.data 
                boundary_indices_ds = nwb.units.spike_times.index
                
                # Get spike times for unit i
                if i == 0:
                    start_idx = 0
                    # Load only the needed boundary index
                    end_idx = boundary_indices_ds[0] 
                else:
                    # Load only the needed boundary indices
                    start_idx = boundary_indices_ds[i-1]
                    end_idx = boundary_indices_ds[i]
                
                # Slice the data for the current unit
                current_unit_spike_times = all_spike_times_ds[start_idx:end_idx]

                if len(current_unit_spike_times) > 10: # Only consider units with at least 10 spikes
                    selected_unit_indices.append(valid_unit_counter) # This will be y-axis
                    selected_unit_ids.append(nwb.units.id[i]) # Actual unit ID for y-tick label
                    spike_times_list.append(current_unit_spike_times)
                    print(f"Selected Unit ID: {nwb.units.id[i]} (index {i}) with {len(current_unit_spike_times)} spikes.")
                    valid_unit_counter += 1
            
            if not spike_times_list:
                print("No units with sufficient spikes found to plot.")
            else:
                plt.figure(figsize=(12, 8))
                # Create an event plot (raster plot)
                plt.eventplot(spike_times_list, colors='black', lineoffsets=selected_unit_indices, linelengths=0.8)
                
                plt.yticks(selected_unit_indices, selected_unit_ids) # Use actual unit IDs for y-ticks
                plt.xlabel("Time (seconds)")
                plt.ylabel("Unit ID")
                plt.title(f"Spike Raster for Selected Units (up to {num_units_to_plot} units)")
                sns.despine()
                
                plot_path = "explore/spike_raster_plot.png"
                plt.savefig(plot_path)
                print(f"Plot saved to {plot_path}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if io:
        pass
    if h5_f:
        h5_f.close()
    if remote_f:
        remote_f.close()

print("Spike raster plot generation script finished.")