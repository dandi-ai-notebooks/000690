# This script explores spike times from the NWB file's units table.
# It loads spike times for a subset of units and plots a raster.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set seaborn theme for plots
sns.set_theme()

def main():
    # Load NWB file
    url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
    print(f"Loading NWB file from: {url}")
    remote_file = remfile.File(url)
    try:
        with h5py.File(remote_file, 'r') as h5_file:
            with pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) as io:
                nwb = io.read()
                print("NWB file loaded successfully.")

                if nwb.units is not None and len(nwb.units.id[:]) > 0:
                    all_unit_ids = nwb.units.id[:]
                    
                    num_units_to_plot = 5 # Reduced number of units
                    if len(all_unit_ids) < num_units_to_plot:
                        num_units_to_plot = len(all_unit_ids)
                    
                    selected_unit_indices = range(num_units_to_plot)
                    # Store the actual IDs for labeling, but use indices for data access
                    selected_unit_ids_for_label = [all_unit_ids[i] for i in selected_unit_indices] 
                    print(f"Selected first {num_units_to_plot} unit IDs for raster: {selected_unit_ids_for_label}")

                    # Define a time window for the raster plot (e.g., first 60 seconds)
                    time_window_start = 0
                    time_window_end = 60 # seconds
                    
                    plt.figure(figsize=(12, 8))
                    
                    all_spikes_in_window = []
                    raster_plot_y_offset = 0
                    actual_units_plotted_ids = []


                    for i in selected_unit_indices:
                        # Access spike times by index from the HDF5 object directly
                        # This assumes nwb.units['spike_times'] is a RaggedArray or similar structure
                        # where each element corresponds to a unit's spike train.
                        # The nwb-file-info shows `nwb.units.spike_times_index[0]` etc.
                        # This implies spike_times is a ragged array.
                        # We need to get the spike times for the i-th unit.
                        # PyNWB's Units table handles this: spike_times is a VectorIndex referencing a VectorData
                        # units['spike_times'][i] should give the spike train for the i-th unit by its original index.
                        
                        spike_times = nwb.units['spike_times'][i] 
                        
                        # Filter spikes within the time window
                        spikes_in_window = spike_times[(spike_times >= time_window_start) & (spike_times <= time_window_end)]
                        
                        current_unit_id_for_label = all_unit_ids[i]
                        if len(spikes_in_window) > 0:
                            # For plotting, use y-offset for each unit
                            plt.eventplot(spikes_in_window, lineoffsets=raster_plot_y_offset, linelengths=0.8, colors='black')
                            all_spikes_in_window.extend(spikes_in_window)
                            actual_units_plotted_ids.append(current_unit_id_for_label)
                            raster_plot_y_offset += 1
                        else:
                            print(f"Unit ID {current_unit_id_for_label} (index {i}) has no spikes in the window {time_window_start}-{time_window_end}s.")
                    
                    actual_units_plotted_count = len(actual_units_plotted_ids)
                    if actual_units_plotted_count > 0:
                        plt.yticks(range(actual_units_plotted_count), actual_units_plotted_ids)
                        plt.ylabel("Unit ID")
                    else:
                        plt.yticks([])
                        plt.ylabel("Unit ID (No spikes in window)")
                        
                    plt.xlabel("Time (s)")
                    plt.title(f"Spike Raster ({actual_units_plotted_count} Units, {time_window_start}-{time_window_end}s)")
                    plt.xlim(time_window_start, time_window_end)
                    
                    plt.savefig("explore/spike_raster.png")
                    plt.close()
                    print(f"Saved spike_raster.png with {len(all_spikes_in_window)} spikes from {actual_units_plotted_count} units.")
                    if actual_units_plotted_count < num_units_to_plot:
                        print(f"Note: Only {actual_units_plotted_count} out of {num_units_to_plot} selected units had spikes in the window.")

                else:
                    print("Units data not found or empty in the NWB file.")
    finally:
        remote_file.close()

if __name__ == "__main__":
    main()