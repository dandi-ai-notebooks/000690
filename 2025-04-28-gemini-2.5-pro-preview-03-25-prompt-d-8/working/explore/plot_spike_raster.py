# Explore spike raster data
# This script loads spike times for a subset of units from the NWB file
# and generates a raster plot for a specific time interval.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set seaborn theme for plotting
sns.set_theme()

# Define parameters
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
t_start = 1000.0  # Start time in seconds
t_end = 1010.0    # End time in seconds
num_units_to_plot = 10 # Number of units to plot
output_path = "explore/spike_raster.png"

print(f"Loading NWB file from: {url}")
try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    # Access units table
    if nwb.units is None:
        print("Units data not found.")
    else:
        units_df = nwb.units.to_dataframe()
        num_total_units = len(units_df)
        print(f"Found {num_total_units} units.")

        if num_total_units == 0:
             print("No units found in the file.")
        else:
            # Select subset of units
            unit_indices_to_plot = np.arange(min(num_units_to_plot, num_total_units))
            selected_unit_ids = units_df.index[unit_indices_to_plot] # Get actual unit IDs

            spike_times_list = []
            plotted_unit_indices = [] # Keep track of which y-axis index maps to which unit

            print(f"Fetching spike times for units {selected_unit_ids.tolist()} between {t_start}s and {t_end}s...")
            for i, unit_index in enumerate(unit_indices_to_plot):
                # Access spike times for the unit using its original index
                all_spikes = nwb.units.spike_times_index[unit_index][:]
                # Filter spikes within the time interval
                spikes_in_interval = all_spikes[(all_spikes >= t_start) & (all_spikes <= t_end)]
                if len(spikes_in_interval) > 0:
                    spike_times_list.append(spikes_in_interval)
                    plotted_unit_indices.append(i) # Use sequential index for plotting
                else:
                    print(f"Unit ID {selected_unit_ids[i]} (index {unit_index}) has no spikes in the interval.")

            if not spike_times_list:
                print("No spikes found for the selected units in the specified time interval.")
            else:
                # Plot raster
                plt.figure(figsize=(12, 6))
                plt.eventplot(spike_times_list, linelengths=0.75, color='black')
                plt.yticks(range(len(plotted_unit_indices)), selected_unit_ids[plotted_unit_indices]) # Use actual unit IDs for labels
                plt.xlabel("Time (s)")
                plt.ylabel("Unit ID")
                plt.title(f"Spike Raster ({t_end - t_start}s interval)")
                plt.xlim(t_start, t_end)
                plt.grid(True, axis='x')

                # Save the plot
                plt.savefig(output_path)
                print(f"Raster plot saved to {output_path}")


except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Ensure resources are closed
    if 'io' in locals() and io is not None:
        try:
            io.close()
            print("NWBHDF5IO closed.")
        except Exception as e_close:
            print(f"Error closing NWBHDF5IO: {e_close}")
    if 'h5_file' in locals() and h5_file is not None:
        try:
            h5_file.close()
            print("H5py file closed.")
        except Exception as e_close:
            print(f"Error closing h5py file: {e_close}")
    if 'remote_file' in locals() and remote_file is not None:
        try:
            remote_file.close()
            print("Remfile closed.")
        except Exception as e_close:
            print(f"Error closing remfile: {e_close}")