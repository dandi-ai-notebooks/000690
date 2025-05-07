# explore_spike_times.py
# This script explores spike times from the NWB file for a subset of units
# and creates a raster plot.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print("Starting explore_spike_times.py")

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from: {url}")
remote_file = None
try:
    remote_file = remfile.File(url)
    with h5py.File(remote_file, 'r') as h5_file:
        with pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True) as io:
            nwb = io.read()
            print("NWB file loaded successfully.")

            if nwb.units is None or len(nwb.units.id[:]) == 0:
                print("No units data found or units table is empty.")
            else:
                units_df = nwb.units.to_dataframe()
                print(f"Units dataframe loaded with shape: {units_df.shape}")
                
                num_units_to_plot = min(20, len(units_df))
                if num_units_to_plot == 0:
                    print("No units to plot.")
                else:
                    selected_units_df = units_df.head(num_units_to_plot)
                    print(f"Selected {num_units_to_plot} units for raster plot.")

                    # Determine a time window, e.g., first 60 seconds
                    # Or use stimulus presentation times if easily available
                    # For simplicity, let's use the first 60 seconds of the recording if timestamps_reference_time is available
                    # otherwise, find a reasonable range from spike data itself.
                    
                    plot_time_window_s = (0, 60) # Default: first 60 seconds

                    all_spike_times = np.concatenate([st for st in selected_units_df["spike_times"]])
                    if len(all_spike_times) > 0:
                        # If there are spikes, try to set a window based on them if the default is bad
                        min_spike_time = np.min(all_spike_times)
                        max_spike_time = np.max(all_spike_times)
                        # If default window is outside spike times, adjust
                        if plot_time_window_s[1] < min_spike_time or plot_time_window_s[0] > max_spike_time:
                             plot_time_window_s = (min_spike_time, min(max_spike_time, min_spike_time + 60))
                    
                    print(f"Plotting spike times in window: {plot_time_window_s} seconds.")

                    sns.set_theme()
                    plt.figure(figsize=(15, 8))
                    
                    event_colors = sns.color_palette("viridis", n_colors=num_units_to_plot)

                    
                    # Use unit IDs from the DataFrame index directly
                    unit_ids_for_plot = selected_units_df.index[:num_units_to_plot]

                    # Determine if actual unit IDs can be used for y-axis or if simple indices are better
                    use_actual_ids_for_y = all(isinstance(uid, (int, float)) and not np.isnan(uid) for uid in unit_ids_for_plot)
                    if use_actual_ids_for_y:
                        # Check if IDs are reasonably dense for plotting
                        min_id, max_id = np.min(unit_ids_for_plot), np.max(unit_ids_for_plot)
                        if max_id - min_id > num_units_to_plot * 5: # Heuristic for "too sparse"
                            use_actual_ids_for_y = False 
                    
                    ytick_locs = []
                    ytick_labels = []

                    if use_actual_ids_for_y:
                        print("Using actual unit IDs for y-axis.")
                        for i, unit_id in enumerate(unit_ids_for_plot):
                            spike_times_s = selected_units_df.loc[unit_id, "spike_times"]
                            spikes_in_window = spike_times_s[(spike_times_s >= plot_time_window_s[0]) & (spike_times_s <= plot_time_window_s[1])]
                            plt.eventplot(spikes_in_window, lineoffsets=unit_id, linelengths=0.8, colors=[event_colors[i % len(event_colors)]])
                            ytick_locs.append(unit_id)
                            ytick_labels.append(f"Unit {unit_id}")
                    else:
                        print("Using sequential indices for y-axis.")
                        for i, unit_id in enumerate(unit_ids_for_plot): # unit_id here is the actual ID from index
                            spike_times_s = selected_units_df.loc[unit_id, "spike_times"]
                            spikes_in_window = spike_times_s[(spike_times_s >= plot_time_window_s[0]) & (spike_times_s <= plot_time_window_s[1])]
                            plt.eventplot(spikes_in_window, lineoffsets=i, linelengths=0.8, colors=[event_colors[i % len(event_colors)]])
                            ytick_locs.append(i)
                            ytick_labels.append(f"Unit {unit_id} (idx {i})") # Label with actual ID but position with index

                    plt.yticks(ytick_locs, ytick_labels)
                    plt.xlabel(f"Time ({nwb.units.spike_times.unit if hasattr(nwb.units.spike_times, 'unit') else 's'})") # Assuming seconds if unit not specified
                    plt.ylabel("Unit")
                    plt.title(f"Spike Raster Plot (First {num_units_to_plot} Units, Window: {plot_time_window_s[0]:.2f}-{plot_time_window_s[1]:.2f} s)")
                    plt.tight_layout()
                    
                    plot_path = "explore/spike_raster_plot.png"
                    plt.savefig(plot_path)
                    print(f"Plot saved to {plot_path}")
                    plt.close()

except FileNotFoundError:
    print("Error: NWB file not found at the URL. Skipping spike time exploration.")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
finally:
    if remote_file:
        remote_file.close()
        print("Remote file closed.")

print("Finished explore_spike_times.py")