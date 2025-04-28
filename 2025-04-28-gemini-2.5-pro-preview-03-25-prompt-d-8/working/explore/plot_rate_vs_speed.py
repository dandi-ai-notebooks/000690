# Explore relationship between spike rate and running speed
# This script loads spike times for a specific unit and running speed data,
# calculates firing rate and average speed in time bins,
# and generates a scatter plot of rate vs. speed.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Set seaborn theme for plotting
sns.set_theme()

# Define parameters
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
unit_id_to_plot = 18 # Choose unit ID based on previous raster plot
bin_width_seconds = 1.0 # Width of time bins
output_path = f"explore/rate_vs_speed_unit{unit_id_to_plot}.png"

print(f"Loading NWB file from: {url}")
try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    # Access units table and running speed
    if nwb.units is None or "running" not in nwb.processing or "running_speed" not in nwb.processing["running"].data_interfaces:
        print("Required data (units or running speed) not found.")
    else:
        units_df = nwb.units.to_dataframe()
        if unit_id_to_plot not in units_df.index:
            print(f"Unit ID {unit_id_to_plot} not found in units table.")
        else:
            # Get spike times for the selected unit
            unit_row = units_df.loc[unit_id_to_plot]
            unit_index = np.where(units_df.index == unit_id_to_plot)[0][0] # Get the integer index
            spike_times = nwb.units.spike_times_index[unit_index][:]
            print(f"Found {len(spike_times)} spikes for unit {unit_id_to_plot}.")

            # Get running speed data
            running_speed_ts = nwb.processing["running"].data_interfaces["running_speed"]
            running_speed_data = running_speed_ts.data[:]
            running_speed_timestamps = running_speed_ts.timestamps[:]
            print(f"Running speed data shape: {running_speed_data.shape}")

            # Determine time range and bins
            t_min = max(np.min(spike_times), np.min(running_speed_timestamps))
            t_max = min(np.max(spike_times), np.max(running_speed_timestamps))
            bins = np.arange(t_min, t_max + bin_width_seconds, bin_width_seconds)
            bin_centers = bins[:-1] + bin_width_seconds / 2

            if len(bins) <= 1:
                 print("Not enough time range overlap between spikes and running speed.")
            else:
                # Calculate firing rate in bins
                firing_rate, _ = np.histogram(spike_times, bins=bins)
                firing_rate = firing_rate / bin_width_seconds # Convert counts to Hz

                # Interpolate running speed onto bin centers
                # Use np.interp - ensure timestamps are monotonically increasing
                if not np.all(np.diff(running_speed_timestamps) >= 0):
                    sort_indices = np.argsort(running_speed_timestamps)
                    running_speed_timestamps = running_speed_timestamps[sort_indices]
                    running_speed_data = running_speed_data[sort_indices]
                    print("Sorted running speed timestamps.")

                interp_running_speed = np.interp(bin_centers, running_speed_timestamps, running_speed_data)

                # Remove bins where speed might be nonsensical (e.g., negative if interpolation goes wrong)
                # Or if firing rate is zero (optional, but avoids plotting many zero points)
                valid_indices = (firing_rate > 0) # Only plot where there are spikes

                # Plotting
                plt.figure(figsize=(8, 8))
                # Use seaborn's regplot for scatter + regression line (optional)
                # sns.regplot(x=interp_running_speed[valid_indices], y=firing_rate[valid_indices], scatter_kws={'alpha':0.5})
                plt.scatter(interp_running_speed[valid_indices], firing_rate[valid_indices], alpha=0.5, s=10) # Basic scatter

                plt.xlabel(f"Average Running Speed in Bin ({running_speed_ts.unit})")
                plt.ylabel(f"Firing Rate (Hz)")
                plt.title(f"Unit {unit_id_to_plot}: Firing Rate vs. Running Speed ({bin_width_seconds}s bins)")
                plt.grid(True)
                # Optional: Add a line for linear regression if desired
                # z = np.polyfit(interp_running_speed[valid_indices], firing_rate[valid_indices], 1)
                # p = np.poly1d(z)
                # plt.plot(np.sort(interp_running_speed[valid_indices]), p(np.sort(interp_running_speed[valid_indices])), "r--")


                # Save the plot
                plt.savefig(output_path)
                print(f"Rate vs. Speed plot saved to {output_path}")

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