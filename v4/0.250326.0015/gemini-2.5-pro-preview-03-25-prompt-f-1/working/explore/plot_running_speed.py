# This script explores the running speed data from the NWB file.
# It loads the first 2000 data points of running_speed.data and
# its timestamps, then plots running speed over time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply seaborn theme
sns.set_theme()

print("Starting running speed data exploration script...")

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

        if 'running' in nwb.processing and 'running_speed' in nwb.processing['running'].data_interfaces:
            running_speed_series = nwb.processing['running'].data_interfaces['running_speed']
            
            num_points_to_plot = 2000
            
            print(f"Accessing first {num_points_to_plot} points of running speed data and timestamps...")
            # Access a subset of data and timestamps
            running_data_subset = running_speed_series.data[:num_points_to_plot]
            running_timestamps_subset = running_speed_series.timestamps[:num_points_to_plot]
            
            print(f"Data shape: {running_data_subset.shape}")
            print(f"Timestamps shape: {running_timestamps_subset.shape}")

            if running_data_subset.shape[0] == 0 or running_timestamps_subset.shape[0] == 0:
                print("No data points to plot. Exiting.")
            else:
                plt.figure(figsize=(12, 6))
                plt.plot(running_timestamps_subset, running_data_subset, label='Running Speed', color='green')
                plt.xlabel(f"Time ({running_speed_series.timestamps_unit})")
                plt.ylabel(f"Speed ({running_speed_series.unit})")
                plt.title(f"Running Speed (First {num_points_to_plot} points)")
                plt.legend()
                sns.despine()
                
                plot_path = "explore/running_speed_plot.png"
                plt.savefig(plot_path)
                print(f"Plot saved to {plot_path}")
        else:
            print("Running speed data not found in the NWB file at the expected location.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if io:
        pass 
    if h5_f:
        h5_f.close()
    if remote_f:
        remote_f.close()

print("Running speed data exploration script finished.")