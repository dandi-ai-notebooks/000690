# explore_running_speed.py
# This script explores running speed data from the NWB file.
# It plots the running speed over a segment of time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("Starting explore_running_speed.py")

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

            # Access running speed data
            if "running" in nwb.processing and \
               "running_speed" in nwb.processing["running"].data_interfaces:
                running_speed_series = nwb.processing["running"].data_interfaces["running_speed"]
                
                data = running_speed_series.data
                timestamps = running_speed_series.timestamps

                print(f"Running speed data shape: {data.shape}")
                print(f"Running speed timestamps shape: {timestamps.shape}")

                if data.shape[0] > 0 and timestamps.shape[0] > 0:
                    # Select a subset of data to plot (e.g., first 5000 points)
                    num_points_to_plot = min(5000, data.shape[0])
                    
                    speed = data[:num_points_to_plot]
                    time_s = timestamps[:num_points_to_plot]

                    print(f"Plotting {num_points_to_plot} points.")

                    # Create plot
                    sns.set_theme()
                    plt.figure(figsize=(12, 6))
                    plt.plot(time_s, speed, label=f'Running Speed ({running_speed_series.unit})')
                    plt.ylabel(f'Speed ({running_speed_series.unit})')
                    plt.xlabel(f'Time ({running_speed_series.timestamps_unit})')
                    plt.title(f'Running Speed Over Time (First {num_points_to_plot} data points)')
                    plt.legend()
                    
                    plt.tight_layout()
                    plot_path = "explore/running_speed_vs_time.png"
                    plt.savefig(plot_path)
                    print(f"Plot saved to {plot_path}")
                    plt.close()
                else:
                    print("Running speed data or timestamps are empty.")
            else:
                print("Running speed data not found in the NWB file at the expected location.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if remote_file:
        remote_file.close()
        print("Remote file closed.")

print("Finished explore_running_speed.py")