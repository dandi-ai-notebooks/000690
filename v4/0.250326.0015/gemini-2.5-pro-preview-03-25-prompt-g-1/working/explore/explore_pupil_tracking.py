# explore_pupil_tracking.py
# This script explores pupil tracking data from the NWB file.
# It plots the x and y coordinates of pupil position over a short segment of time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

print("Starting explore_pupil_tracking.py")

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

            # Access pupil tracking data
            if "EyeTracking" in nwb.acquisition and \
               "pupil_tracking" in nwb.acquisition["EyeTracking"].spatial_series:
                pupil_tracking_series = nwb.acquisition["EyeTracking"].spatial_series["pupil_tracking"]
                
                data = pupil_tracking_series.data
                timestamps = pupil_tracking_series.timestamps

                print(f"Pupil tracking data shape: {data.shape}")
                print(f"Pupil tracking timestamps shape: {timestamps.shape}")

                if data.shape[0] > 0 and timestamps.shape[0] > 0:
                    # Select a subset of data to plot (e.g., first 1000 points)
                    num_points_to_plot = min(1000, data.shape[0])
                    
                    pupil_x = data[:num_points_to_plot, 0]
                    pupil_y = data[:num_points_to_plot, 1]
                    time_s = timestamps[:num_points_to_plot]

                    print(f"Plotting {num_points_to_plot} points.")

                    # Create plot
                    sns.set_theme()
                    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                    axs[0].plot(time_s, pupil_x, label='Pupil X')
                    axs[0].set_ylabel(f'Pupil X ({pupil_tracking_series.unit})')
                    axs[0].legend()
                    axs[0].set_title('Pupil Position Over Time (First 1000 data points)')

                    axs[1].plot(time_s, pupil_y, label='Pupil Y', color='orange')
                    axs[1].set_ylabel(f'Pupil Y ({pupil_tracking_series.unit})')
                    axs[1].set_xlabel(f'Time ({pupil_tracking_series.timestamps_unit})')
                    axs[1].legend()

                    plt.tight_layout()
                    plot_path = "explore/pupil_position_vs_time.png"
                    plt.savefig(plot_path)
                    print(f"Plot saved to {plot_path}")
                    plt.close(fig)
                else:
                    print("Pupil tracking data or timestamps are empty.")
            else:
                print("Pupil tracking data not found in the NWB file at the expected location.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if remote_file:
        remote_file.close()
        print("Remote file closed.")

print("Finished explore_pupil_tracking.py")