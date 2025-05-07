# This script explores the eye tracking data from the NWB file.
# It loads the first 1000 data points of eye_tracking.data and
# its timestamps, then plots x and y eye position over time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Apply seaborn theme for better aesthetics
sns.set_theme()

print("Starting eye tracking data exploration script...")

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from: {url}")
remote_f = None
h5_f = None
io = None
try:
    remote_f = remfile.File(url) # Open remfile.File without context manager
    h5_f = h5py.File(remote_f, 'r') # Pass the remfile object to h5py.File
    with pynwb.NWBHDF5IO(file=h5_f, mode='r', load_namespaces=True) as nwb_io: # Use context manager for NWBHDF5IO
        nwb = nwb_io.read()
        print("NWB file loaded successfully.")

        eye_tracking_series = nwb.acquisition['EyeTracking'].spatial_series['eye_tracking']
        
        num_points_to_plot = 1000
        
        print(f"Accessing first {num_points_to_plot} points of eye tracking data and timestamps...")
        # Access a subset of data and timestamps
        eye_data_subset = eye_tracking_series.data[:num_points_to_plot, :]
        eye_timestamps_subset = eye_tracking_series.timestamps[:num_points_to_plot]
        
        print(f"Data shape: {eye_data_subset.shape}")
        print(f"Timestamps shape: {eye_timestamps_subset.shape}")

        if eye_data_subset.shape[0] == 0 or eye_timestamps_subset.shape[0] == 0:
            print("No data points to plot. Exiting.")
        else:
            fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            
            # Plot X-coordinate
            axs[0].plot(eye_timestamps_subset, eye_data_subset[:, 0], label='X-coordinate', color='blue')
            axs[0].set_ylabel(f"X-coordinate ({eye_tracking_series.unit})")
            axs[0].set_title(f"Eye Tracking Data (First {num_points_to_plot} points)")
            axs[0].legend()
            sns.despine(ax=axs[0])

            # Plot Y-coordinate
            axs[1].plot(eye_timestamps_subset, eye_data_subset[:, 1], label='Y-coordinate', color='red')
            axs[1].set_xlabel(f"Time ({eye_tracking_series.timestamps_unit})")
            axs[1].set_ylabel(f"Y-coordinate ({eye_tracking_series.unit})")
            axs[1].legend()
            sns.despine(ax=axs[1])

            plt.tight_layout()
            plot_path = "explore/eye_tracking_plot.png"
            plt.savefig(plot_path)
            print(f"Plot saved to {plot_path}")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if io: # This would be nwb_io now; NWBHDF5IO's __exit__ should handle closing the h5_f if it opened it.
        pass # If NWBHDF5IO did not open h5_f (e.g. if file was passed as path), it won't close it.
             # If file was passed as an open HDF5 file object, it won't close it.
    if h5_f: # h5py.File needs to be closed if not managed by NWBHDF5IO.
        h5_f.close()
    if remote_f: # remfile.File must be closed.
        remote_f.close()


print("Eye tracking data exploration script finished.")