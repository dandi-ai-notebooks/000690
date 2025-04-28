# Explore running speed data
# This script loads running speed data and timestamps from the NWB file
# and generates a plot of running speed over time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set seaborn theme for plotting
sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from: {url}")
try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r') # Ensure read mode
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Ensure read mode
    nwb = io.read()
    print("NWB file loaded successfully.")

    # Access running speed data
    if "running" in nwb.processing and "running_speed" in nwb.processing["running"].data_interfaces:
        running_speed_ts = nwb.processing["running"].data_interfaces["running_speed"]
        running_speed_data = running_speed_ts.data[:]
        running_speed_timestamps = running_speed_ts.timestamps[:]
        print(f"Running speed data shape: {running_speed_data.shape}")
        print(f"Running speed timestamps shape: {running_speed_timestamps.shape}")

        # Plot running speed
        plt.figure(figsize=(12, 6))
        plt.plot(running_speed_timestamps, running_speed_data)
        plt.xlabel("Time (s)")
        plt.ylabel(f"Running Speed ({running_speed_ts.unit})")
        plt.title("Running Speed Over Time")
        plt.grid(True)

        # Save the plot
        output_path = "explore/running_speed.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        print("Running speed data not found in the expected location.")

except Exception as e:
    print(f"An error occurred: {e}")
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