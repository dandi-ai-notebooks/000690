# This script explores the running_speed data from the NWB file.
# It loads the running_speed data and timestamps, plots them,
# and prints some basic statistics.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
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

                # Access running speed data
                if "running" in nwb.processing and "running_speed" in nwb.processing["running"].data_interfaces:
                    running_speed_ts = nwb.processing["running"]["running_speed"]
                    
                    running_speed_data = running_speed_ts.data[:]
                    running_speed_timestamps = running_speed_ts.timestamps[:]
                    
                    print(f"Running speed data shape: {running_speed_data.shape}")
                    print(f"Running speed timestamps shape: {running_speed_timestamps.shape}")
                    
                    if running_speed_data.size > 0 and running_speed_timestamps.size > 0:
                        print(f"Min running speed: {np.min(running_speed_data)}")
                        print(f"Max running speed: {np.max(running_speed_data)}")
                        print(f"Mean running speed: {np.mean(running_speed_data)}")
                        
                        # Plot running speed
                        plt.figure(figsize=(12, 6))
                        plt.plot(running_speed_timestamps, running_speed_data)
                        plt.xlabel("Time (s)")
                        plt.ylabel(f"Running Speed ({running_speed_ts.unit})")
                        plt.title("Running Speed Over Time")
                        plt.savefig("explore/running_speed.png")
                        plt.close()
                        print("Saved running_speed.png")
                    else:
                        print("Running speed data or timestamps are empty.")
                else:
                    print("Running speed data not found in the NWB file.")
    finally:
        remote_file.close()

if __name__ == "__main__":
    main()