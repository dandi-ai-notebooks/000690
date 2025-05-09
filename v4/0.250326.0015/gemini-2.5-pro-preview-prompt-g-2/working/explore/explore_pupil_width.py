# This script explores the pupil_tracking data (width) from the NWB file.
# It loads the pupil width data and timestamps, plots them,
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

                # Access pupil tracking data
                if "EyeTracking" in nwb.acquisition and \
                   "pupil_tracking" in nwb.acquisition["EyeTracking"].spatial_series:
                    
                    pupil_tracking_ts = nwb.acquisition["EyeTracking"].spatial_series["pupil_tracking"]
                    
                    if hasattr(pupil_tracking_ts, "width") and hasattr(pupil_tracking_ts, "timestamps"):
                        pupil_width_data = pupil_tracking_ts.width[:]
                        
                        if hasattr(pupil_tracking_ts.timestamps, 'data'): # It's an EllipseSeries
                             pupil_timestamps = pupil_tracking_ts.timestamps.timestamps[:]
                        else: # It's a direct dataset
                             pupil_timestamps = pupil_tracking_ts.timestamps[:]
                        
                        print(f"Pupil width data shape: {pupil_width_data.shape}")
                        print(f"Pupil timestamps shape: {pupil_timestamps.shape}")
                        
                        if pupil_width_data.size > 0 and pupil_timestamps.size > 0:
                            # Take a subset of data to speed up plotting
                            num_points_to_plot = 50000
                            if len(pupil_timestamps) > num_points_to_plot:
                                print(f"Plotting a subset of {num_points_to_plot} points for pupil width.")
                                indices = np.linspace(0, len(pupil_timestamps) - 1, num_points_to_plot, dtype=int)
                                pupil_timestamps_subset = pupil_timestamps[indices]
                                pupil_width_data_subset = pupil_width_data[indices]
                            else:
                                pupil_timestamps_subset = pupil_timestamps
                                pupil_width_data_subset = pupil_width_data
                            
                            # Remove NaNs before plotting and stats
                            nan_mask = ~np.isnan(pupil_width_data_subset)
                            pupil_timestamps_clean = pupil_timestamps_subset[nan_mask]
                            pupil_width_data_clean = pupil_width_data_subset[nan_mask]
                            
                            if pupil_width_data_clean.size > 0:
                                print(f"Min pupil width (cleaned): {np.min(pupil_width_data_clean)}")
                                print(f"Max pupil width (cleaned): {np.max(pupil_width_data_clean)}")
                                print(f"Mean pupil width (cleaned): {np.mean(pupil_width_data_clean)}")
                            
                                plt.figure(figsize=(12, 6))
                                plt.plot(pupil_timestamps_clean, pupil_width_data_clean)
                                plt.xlabel("Time (s)")
                                plt.ylabel(f"Pupil Width ({pupil_tracking_ts.unit})")
                                plt.title("Pupil Width Over Time")
                                plt.savefig("explore/pupil_width.png")
                                plt.close()
                                print("Saved pupil_width.png")
                            else:
                                print("Pupil width data is all NaN after subsetting and cleaning.")
                                # Create an empty plot if all data is NaN
                                plt.figure(figsize=(12, 6))
                                plt.plot([],[])
                                plt.xlabel("Time (s)")
                                plt.ylabel(f"Pupil Width ({pupil_tracking_ts.unit})")
                                plt.title("Pupil Width Over Time (No Valid Data)")
                                plt.savefig("explore/pupil_width.png")
                                plt.close()
                                print("Saved pupil_width.png (empty plot).")
                        else:
                            print("Pupil width data or timestamps are empty.")
                    else:
                        print("Pupil width or timestamps data fields not found in pupil_tracking_ts.")
                else:
                    print("Pupil tracking data not found in the NWB file.")
    finally:
        remote_file.close()

if __name__ == "__main__":
    main()