# This script explores the pupil_tracking data from the NWB file.
# It loads the pupil area data and timestamps, plots them,
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
                    
                    # It seems pupil_tracking itself has timestamps, and also pupil_tracking.timestamps is an EllipseSeries
                    # The nwb-file-info output suggests pupil_tracking.timestamps are the relevant timestamps for the data series
                    # and pupil_tracking.area is derived from the main data.
                    # Lets use pupil_tracking.timestamps.timestamps for time and pupil_tracking.area for pupil area.
                    
                    if hasattr(pupil_tracking_ts, "area") and hasattr(pupil_tracking_ts, "timestamps"):
                        # The nwb-file-info indicates pupil_tracking.timestamps is an EllipseSeries, 
                        # and data is within pupil_tracking.timestamps.timestamps
                        # However, more typically, the timestamps for pupil_tracking.area would be pupil_tracking_ts.timestamps
                        # Let's try the direct timestamps from pupil_tracking_ts first.
                        
                        pupil_area_data = pupil_tracking_ts.area[:]
                        
                        # Check if pupil_tracking_ts.timestamps is a dataset or another TimeSeries/EllipseSeries
                        if hasattr(pupil_tracking_ts.timestamps, 'data'): # It's an EllipseSeries
                             pupil_timestamps = pupil_tracking_ts.timestamps.timestamps[:]
                        else: # It's a direct dataset
                             pupil_timestamps = pupil_tracking_ts.timestamps[:]
                        
                        print(f"Pupil area data shape: {pupil_area_data.shape}")
                        print(f"Pupil timestamps shape: {pupil_timestamps.shape}")
                        
                        if pupil_area_data.size > 0 and pupil_timestamps.size > 0:
                            # Take a subset of data to speed up plotting if it's too large
                            num_points_to_plot = 50000
                            if len(pupil_timestamps) > num_points_to_plot:
                                print(f"Plotting a subset of {num_points_to_plot} points for pupil area.")
                                indices = np.linspace(0, len(pupil_timestamps) - 1, num_points_to_plot, dtype=int)
                                pupil_timestamps_subset = pupil_timestamps[indices]
                                pupil_area_data_subset = pupil_area_data[indices]
                            else:
                                pupil_timestamps_subset = pupil_timestamps
                                pupil_area_data_subset = pupil_area_data
                            
                            # Remove NaNs before plotting and stats
                            nan_mask = ~np.isnan(pupil_area_data_subset)
                            pupil_timestamps_clean = pupil_timestamps_subset[nan_mask]
                            pupil_area_data_clean = pupil_area_data_subset[nan_mask]
                            
                            if pupil_area_data_clean.size > 0:
                                print(f"Min pupil area (cleaned): {np.min(pupil_area_data_clean)}")
                                print(f"Max pupil area (cleaned): {np.max(pupil_area_data_clean)}")
                                print(f"Mean pupil area (cleaned): {np.mean(pupil_area_data_clean)}")
                            
                                plt.figure(figsize=(12, 6))
                                plt.plot(pupil_timestamps_clean, pupil_area_data_clean)
                            else:
                                print("Pupil area data is all NaN after subsetting.")
                                # Create an empty plot if all data is NaN after cleaning
                                plt.figure(figsize=(12, 6))
                                plt.plot([], [])


                            plt.xlabel("Time (s)")
                            plt.ylabel(f"Pupil Area ({pupil_tracking_ts.unit}^2)") # Assuming area is in unit^2
                            plt.title("Pupil Area Over Time")
                            plt.savefig("explore/pupil_area.png")
                            plt.close()
                            print("Saved pupil_area.png")
                        else:
                            print("Pupil area data or timestamps are empty.")
                    else:
                        print("Pupil area or timestamps data fields not found in pupil_tracking_ts.")
                else:
                    print("Pupil tracking data not found in the NWB file.")
    finally:
        remote_file.close()

if __name__ == "__main__":
    main()