# This script explores stimulus presentation times from the NWB file.
# It loads start and stop times for a specific stimulus type and plots them.

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

                stimulus_key = "SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations"
                if stimulus_key in nwb.intervals:
                    stim_intervals = nwb.intervals[stimulus_key]
                    
                    start_times = stim_intervals.start_time[:]
                    stop_times = stim_intervals.stop_time[:]
                    
                    num_intervals = len(start_times)
                    print(f"Number of intervals for '{stimulus_key}': {num_intervals}")
                    
                    if num_intervals > 0:
                        # Plot a subset of intervals to avoid clutter
                        max_intervals_to_plot = 100
                        intervals_to_plot = min(num_intervals, max_intervals_to_plot)
                        
                        plt.figure(figsize=(12, 8))
                        for i in range(intervals_to_plot):
                            # Plot each interval as a horizontal line
                            plt.plot([start_times[i], stop_times[i]], [i, i], linewidth=2)
                        
                        plt.xlabel("Time (s)")
                        plt.ylabel("Stimulus Presentation Index")
                        plt.title(f"Stimulus Presentation Times (First {intervals_to_plot} of '{stimulus_key}')")
                        plt.ylim(-1, intervals_to_plot) # Adjust y-limits for better visualization
                        plt.savefig("explore/stimulus_times.png")
                        plt.close()
                        print(f"Saved stimulus_times.png showing first {intervals_to_plot} intervals.")
                    else:
                        print(f"No intervals found for '{stimulus_key}'.")
                else:
                    print(f"Stimulus interval data for '{stimulus_key}' not found.")
    finally:
        remote_file.close()

if __name__ == "__main__":
    main()