# Explore stimulus presentation table
# This script loads a stimulus presentation TimeIntervals table from the NWB file,
# displays its structure, and plots the timing of the first few intervals.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set seaborn theme for plotting
sns.set_theme()

# Define parameters
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
# Choose a stimulus table name from the nwb-file-info output
stimulus_table_name = "SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations"
num_intervals_to_plot = 50
output_plot_path = "explore/stimulus_intervals.png"

print(f"Loading NWB file from: {url}")
try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file, 'r')
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB file loaded successfully.")

    # Access stimulus presentations table
    if nwb.intervals is None or stimulus_table_name not in nwb.intervals:
        print(f"Stimulus table '{stimulus_table_name}' not found.")
    else:
        stim_table = nwb.intervals[stimulus_table_name]
        print(f"\nAccessing stimulus table: {stimulus_table_name}")
        print(f"Description: {stim_table.description}\n")

        # Convert to DataFrame and display info
        try:
            stim_df = stim_table.to_dataframe()
            print("Columns:", stim_df.columns.tolist())
            print(f"\nFirst 5 rows of {stimulus_table_name}:\n", stim_df.head())
            print(f"\nTotal intervals in table: {len(stim_df)}")

            # Plot the first N intervals
            intervals_to_plot = stim_df.head(num_intervals_to_plot)
            plt.figure(figsize=(12, 8))

            for i, row in intervals_to_plot.iterrows():
                plt.plot([row['start_time'], row['stop_time']], [i, i], marker='|', markersize=10, linestyle='-', linewidth=2)

            plt.yticks(range(num_intervals_to_plot), intervals_to_plot.index[:num_intervals_to_plot]) # Use index as label
            plt.xlabel("Time (s)")
            plt.ylabel("Interval Index")
            plt.title(f"First {num_intervals_to_plot} Intervals for {stimulus_table_name}")
            # Adjust y-limits for better visibility
            plt.ylim(-1, num_intervals_to_plot)
            plt.gca().invert_yaxis() # Show 0 at the top

            # Save the plot
            plt.savefig(output_plot_path)
            print(f"\nStimulus interval plot saved to {output_plot_path}")

        except Exception as e_df:
            print(f"Error processing or plotting DataFrame: {e_df}")
            import traceback
            traceback.print_exc()


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