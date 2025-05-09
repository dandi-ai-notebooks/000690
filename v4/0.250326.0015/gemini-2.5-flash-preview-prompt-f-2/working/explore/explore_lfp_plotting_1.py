# This script explores a subset of LFP data from a specific NWB file by plotting a few channels.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import os

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/"

# Ensure the explore directory exists for saving plots
os.makedirs("explore", exist_ok=True)

try:
    # Load
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()

    # Access LFP data
    # As indicated by the nwb-file-info tool output
    probe_1_lfp_data = nwb.acquisition["probe_1_lfp"].electrical_series["probe_1_lfp_data"]

    # Load a small subset of the data and timestamps to avoid excessive memory usage
    # Load data from channels 0, 10, and 20 for the first 50000 timepoints
    num_timepoints_to_load = 50000
    channels_to_plot = [0, 10, 20]
    lfp_data_subset = probe_1_lfp_data.data[0:num_timepoints_to_load, channels_to_plot]
    timestamps_subset = probe_1_lfp_data.timestamps[0:num_timepoints_to_load]

    # Plot the data
    plt.figure(figsize=(12, 6))
    for i, channel_index in enumerate(channels_to_plot):
        plt.plot(timestamps_subset, lfp_data_subset[:, i], label=f'Channel {channel_index}')

    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (Volts)')
    plt.title('Subset of LFP Data from Selected Channels')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig('explore/lfp_subset_plot_1.png')
    plt.close()

    print("LFP subset plot saved to explore/lfp_subset_plot_1.png")

except Exception as e:
    print(f"An error occurred: {e}")