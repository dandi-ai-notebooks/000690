# explore/plot_lfp_segment.py
# Script to plot the first second of LFP data for the first 5 channels

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create explore directory if it doesn't exist
os.makedirs("explore", exist_ok=True)

try:
    print("Loading NWB file...")
    # URL of the NWB file
    url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"

    # Open the remote file
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Specify mode='r'
    nwb = io.read()
    print("NWB file loaded.")

    # Access LFP data
    lfp_electrical_series = nwb.acquisition['probe_0_lfp'].electrical_series['probe_0_lfp_data']
    lfp_data = lfp_electrical_series.data
    lfp_timestamps = lfp_electrical_series.timestamps

    # Assuming lfp_sampling_rate is available from electrode group info or estimated
    # From the nwb-file-info tool output: nwb.electrode_groups['probeA'].lfp_sampling_rate = 625.0
    sampling_rate = 625.0 # Hz
    if hasattr(nwb.electrode_groups['probeA'], 'lfp_sampling_rate'): # Check if attribute exists
         sampling_rate = nwb.electrode_groups['probeA'].lfp_sampling_rate
         print(f"Using LFP sampling rate from electrode group: {sampling_rate} Hz")
    else:
         # Estimate sampling rate from timestamps if necessary
         if len(lfp_timestamps) > 1:
             sampling_rate = 1.0 / (lfp_timestamps[1] - lfp_timestamps[0])
             print(f"Estimated sampling rate from timestamps: {sampling_rate} Hz")
         else:
             print("Warning: Could not determine sampling rate accurately.")

    # Calculate number of samples for 1 second
    num_samples_1sec = int(sampling_rate * 1.0)

    print(f"Loading first {num_samples_1sec} samples ({1.0} second) for first 5 channels...")
    # Load the first second of data and timestamps for the first 5 channels
    # Important: Load timestamps first to get the time range
    ts_1sec = lfp_timestamps[:num_samples_1sec]
    data_1sec_5ch = lfp_data[:num_samples_1sec, :5]
    print("Data loaded.")

    # Get actual channel IDs for the first 5 channels
    electrodes_df = nwb.electrodes.to_dataframe()
    channel_ids = electrodes_df.index[:5].tolist() # Use index as ID if 'id' column not present or same
    if 'channel_name' in electrodes_df.columns: # Prefer channel_name if available
        channel_ids = electrodes_df['channel_name'][:5].tolist()
    elif 'id' in electrodes_df.columns:
        channel_ids = electrodes_df['id'][:5].tolist()
    print(f"Channel IDs: {channel_ids}")


    # Plotting
    print("Generating plot...")
    sns.set_theme()
    plt.figure(figsize=(15, 6))

    # Offset traces for visibility
    offset = np.std(data_1sec_5ch) * 3 # Calculate offset based on std dev
    for i in range(data_1sec_5ch.shape[1]):
        plt.plot(ts_1sec, data_1sec_5ch[:, i] + i * offset, label=f'Channel {channel_ids[i]}')

    plt.title(f'LFP Data Segment (First 1 second, First 5 Channels)')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (offsetted)')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Save the plot
    plot_filename = "explore/lfp_segment.png"
    plt.savefig(plot_filename)
    plt.close() # Close the figure to free memory
    print(f"Plot saved to {plot_filename}")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Ensure the NWB file is closed
    if 'io' in locals() and io:
        print("Closing NWB file.")
        io.close()
    print("Script finished.")