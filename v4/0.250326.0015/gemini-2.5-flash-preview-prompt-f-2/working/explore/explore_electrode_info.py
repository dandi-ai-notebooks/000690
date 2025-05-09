# This script explores the electrode table in the NWB file to understand electrode locations and characteristics.

import pynwb
import h5py
import remfile
import pandas as pd # Import pandas

# URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/"

try:
    # Load
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()

    # Access the electrode table
    # As indicated by the nwb-file-info tool output
    electrode_table = nwb.electrodes

    # Convert to a pandas DataFrame and print the head
    electrode_df = electrode_table.to_dataframe()
    print("Electrode table head:")
    print(electrode_df.head())

except Exception as e:
    print(f"An error occurred: {e}")