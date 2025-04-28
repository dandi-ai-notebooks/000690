# explore/explore_electrodes.py
# Script to load and print electrode information

import pynwb
import h5py
import remfile
import pandas as pd
import os

# Create explore directory if it doesn't exist
os.makedirs("explore", exist_ok=True)

# Set pandas display options
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

try:
    print("Loading NWB file...")
    # URL of the NWB file
    url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"

    # Open the remote file
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
    nwb = io.read()
    print("NWB file loaded.")

    # Access the electrodes table
    print("\nAccessing electrodes table...")
    electrodes_df = nwb.electrodes.to_dataframe()
    print("Electrodes table loaded into DataFrame.")

    # Print the first 5 rows
    print("\nFirst 5 rows of the electrodes table:")
    print(electrodes_df.head())

    # Print unique locations
    print("\nUnique brain regions recorded by this probe:")
    if 'location' in electrodes_df.columns:
        unique_locations = electrodes_df['location'].unique()
        print(unique_locations)
    else:
        print("'location' column not found in the table.")

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