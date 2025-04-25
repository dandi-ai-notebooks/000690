# %%
# This script explores the electrode metadata in the NWB file.
# It loads the electrode metadata and prints some of it to the standard output.

import pynwb
import h5py
import remfile
import pandas as pd

# Load
url = "https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get electrode metadata
electrodes = nwb.electrodes
electrodes_df = electrodes.to_dataframe()

# Print some of the electrode metadata
print(electrodes_df.head())
print(electrodes_df.describe())