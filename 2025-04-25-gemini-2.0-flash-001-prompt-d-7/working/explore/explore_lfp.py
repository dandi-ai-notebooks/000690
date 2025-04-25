# %%
# This script explores the LFP data in the NWB file.
# It loads a small subset of the data and timestamps and saves a plot to a PNG file.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data
acquisition = nwb.acquisition
probe_1_lfp = acquisition["probe_1_lfp"]
electrical_series = probe_1_lfp.electrical_series
probe_1_lfp_data = electrical_series["probe_1_lfp_data"]

# Load a small subset of the data and timestamps
num_samples = 1000
data = probe_1_lfp_data.data[:num_samples, :10]
timestamps = probe_1_lfp_data.timestamps[:num_samples]

# Plot the LFP data
plt.figure(figsize=(10, 5))
plt.plot(timestamps, data)
plt.xlabel("Time (s)")
plt.ylabel("LFP (V)")
plt.title("LFP Data from probe 1")
plt.savefig("explore/lfp_data.png")
plt.close()