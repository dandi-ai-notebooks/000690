"""
# Explore NWB File Script

This script aims to explore the LFP data from a selected NWB file (probe-1_ecephys). It will load the data, inspect metadata, and generate initial plots to understand the data structure and insights.

We'll generate plots and save them as PNG files for review and use in the later stages.

"""

import h5py
import remfile
import pandas as pd
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import numpy as np

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access the LFP data
lfp_data = nwb.acquisition["probe_1_lfp_data"].data
timestamps = nwb.acquisition["probe_1_lfp_data"].timestamps
electrodes_table = nwb.acquisition["probe_1_lfp_data"].electrodes.table.to_dataframe()

# Generate and save a plot of the first 1000 points of the first channel
plt.figure(figsize=(10, 5))
plt.plot(timestamps[:1000], lfp_data[:1000, 0], label="Channel 0 LFP")
plt.title("LFP Data - First 1000 Points - Channel 0")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.savefig("explore/lfp_channel_0.png", dpi=300)
plt.close()

# Generate and save a plot of electrode positions
plt.figure(figsize=(8, 8))
plt.scatter(electrodes_table["x"], electrodes_table["y"], c="r", label="Electrode Positions")
plt.title("Electrode Positions")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.savefig("explore/electrode_positions.png", dpi=300)
plt.close()

# Close file handles
io.close()
h5_file.close()
remote_file.close()