# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
#
# **Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.
#
# ## Overview
#
# This notebook explores the dataset from the Allen Institute Openscope - Vision2Hippocampus project. The data consists of electrical recordings from mouse brains, aimed at understanding neural activity across various regions.
#
# [Dandiset 000690 - Full Details](https://dandiarchive.org/dandiset/000690)
#
# The notebook will cover loading the dataset, visualizing local field potentials (LFP), and understanding electrode placements.
#
# ## Requirements
#
# The following Python packages are required to run this notebook:
# - pynwb
# - h5py
# - remfile
# - pandas
# - matplotlib
# - numpy

# %% [markdown]
# ## Loading the Dandiset using DANDI API
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("000690")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List the assets in the Dandiset
assets = list(dandiset.get_assets())
print(f"\nFound {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading an NWB File
#
# We will use the file containing probe-1 recordings.
# Path: sub-692072/sub-692072_ses-1298465622_probe-1_ecephys.nwb
# URL: [Direct Download](https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/)

import pynwb
import h5py
import remfile

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Display some metadata
nwb.session_description, nwb.identifier, nwb.session_start_time

# %% [markdown]
# ## Visualizing LFP Data
#
# Here, we visualize the LFP data recorded from the first channel.

import numpy as np
import matplotlib.pyplot as plt

lfp_data = nwb.acquisition["probe_1_lfp_data"].data
timestamps = nwb.acquisition["probe_1_lfp_data"].timestamps

plt.figure(figsize=(10, 5))
plt.plot(timestamps[:1000], lfp_data[:1000, 0], label="Channel 0 LFP")
plt.title("LFP Data - First 1000 Points - Channel 0")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Electrode Positions
#
# The scatter plot below illustrates the spatial distribution of the electrodes on probe 1.

electrodes_table = nwb.acquisition["probe_1_lfp_data"].electrodes.table.to_dataframe()

plt.figure(figsize=(8, 8))
plt.scatter(electrodes_table["x"], electrodes_table["y"], c="r", label="Electrode Positions")
plt.title("Electrode Positions")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ## Summary
#
# This notebook presented an overview of the available data in Dandiset 000690, focusing on LFP visualization and electrode layout. Future analysis could delve into more probing questions, exploring cross-regional neural dynamics or looking at different stimulation conditions.