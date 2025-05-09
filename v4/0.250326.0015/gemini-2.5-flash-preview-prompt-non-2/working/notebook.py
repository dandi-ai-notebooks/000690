# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
#
# This notebook was AI-generated and has not been fully verified. Users should be cautious when interpreting the code or results presented.
#
# ## Dandiset Overview
#
# This Dandiset, "Allen Institute Openscope - Vision2Hippocampus project", contains extracellular electrophysiology data from mice presented with simple and complex visual stimuli.
#
# You can find more information about this Dandiset here: https://dandiarchive.org/dandiset/000690/0.250326.0015
#
# ## Notebook Contents
#
# This notebook will demonstrate how to:
# - Load the Dandiset metadata and list assets using the DANDI API.
# - Load a specific NWB file from the Dandiset.
# - Explore the structure and metadata of the selected NWB file.
# - Access and visualize a subset of the electrophysiology data.
#
# ## Required Packages
#
# The following packages are required to run this notebook:
# - `dandi`
# - `pynwb`
# - `h5py`
# - `remfile`
# - `numpy`
# - `matplotlib`
# - `seaborn`
# - `pandas`

# %% [markdown]
# ## Loading the Dandiset
#
# We will use the DANDI API to connect to the archive and load the specified Dandiset.

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("000690", "0.250326.0015")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Loading an NWB file
#
# We will now load a specific NWB file from the Dandiset using its asset ID and utilize `remfile` and `h5py` for remote access.
#
# We are loading the file at path `sub-692072/sub-692072_ses-1298465622_probe-0_ecephys.nwb` with asset ID `ba8760f9-91fe-4c1c-97e6-590bed6a783b`.
#
# The URL for this asset is: https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/

# %%
import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-b154-b67f783f23dc/download/" # corrected url
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Access basic NWB file metadata
print(f"NWB file session description: {nwb.session_description}")
print(f"NWB file identifier: {nwb.identifier}")
print(f"NWB file session start time: {nwb.session_start_time}")

# %% [markdown]
# ## NWB file contents summary
#
# This NWB file contains extracellular electrophysiology data from probe 0 taken during session 1298465622 for subject 692072.
#
# The main electrical series data is stored under `acquisition['probe_0_lfp']['electrical_series']['probe_0_lfp_data']`.
#
# The metadata for the electrodes is available in `nwb.electrodes`.
#
# Here is a summary of the relevant parts of the NWB file structure:
#
# ```
# ├── acquisition
# │   └── probe_0_lfp (LFP)
# │       └── electrical_series
# │           └── probe_0_lfp_data (ElectricalSeries)
# │               ├── data (Dataset: shape (10117092, 95), dtype float32)
# │               ├── timestamps (Dataset: shape (10117092,), dtype float64)
# │               └── electrodes (DynamicTableRegion) - links to nwb.electrodes
# ├── electrodes (DynamicTable)
# │   ├── id
# │   ├── location
# │   ├── group
# │   ├── group_name
# │   ├── probe_vertical_position
# │   ├── probe_horizontal_position
# │   ├── probe_id
# │   ├── local_index
# │   ├── valid_data
# │   ├── x
# │   ├── y
# │   ├── z
# │   ├── imp
# │   └── filtering
# ├── electrode_groups
# │   └── probeA (EcephysElectrodeGroup)
# └── devices
#    └── probeA (EcephysProbe)
#
# ```
#
# You can explore this NWB file on Neurosift: https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ba8760f9-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=0.250326.0015

# %% [markdown]
# ## Exploring Electrode Metadata
#
# We can view the electrode metadata as a pandas DataFrame to understand the channels and their properties.

# %%
import pandas as pd

# Convert electrode table to DataFrame
electrode_table = nwb.electrodes.to_dataframe()

# Display the first few rows of the electrode table
print("Electrode table (first 5 rows):")
print(electrode_table.head())

# Print the columns to see available metadata
print("\nElectrode table columns:", electrode_table.columns.tolist())

# %% [markdown]
# ## Loading and Visualizing LFP Data
#
# We will load a small subset of the LFP data for visualization. Since the dataset is large, we will only load the first 10,000 time points for the first 10 channels.

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn theme for better visualization
sns.set_theme()

# Access the LFP data and timestamps datasets
lfp_data_dataset = nwb.acquisition['probe_0_lfp']['electrical_series']['probe_0_lfp_data'].data
lfp_timestamps_dataset = nwb.acquisition['probe_0_lfp']['electrical_series']['probe_0_lfp_data'].timestamps

# Define the subset to load
num_timepoints = 10000
num_channels = 10

# Load the subset of data and timestamps
# Note: We load a slice of the dataset directly to avoid loading the entire data into memory
lfp_data_subset = lfp_data_dataset[0:num_timepoints, 0:num_channels]
lfp_timestamps_subset = lfp_timestamps_dataset[0:num_timepoints]

print(f"\nLoaded LFP data subset with shape: {lfp_data_subset.shape}")
print(f"Loaded LFP timestamps subset with shape: {lfp_timestamps_subset.shape}")

# Visualize the LFP data subset
plt.figure(figsize=(12, 6))
for i in range(num_channels):
    plt.plot(lfp_timestamps_subset, lfp_data_subset[:, i] + i * 100, lw=0.5) # Offset channels for visibility

plt.xlabel("Time (s)")
plt.ylabel("Channel (offset for visibility)")
plt.title("Subset of LFP Data")
plt.grid(True)
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook provided a basic introduction to accessing and exploring the electrophysiology data within Dandiset 000690. We demonstrated how to load the Dandiset, inspect assets, load an NWB file, view electrode metadata, and visualize a subset of the LFP data.
#
# Future analysis could involve:
# - Exploring other assets in the Dandiset, including potentially image data.
# - Analyzing the LFP data in more detail (e.g., spectral analysis).
# - Investigating the relationship between the neural activity and the presented visual stimuli.
# - Utilizing spike sorting results if available in other NWB files within the Dandiset.
#
# Remember to consult the Dandiset metadata and the NWB file structure for more detailed information and potential avenues for further research.

# %%
# No need to explicitly close the NWB file or remfile