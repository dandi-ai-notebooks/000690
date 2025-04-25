# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project

# %% [markdown]
# **Important Note:** This notebook was AI-generated and has not been fully verified. Use caution when interpreting the code or results.

# %% [markdown]
# This notebook provides an overview of Dandiset 000690, the Allen Institute Openscope - Vision2Hippocampus project.
# We will explore the data and metadata available in this Dandiset, including LFP data and electrode metadata.
#
# Link to the Dandiset: https://dandiarchive.org/dandiset/000690

# %% [markdown]
# This notebook will cover the following:
#
# *   Loading the Dandiset using the DANDI API
# *   Exploring the assets in the Dandiset
# *   Loading and visualizing LFP data from an NWB file
# *   Examining the electrode metadata

# %% [markdown]
# Required packages:
#
# *   pynwb
# *   h5py
# *   remfile
# *   matplotlib
# *   numpy
# *   pandas
# *   seaborn

# %%
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
# Load one of the NWB files and show some metadata.
# We will load the file `sub-692072/sub-692072_ses-1298465622_probe-1_ecephys.nwb`.
#
# Here is the link to the NWB file on neurosift:
# https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/&dandisetId=000690&dandisetVersion=draft

# %%
import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

nwb.session_description # (str) LFP data and associated info for one probe
nwb.identifier # (str) 1
nwb.session_start_time # (datetime) 2023-09-21T00:00:00-07:00
nwb.file_create_date # (list) [datetime.datetime(2025, 2, 25, 16, 39, 27, 898747, tzinfo=tzoffset(None, -28800))]
nwb.subject # (EcephysSpecimen)
nwb.subject.age # (str) P82D
nwb.subject.genotype # (str) wt/wt
nwb.subject.sex # (str) M
nwb.subject.species # (str) Mus musculus

# %% [markdown]
# Load and visualize LFP data from the NWB file.

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Get LFP data
acquisition = nwb.acquisition
probe_1_lfp = acquisition["probe_1_lfp"]
electrical_series = probe_1_lfp.electrical_series
probe_1_lfp_data = electrical_series["probe_1_lfp_data"]

# Load a small subset of the data and timestamps
num_samples = 1000
data = probe_1_lfp_data.data[:num_samples, :10]
timestamps = probe_1_lfp_data.timestamps[:num_samples]

# Convert LFP data to microvolts
data_uv = data * 1e6

# Plot the LFP data
plt.figure(figsize=(10, 5))
plt.plot(timestamps, data_uv)
plt.xlabel("Time (s)")
plt.ylabel("LFP (μV)")
plt.title("LFP Data from probe 1")
plt.show()

# %% [markdown]
# Examine the electrode metadata.

# %%
import pandas as pd
from IPython.display import display

# Get electrode metadata
electrodes = nwb.electrodes
electrodes_df = electrodes.to_dataframe()

# Print some of the electrode metadata
display(electrodes_df.head())
display(electrodes_df.describe())

# The 'imp' column contains NaN values

# %% [markdown]
# Summary:
#
# This notebook provided an overview of Dandiset 000690, the Allen Institute Openscope - Vision2Hippocampus project.
# We explored the data and metadata available in this Dandiset, including LFP data and electrode metadata.

# %% [markdown]
# Possible future directions:
#
# *   Explore other NWB files in the Dandiset.
# *   Analyze the LFP data in more detail.
# *   Investigate the relationship between the LFP data and the electrode metadata.