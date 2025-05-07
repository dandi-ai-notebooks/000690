# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
#
# **Important:** This notebook was AI-generated and has not been fully verified.
# Please be cautious when interpreting the code or results.
#
# This notebook explores the Allen Institute Openscope - Vision2Hippocampus project, available on the DANDI Archive.
#
# [Dandiset 000690](https://dandiarchive.org/dandiset/000690/0.250326.0015)
#
# ## What this notebook will cover:
#
# *   Loading the Dandiset metadata
# *   Listing the assets in the Dandiset
# *   Loading an NWB file from the Dandiset
# *   Exploring the contents of the NWB file
# *   Visualizing data from the NWB file
#
# ## Required packages:
#
# *   pynwb
# *   h5py
# *   remfile
# *   numpy
# *   matplotlib
# *   seaborn

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
# In this section, we will load an NWB file and explore its contents.
# We will load the file `sub-692072/sub-692072_ses-1298465622_probe-0_ecephys.nwb`.
# The URL for this file is: `https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/`

# %%
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# Load
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(nwb)
print(f"Dandiset name: {nwb.session_description}")
print(f"Dandiset URL: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

# %% [markdown]
# You can explore this NWB file on neurosift using the following link:
#
# [Neurosift link](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/&dandisetId=000690&dandisetVersion=draft)

# %%
# Load LFP data and timestamps
lfp_data = nwb.acquisition["probe_0_lfp"].electrical_series["probe_0_lfp_data"].data
lfp_timestamps = nwb.acquisition["probe_0_lfp"].electrical_series["probe_0_lfp_data"].timestamps

# Select a small segment of data
start_time = 100
end_time = 101
start_index = np.searchsorted(lfp_timestamps, start_time)
end_index = np.searchsorted(lfp_timestamps, end_time)

lfp_data_segment = lfp_data[start_index:end_index, 0]
lfp_timestamps_segment = lfp_timestamps[start_index:end_index]

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(lfp_timestamps_segment, lfp_data_segment)
plt.xlabel("Time (s)")
plt.ylabel("LFP (V)")
plt.title("LFP Data from Probe 0")
plt.show()