# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus Project
# 
# This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview
# 
# The dataset is part of the Allen Institute's Openscope project, which explores neural responses to visual stimuli across various brain regions. The dataset contains data from extracellular electrophysiology studies performed on mice. For more details, visit the [Dandiset 000690](https://dandiarchive.org/dandiset/000690).

# %% [markdown]
# ## What the Notebook Covers
# 
# In this notebook, we will:
# - Load metadata and assets using the DANDI API.
# - Access and explore data from an NWB file.
# - Visualize select data to understand its structure and properties.

# %% [markdown]
# ## Required Packages
# - pynwb
# - dandi
# - h5py
# - matplotlib
# - remfile

# %%
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt

# %% [markdown]
# ## Loading Dandiset Metadata and Assets

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("000690")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# List the assets in the Dandiset
assets = list(dandiset.get_assets())
print(f"Found {len(assets)} assets in the dataset")
print("First 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Exploring an NWB File

# %%
# Load
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ## NWB Metadata

# %%
# Display basic metadata
print(f"Session ID: {nwb.identifier}")
print(f"Session Date: {nwb.session_start_time.isoformat()}")
print(f"Experiment Institution: {nwb.institution}")

# %%
# Display the available eye-tracking data
eye_tracking_data = nwb.acquisition["EyeTracking"].spatial_series["corneal_reflection_tracking"].data[0:50, :]
timestamps = nwb.acquisition["EyeTracking"].spatial_series["corneal_reflection_tracking"].timestamps[0:50]

# Plot a sample of the eye-tracking data
plt.figure(figsize=(12, 6))
plt.plot(timestamps, eye_tracking_data[:, 0], label='Eye X')
plt.plot(timestamps, eye_tracking_data[:, 1], label='Eye Y')
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.title("Eye Tracking Over Time")
plt.legend()
plt.show()

# %% [markdown]
# ## Future Directions
# 
# Further analysis could include exploring relationships between neuronal activities and visual stimuli, as well as more in-depth investigations within specific brain regions. Using the intervals data, one can understand the exact timing of stimulus presentations and correlate them with recorded neural responses.