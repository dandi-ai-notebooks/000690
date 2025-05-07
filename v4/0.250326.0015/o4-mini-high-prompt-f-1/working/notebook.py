# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
#
# **Warning**: This notebook was AI-generated and has not been fully verified. Users should exercise caution when interpreting code and results.

# %% [markdown]
# ## Overview
# This notebook demonstrates how to:
# - Connect to DANDI and load Dandiset metadata  
# - List available assets  
# - Load an NWB file and explore its contents  
# - Visualize example data (eye-tracking positions)  
# - Summarize high-level structure and suggest future directions

# %% [markdown]
# ## Required packages
# - dandi (dandi-api-client)  
# - pynwb  
# - remfile  
# - h5py  
# - numpy  
# - matplotlib  

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
# ## Load NWB file
# We select the NWB file `sub-692072_ses-1298465622.nwb` for exploration.

# %%
import remfile
import h5py
import pynwb

nwb_url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file, mode='r')
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

# %% [markdown]
# ## High-level structure overview
# Use keys and counts to inspect major components.

# %%
# Acquisition, processing modules, intervals, electrodes, and units overview
print("Acquisition modules:", list(nwb.acquisition.keys()))
print("Processing modules:", list(nwb.processing.keys()))
print("Intervals (examples):", list(nwb.intervals.keys())[:5], "…")
print("Electrode count:", nwb.electrodes.id.shape[0])
print("Unit count:", len(nwb.units.id))

# %% [markdown]
# **Summary of contents**  
# - **Acquisition**: EyeTracking (multiple spatial series & blinks), raw wheel signals  
# - **Processing**: running (running_speed, etc.), stimulus (timestamps)  
# - **Intervals**: several TimeIntervals for stimulus presentations and invalid_times  
# - **Electrodes**: 1536 channels in `nwb.electrodes`  
# - **Units**: 2764 units in `nwb.units`  
# - **Subject metadata** available under `nwb.subject`

# %% [markdown]
# You can explore this NWB file interactively on Neurosift:  
# https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=draft

# %% [markdown]
# ## Visualizing eye-tracking data
# Plotting the first 1000 eye-tracking positions to illustrate data loading and visualization.

# %%
import numpy as np
import matplotlib.pyplot as plt

# Access eye-tracking series
eye_tracking = nwb.acquisition["EyeTracking"].spatial_series["eye_tracking"]

# Select subset for plotting
n_samples = min(1000, eye_tracking.data.shape[0])
data = eye_tracking.data[:n_samples, :]
timestamps = eye_tracking.timestamps[:n_samples]

# Create scatter plot
plt.figure(figsize=(6,6))
plt.scatter(data[:, 0], data[:, 1], c=timestamps, cmap='viridis', s=5)
plt.colorbar(label='Time (s)')
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title(f'Eye tracking positions (first {n_samples} samples)')
plt.show()

# %% [markdown]
# **Observation**: The scatter plot shows a dense cluster of gaze positions with a smooth time gradient, indicating continuous eye movement without obvious artifacts.

# %% [markdown]
# ## Conclusions and future directions
# This notebook has guided you through:
# - Connecting to DANDI and listing assets  
# - Loading and inspecting an NWB file  
# - Summarizing its major components  
# - Visualizing example eye-tracking data  
#
# **Possible future analyses**:
# - Explore running wheel and stimulus intervals in detail  
# - Inspect neural unit activity and summary statistics  
# - Combine behavioral and neural data to probe sensorimotor relationships  
# - Extend to additional NWB files in this Dandiset