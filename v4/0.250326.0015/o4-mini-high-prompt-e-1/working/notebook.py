# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
# **Warning: This notebook was AI-generated and has not been fully verified. Use caution when interpreting code or results.**

# %% [markdown]
# ## Overview
#
# This notebook explores Dandiset `000690` (version `0.250326.0015`) from the DANDI archive.  
# Dandiset URL: https://dandiarchive.org/dandiset/000690/0.250326.0015
#
# We will cover:
# - Loading dandiset metadata via the DANDI API  
# - Listing assets and selecting an NWB file  
# - Loading NWB file metadata and summarizing its structure  
# - Loading and visualizing example data (eye tracking, running speed)  
# - Advanced visualization combining eye position and running speed  
# - Summary and future directions  

# %% [markdown]
# ## Required Packages
# (Assumes these are already installed)

# %% 
import warnings
warnings.filterwarnings('ignore')

from itertools import islice
from dandi.dandiapi import DandiAPIClient
import remfile
import h5py
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()

# %% [markdown]
# ## Load Dandiset Metadata

# %% 
client = DandiAPIClient()
dandiset = client.get_dandiset("000690", "0.250326.0015")
metadata = dandiset.get_raw_metadata()

print(f"Dandiset name: {metadata.get('name')}")
print(f"Dandiset URL: {metadata.get('url')}")
print("\nFirst 5 assets:")
assets = dandiset.get_assets()
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Load NWB File
#
# Selecting the first NWB file for demonstration:
# `sub-692072/sub-692072_ses-1298465622.nwb`

# %% 
remote_url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(remote_url)
h5_file = h5py.File(remote_file, mode='r')
io = NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Loaded NWB session identifier: {nwb.identifier}")

# %% [markdown]
# ## NWB File Structure
#
# - **Acquisition**: data directly recorded in the session  
# - **Processing**: processed timeseries (e.g., running speed, stimulus times)  
# - **Intervals**: time interval tables for stimulus presentations  
# - **Electrode groups** / **Devices**  
# - **Units**: sorted spike times  
# - **Subject** metadata  

# %% 
print("Acquisition keys:", list(nwb.acquisition.keys()))
print("Processing modules:", list(nwb.processing.keys()))
print("Intervals:", list(nwb.intervals.keys()))
print("Electrode groups:", list(nwb.electrode_groups.keys()))
print("Devices:", list(nwb.devices.keys()))
print("Units columns:", nwb.units.colnames)
print("Subject:", {k: getattr(nwb.subject, k) for k in ['subject_id','age','sex','species']})

# %% [markdown]
# NWB file link on NeuroSift:  
# [Open in NeuroSift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=0.250326.0015)

# %% [markdown]
# ## Visualize Eye Tracking Data

# %% 
# Load a subset of eye tracking data to avoid large downloads
eye = nwb.acquisition['EyeTracking'].spatial_series['eye_tracking']
timestamps_eye = eye.timestamps[:1000]
data_eye = eye.data[:1000]

plt.figure(figsize=(8,3))
plt.plot(timestamps_eye, data_eye[:,0], label='X position')
plt.plot(timestamps_eye, data_eye[:,1], label='Y position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Eye Tracking Position (First 1000 samples)')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize Running Speed

# %% 
speed = nwb.processing['running'].data_interfaces['running_speed']
timestamps_speed = speed.timestamps[:1000]
data_speed = speed.data[:1000]

plt.figure(figsize=(8,3))
plt.plot(timestamps_speed, data_speed, color='tab:green')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Running Speed (First 1000 samples)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Combined Visualization
#
# Eye position colored by running speed

# %% 
plt.figure(figsize=(4,4))
plt.scatter(data_eye[:,0], data_eye[:,1], c=data_speed, cmap='viridis', s=5)
cbar = plt.colorbar()
cbar.set_label('Running speed (cm/s)')
plt.xlabel('Eye X (m)')
plt.ylabel('Eye Y (m)')
plt.title('Eye Position with Running Speed')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to:
# - Load DANDI metadata and list assets  
# - Load an NWB file remotely using PyNWB and `remfile`  
# - Summarize the NWB file structure  
# - Visualize key recorded and processed data  
#
# Future analyses could include:
# - Exploring LFP signals or spike trains from `nwb.units`  
# - Analysis of stimulus-response intervals via `nwb.intervals`  
# - Correlating behavioral metrics with neural data