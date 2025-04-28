# %% [markdown]
# Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project

# %% [markdown]
# **Important Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This notebook provides an exploration of Dandiset 000690, which contains data from the Allen Institute Openscope - Vision2Hippocampus project. The project investigates how neural representations of visual stimuli evolve from the LGN through V1 and hippocampal regions.
#
# You can find more information about this Dandiset at: https://dandiarchive.org/dandiset/000690

# %% [markdown]
# ## What this Notebook Covers
#
# This notebook will guide you through the process of loading and visualizing data from this Dandiset. We will cover:
#
# 1.  Loading the Dandiset metadata
# 2.  Accessing and exploring an NWB file
# 3.  Visualizing eye tracking data
# 4.  Visualizing running wheel data

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook. Please ensure that they are installed in your environment.
#
# *   pynwb
# *   h5py
# *   remfile
# *   matplotlib
# *   numpy

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
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Loading an NWB File and Exploring Metadata
#
# In this section, we will load one of the NWB files from the Dandiset and explore its metadata. We will load the file `sub-692072/sub-692072_ses-1298465622.nwb`.
# The URL for this asset is: https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/
#
# You can explore this file in NeuroSift here: https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=draft

# %%
import pynwb
import h5py
import remfile

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

nwb

# %% [markdown]
# The NWB file contains various data interfaces, including:
#
# *   **EyeTracking:** Contains data related to eye movements, including corneal reflection, pupil tracking, and eye position.
# *   **Running:** Contains data related to running speed and wheel rotation.
# *   **Stimulus:** Contains data related to stimulus presentation times.
# *   **ElectrodeGroups:**  Contains information about the electrode groups and their properties.
# *   **Intervals:** Contains information about different experimental intervals and stimulus presentations.
# *   **Units:** Contains information about the recorded units, their spike times, and waveforms.

# %% [markdown]
# ## Visualizing Eye Tracking Data
#
# Here, we will visualize the corneal reflection tracking data from the NWB file.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Get eye tracking data
eye_tracking = nwb.acquisition['EyeTracking']
corneal_reflection_tracking = eye_tracking.spatial_series['corneal_reflection_tracking']
data = corneal_reflection_tracking.data
timestamps = corneal_reflection_tracking.timestamps[:]

# Plot the first 1000 data points
n_samples = min(1000, data.shape[0])
plt.figure(figsize=(10, 5))
plt.plot(timestamps[:n_samples], data[:n_samples, 0], label='X')
plt.plot(timestamps[:n_samples], data[:n_samples, 1], label='Y')
plt.xlabel('Time (s)')
plt.ylabel('Position (meters)')
plt.title('Corneal Reflection Tracking Data (First 1000 Samples)')
plt.legend()
plt.show()

# %% [markdown]
# The plot above shows the X and Y positions of the corneal reflection over time. Notice the spike in the Y data near 22 seconds.

# %% [markdown]
# ## Visualizing Running Wheel Data
#
# Next, we will visualize the running wheel data from the NWB file.

# %%
# Get running speed data
running_speed = nwb.processing['running'].data_interfaces['running_speed']
data = running_speed.data
timestamps = running_speed.timestamps

# Plot the first 1000 data points
n_samples = min(1000, data.shape[0])
plt.figure(figsize=(10, 5))
plt.plot(timestamps[:n_samples], data[:n_samples], label='Running Speed')
plt.xlabel('Time (s)')
plt.ylabel('Speed (cm/s)')
plt.title('Running Speed Data (First 1000 Samples)')
plt.legend()
plt.show()

# %% [markdown]
# The plot above shows the running speed of the animal over time. The data shows periods of acceleration and deceleration.

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook has provided a basic introduction to exploring Dandiset 000690. We have demonstrated how to load the Dandiset metadata, access NWB files, and visualize eye tracking and running wheel data.
#
# Possible future directions for analysis include:
#
# *   Exploring the relationship between eye movements and running speed.
# *   Analyzing the neural activity recorded in the ecephys data.
# *   Investigating the stimulus presentation times and their correlation with neural responses.