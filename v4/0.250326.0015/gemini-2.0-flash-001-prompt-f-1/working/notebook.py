# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project

# %% [markdown]
# **Important Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This notebook explores Dandiset 000690, version 0.250326.0015, which contains data from the Allen Institute Openscope - Vision2Hippocampus project. The project investigates how neural representations of visual stimuli evolve from the LGN through V1 and hippocampal regions.
#
# Link to the Dandiset: [https://dandiarchive.org/dandiset/000690/0.250326.0015](https://dandiarchive.org/dandiset/000690/0.250326.0015)

# %% [markdown]
# ## What this notebook covers
#
# This notebook demonstrates how to:
#
# *   Load the Dandiset metadata using the DANDI API.
# *   List the assets (files) available in the Dandiset.
# *   Load an NWB file from the Dandiset.
# *   Explore the contents of the NWB file, including EyeTracking and running speed data.
# *   Visualize the running speed data.

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook:
#
# *   `pynwb`
# *   `h5py`
# *   `remfile`
# *   `matplotlib`
# *   `numpy`

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
# ## Loading an NWB file and showing some metadata

# %% [markdown]
# We will load the NWB file `sub-692072/sub-692072_ses-1298465622.nwb` and show some of its metadata.
#
# The URL for this asset is: `https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/`

# %%
import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

nwb

# %% [markdown]
# ## Listing Key NWB File Metadata
#
# *   **session\_description**: Data and metadata for an Ecephys session
# *   **identifier**: 1298465622
# *   **session\_start\_time**: 2023-09-21T00:00:00-07:00
# *   **timestamps\_reference\_time**: 2023-09-21T00:00:00-07:00
# *   **file\_create\_date**: \[datetime.datetime(2025, 2, 25, 16, 36, 10, 235851, tzinfo=tzoffset(None, -28800))]

# %% [markdown]
# ## Exploring the EyeTracking data
#
# We will load the `EyeTracking` data and plot the pupil area over time.

# %%
import numpy as np

# Get the EyeTracking data
eye_tracking = nwb.acquisition['EyeTracking']
pupil_tracking = eye_tracking.pupil_tracking
area = np.array(pupil_tracking.area[:])

# Print the first 10 pupil areas
print("First 10 pupil areas:", area[:10])

# %% [markdown]
# ## Exploring Running Speed Data
#
# Let's examine the running speed data in this NWB file.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Get the running speed data
running = nwb.processing['running']
running_speed = running.data_interfaces['running_speed']
timestamps = np.array(running_speed.timestamps[:])
data = np.array(running_speed.data[:])

# Print the first 10 running speeds
print("First 10 running speeds:", data[:10])

# Plot running speed vs. time
plt.figure(figsize=(10, 6))
plt.plot(timestamps, data)
plt.xlabel("Time (s)")
plt.ylabel("Running Speed (cm/s)")
plt.title("Running Speed vs. Time")
plt.show()

# %% [markdown]
# Include a link to that NWB file on neurosift so the user can explore that if they wish: https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=draft

# %% [markdown]
# ## Summary and Possible Future Directions
#
# This notebook provided a basic introduction to exploring Dandiset 000690 and demonstrated how to load and visualize data from an NWB file within the Dandiset. Possible future directions for analysis include:
#
# *   Performing more detailed analysis of the EyeTracking data, such as identifying and analyzing blink events.
# *   Investigating the relationship between running speed and neural activity.
# *   Exploring the stimulus presentation data to understand how neural responses are related to different visual stimuli.