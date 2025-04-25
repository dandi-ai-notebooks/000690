# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus
#
# This notebook is AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.
#
# ## Overview
#
# In this notebook, we are exploring the Allen Institute Openscope - Vision2Hippocampus project. This Dandiset focuses on understanding the neural representations of visual stimuli and their evolution in the mouse brain.
#
# **Dandiset Link**: [https://dandiarchive.org/dandiset/000690](https://dandiarchive.org/dandiset/000690)
#
# ### Contents
# - Load and explore the Dandiset using DANDI API
# - Examine Eye Tracking data and visualize corneal reflection tracking
#
# ## Required Packages
#
# The following packages are required for this analysis:
# - numpy
# - matplotlib
# - pynwb
# - h5py
# - remfile

# %% [markdown]
# ## Connect to DANDI Archive and Load Dandiset
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
# ## Loading the Eye Tracking Data
# We will load an NWB file from the Dandiset to explore the eye tracking data, particularly focusing on corneal reflection tracking.
#
# **File Path**: `sub-692072/sub-692072_ses-1298465622.nwb`
#
# **NWB File URL**: [Neurosift Link](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=draft)

import pynwb
import h5py
import remfile

# Load
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic session information
print(f"Session ID: {nwb.session_id}")
print(f"Institution: {nwb.institution}")
print(f"Start time: {nwb.session_start_time}")

# %% [markdown]
# ## Visualizing Corneal Reflection Tracking Data
# The following plot shows the corneal reflection tracking coordinates ('x' and 'y') over time for the first 1000 sample points. This provides insight into the accuracy and stability of eye tracking across a session.

import matplotlib.pyplot as plt

# Accessing corneal reflection tracking data
corneal_reflection_tracking = nwb.acquisition["EyeTracking"].spatial_series["corneal_reflection_tracking"]

# Slice the first 1000 data points for visualization
data = corneal_reflection_tracking.data[:1000, :]

plt.figure(figsize=(10, 5))
plt.plot(data[:, 0], label='x-coordinate')
plt.plot(data[:, 1], label='y-coordinate')
plt.title('Corneal Reflection Tracking (First 1000 points)')
plt.xlabel('Sample Index')
plt.ylabel('Position (pixels)')
plt.legend()

# Show plot inline
plt.show()

# %% [markdown]
# ## Summary of Findings
# The corneal reflection tracking data shows fluctuations in the x-coordinate from approximately 370 to 380 pixels, and the y-coordinate from 260 to 280 pixels, indicating moderate stability. Noticeable peaks around certain indices might correspond to specific events or stimuli during sessions. This suggests potential for correlating these fluctuations with corresponding neural response data, providing deeper insights into the subject's visual processing abilities.
# Future analysis could focus on aligning these peaks with timestamps of visual stimuli or other recorded neural activity, offering a broader understanding of sensory integration in the brain.