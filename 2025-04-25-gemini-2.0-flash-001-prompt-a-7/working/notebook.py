# %% [markdown]
# Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project

# %% [markdown]
# **Important Note:** This notebook was AI-generated and has not been fully verified. Users should be cautious when interpreting the code or results.

# %% [markdown]
# This notebook provides an overview of the Dandiset 000690, which contains data from the Allen Institute Openscope - Vision2Hippocampus project. The project aims to understand how neural representations of visual stimuli evolve from the LGN through V1 to the hippocampus.
#
# Link to the Dandiset: https://dandiarchive.org/dandiset/000690

# %% [markdown]
# This notebook will cover the following:
# 1.  Loading the Dandiset metadata using the DANDI API.
# 2.  Listing the assets (files) available in the Dandiset.
# 3.  Loading an NWB file and exploring its contents.
# 4.  Visualizing data from the NWB file.
# 5.  Concluding with possible future directions for analysis.

# %% [markdown]
# ### Required Packages
# The following packages are required to run this notebook. Please ensure that they are installed on your system.
# - pynwb
# - h5py
# - remfile
# - matplotlib
# - numpy
# - pandas
# - seaborn

# %%
# Import necessary libraries
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %% [markdown]
# ### Loading the Dandiset

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
# ### Loading and Exploring an NWB File

# %% [markdown]
# We will load the first NWB file in the Dandiset: `sub-692072/sub-692072_ses-1298465622.nwb`.
# The asset ID is `fbcd4fe5-7107-41b2-b154-b67f783f23dc`, and the URL is `https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/`.
#
# Here's how to load this NWB file and explore some of its metadata.
#
# You can also explore this file on neurosift: https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=draft

# %%
# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Print some basic information about the NWB file
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

# %% [markdown]
# ### Exploring Eye Tracking Data
# This NWB file contains eye tracking data. Let's explore the `EyeTracking` acquisition.

# %%
# Access eye tracking data
acquisition = nwb.acquisition
EyeTracking = acquisition["EyeTracking"]
spatial_series = EyeTracking.spatial_series
eye_tracking = spatial_series["eye_tracking"]

# Print some information about the eye tracking data
print(f"Eye tracking data shape: {eye_tracking.data.shape}")
print(f"Eye tracking timestamps shape: {eye_tracking.timestamps.shape}")
print(f"Eye tracking unit: {eye_tracking.unit}")
eye_tracking_data = eye_tracking.data[:]
eye_tracking_timestamps = eye_tracking.timestamps[:]
eye_tracking_df = pd.DataFrame(eye_tracking_data, columns=['x', 'y'])
eye_tracking_df['timestamps'] = eye_tracking_timestamps

# %%
# Plot the eye tracking data
sns.set_theme()
plt.figure(figsize=(10, 5))
plt.plot(eye_tracking_df['timestamps'][:1000], eye_tracking_df['x'][:1000], label='x')
plt.plot(eye_tracking_df['timestamps'][:1000], eye_tracking_df['y'][:1000], label='y')
plt.xlabel('Time (s)')
plt.ylabel('Eye Position (meters)')
plt.title('Eye Tracking Data')
plt.legend()
plt.show()

# %% [markdown]
#  The above plot shows the x and y coordinates of the eye position over time.

# %% [markdown]
# ### Exploring Running Speed Data
# The running speed is stored in the `processing` module. Let's explore it.

# %%
# Access running speed data
processing = nwb.processing
running = processing["running"]
data_interfaces = running.data_interfaces
running_speed = data_interfaces["running_speed"]

# Print some information about the running speed data
print(f"Running speed data shape: {running_speed.data.shape}")
print(f"Running speed timestamps shape: {running_speed.timestamps.shape}")
print(f"Running speed unit: {running_speed.unit}")

running_speed_data = running_speed.data[:]
running_speed_timestamps = running_speed.timestamps[:]

running_speed_df = pd.DataFrame({'speed': running_speed_data, 'timestamps': running_speed_timestamps})

# %%
# Plot the running speed data
plt.figure(figsize=(10, 5))
plt.plot(running_speed_df['timestamps'][:1000], running_speed_df['speed'][:1000])
plt.xlabel('Time (s)')
plt.ylabel('Running Speed (cm/s)')
plt.title('Running Speed Data')
plt.show()

# %% [markdown]
# The above plot shows the running speed of the mouse over time.

# %% [markdown]
# ### Exploring Stimulus Presentation Times
# The stimulus presentation times are stored in the `intervals` module. Let's explore the `SAC_Wd15_Vel8_Bndry1_Cntst0_loop_presentations` interval.

# %%
# Access stimulus presentation times
intervals = nwb.intervals
stimulus_presentations = intervals["SAC_Wd15_Vel8_Bndry1_Cntst0_loop_presentations"]

# Print some information about the stimulus presentation times
print(f"Stimulus presentation times description: {stimulus_presentations.description}")
print(f"Stimulus presentation times columns: {stimulus_presentations.colnames}")

stimulus_presentations_df = stimulus_presentations.to_dataframe()

# Display the first few rows of the DataFrame
print(stimulus_presentations_df.head())

# %% [markdown]
# ### Combining Eye Tracking and Running Speed Data:

# %%
# Synchronize eye tracking and running speed
time_index = np.searchsorted(running_speed_df['timestamps'], eye_tracking_df['timestamps'])
time_index = np.clip(time_index, 0, len(running_speed_df) - 1)
eye_tracking_df['running_speed'] = running_speed_df['speed'].iloc[time_index].values

# Plot eye position vs running speed
plt.figure(figsize=(10, 5))
plt.scatter(eye_tracking_df['x'][:1000], eye_tracking_df['running_speed'][:1000], alpha=0.5)
plt.xlabel('Eye Position X (meters)')
plt.ylabel('Running Speed (cm/s)')
plt.title('Eye Position vs Running Speed')
plt.show()

# %% [markdown]
# ### Conclusions and Future Directions
# In this notebook, we have demonstrated how to load and explore data from Dandiset 000690 using the DANDI API and PyNWB. We loaded the Dandiset metadata, accessed the eye tracking and running speed data, visualized the data, and combined eye tracking and running speed data to examine their interrelationship.
#
# Possible future directions for analysis include:
# 1.  Exploring the neural activity data in the `ecephys` modules.
# 2.  Analyzing the relationship between eye tracking and stimulus presentation.
# 3.  Investigating the neural correlates of running speed and behavior.
# 4.  Performing more advanced signal processing and machine learning analyses on the neural data.