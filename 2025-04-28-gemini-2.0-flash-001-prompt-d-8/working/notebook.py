# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project

# %% [markdown]
# **Important Note:** This notebook was AI-generated and has not been fully verified. Exercise caution when interpreting the code or results.

# %% [markdown]
# This notebook provides an overview of Dandiset 000690, which contains data from the Allen Institute Openscope - Vision2Hippocampus project. The project aims to understand how neural representations of visual stimuli evolve from the LGN through V1 and hippocampal regions.
#
# You can find the Dandiset at: https://dandiarchive.org/dandiset/000690

# %% [markdown]
# This notebook will cover the following:
#
# 1.  Loading the Dandiset using the DANDI API.
# 2.  Loading an NWB file and displaying its metadata.
# 3.  Visualizing eye tracking data.
# 4.  Visualizing running wheel data.
# 5.  Summarizing findings and suggesting future directions.

# %% [markdown]
# **Required Packages:**
#
# *   pynwb
# *   h5py
# *   remfile
# *   matplotlib
# *   numpy
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
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# Now, let's load one of the NWB files in the Dandiset and show some metadata.
# We will load the file `sub-692072/sub-692072_ses-1298465622.nwb`.
# The URL for this asset is: https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/

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

print(nwb) # Print the NWB file object

# Print some basic metadata
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

# %% [markdown]
# You can explore this NWB file on neurosift:
#
# https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=draft

# %% [markdown]
# ### Contents of the NWB file:
#
# *   **acquisition:**
#     *   EyeTracking
#         *   EllipseEyeTracking
#         *   spatial\_series
#             *   corneal\_reflection\_tracking
#             *   eye\_tracking
#             *   pupil\_tracking
#         *   TimeSeries
#             *   likely\_blink
#     *   TimeSeries
#         *   raw\_running\_wheel\_rotation
#         *   running\_wheel\_signal\_voltage
#         *   running\_wheel\_supply\_voltage
# *   **processing:**
#     *   running
#         *   TimeSeries
#             *   running\_speed
#             *   running\_speed\_end\_times
#         *   running\_wheel\_rotation
#     *   stimulus
#         *   TimeSeries
#             *   timestamps
# *   **electrode\_groups:**
#     *   probeA, probeB, probeE, probeF
# *   **devices:**
#     *   probeA, probeB, probeE, probeF
# *   **intervals:**
#     *   TimeIntervals
#         *   Disco2SAC\_Wd15\_Vel2\_Bndry1\_Cntst0\_loop\_presentations
#         *   Disk\_Wd15\_Vel2\_Bndry1\_Cntst0\_loop\_presentations
#         *   GreenSAC\_Wd15\_Vel2\_Bndry1\_Cntst0\_loop\_presentations
#         *   Ring\_Wd15\_Vel2\_Bndry1\_Cntst0\_loop\_presentations
#         *   SAC\_Wd15\_Vel2\_Bndry1\_Cntst0\_loop\_presentations
#         *   SAC\_Wd15\_Vel2\_Bndry1\_Cntst1\_loop\_presentations
#         *   SAC\_Wd15\_Vel2\_Bndry2\_Cntst0\_loop\_presentations
#         *   SAC\_Wd15\_Vel2\_Bndry2\_Cntst0\_oneway\_presentations
#         *   SAC\_Wd15\_Vel2\_Bndry3\_Cntst0\_loop\_presentations
#         *   SAC\_Wd15\_Vel8\_Bndry1\_Cntst0\_loop\_presentations
#         *   SAC\_Wd45\_Vel2\_Bndry1\_Cntst0\_loop\_presentations
#         *   UD\_Wd15\_Vel2\_Bndry1\_Cntst0\_loop\_presentations
#         *   acurl\_Wd15\_Vel2\_Bndry1\_Cntst0\_oneway\_presentations
#         *   curl\_Wd15\_Vel2\_Bndry1\_Cntst0\_oneway\_presentations
#     *   invalid\_times
# *   **electrodes:**
#     *   Columns: location, group, group\_name, probe\_vertical\_position, probe\_horizontal\_position, probe\_id, local\_index, valid\_data, x, y, z, imp, filtering
# *   **units:**
#     *   Columns: recovery\_slope, l\_ratio, d\_prime, max\_drift, firing\_rate, isi\_violations, presence\_ratio, spread, velocity\_above, repolarization\_slope, cluster\_id, isolation\_distance, nn\_miss\_rate, waveform\_duration, waveform\_halfwidth, peak\_channel\_id, quality, velocity\_below, amplitude, PT\_ratio, snr, nn\_hit\_rate, cumulative\_drift, amplitude\_cutoff, silhouette\_score, local\_index, spike\_times, spike\_amplitudes, waveform\_mean
# *   **subject:**
#     *   age, genotype, sex, species, subject\_id, strain, specimen\_name, age\_in\_days
# *   **invalid\_times:**
#     *   start\_time, stop\_time, tags

# %% [markdown]
# Now, let's load and visualize some eye tracking data from the NWB file.
# We will plot the X and Y positions of the pupil over time.

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Use seaborn style, but only for this plot
plt.figure(figsize=(10, 5))
eye_tracking = nwb.acquisition['EyeTracking']
pupil_tracking = eye_tracking.pupil_tracking
eye_tracking_data = pupil_tracking.data
eye_tracking_timestamps = pupil_tracking.timestamps[:]

# Plot the eye tracking data (first 1000 samples)
num_samples = min(1000, len(eye_tracking_data))
plt.plot(eye_tracking_timestamps[:num_samples], eye_tracking_data[:num_samples, 0], label='X')
plt.plot(eye_tracking_timestamps[:num_samples], eye_tracking_data[:num_samples, 1], label='Y')
plt.xlabel('Time (s)')
plt.ylabel('Position (pixels)')
plt.title('Eye Tracking Data')
plt.legend()
plt.show()

# %% [markdown]
# The plot shows the X and Y positions of the eye over time. There's a noticeable deviation around second 23.

# %% [markdown]
# Now, let's visualize some running wheel data from the NWB file.
# We will plot the rotation of the running wheel over time.

# %%
import matplotlib.pyplot as plt
import numpy as np

# Get running wheel data
running_wheel_rotation = nwb.processing['running'].data_interfaces['running_wheel_rotation']
running_wheel_data = running_wheel_rotation.data
running_wheel_timestamps = running_wheel_rotation.timestamps[:]

# Plot the running wheel data (first 1000 samples)
num_samples = min(1000, len(running_wheel_data))
plt.figure(figsize=(10, 5))
plt.plot(running_wheel_timestamps[:num_samples], running_wheel_data[:num_samples], label='Rotation')
plt.xlabel('Time (s)')
plt.ylabel('Rotation (radians)')
plt.title('Running Wheel Data')
plt.legend()
plt.show()

# %% [markdown]
# The plot shows the rotation of the running wheel over time. The wheel is relatively still for the first portion of the capture and then it begins to rotate cyclically for the remainder of the capture.

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to load and explore data from Dandiset 000690, focusing on eye tracking and running wheel data.  Further analysis could involve:
#
# *   Investigating the relationship between eye movements and running wheel activity.
# *   Analyzing neural activity in relation to the presented visual stimuli.
# *   Exploring other data modalities available in the NWB file, such as LFP data.