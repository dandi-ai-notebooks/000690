# %% [markdown]
# Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus Project

# %% [markdown]
# **Warning:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview
# This notebook explores Dandiset 000690 version 0.250326.0015 from the Allen Institute's OpenScope Vision2Hippocampus project. The dataset contains extracellular electrophysiology recordings from mouse brains using Neuropixels probes during visual stimulation.

# %% [markdown]
# - **Dandiset URL:** https://dandiarchive.org/dandiset/000690/0.250326.0015
# - **Study Goal:** Investigate how neural representations of visual stimuli evolve from thalamus through visual cortex to hippocampus
# - **Stimuli:** Simple visual motion (bars of light) and complex natural stimuli (movies)
# - **Subjects:** Mice with Neuropixels probes targeting visual cortex and hippocampus

# %% [markdown]
# ## Notebook Contents
# This notebook will demonstrate:
# 1. How to connect to and load data from the DANDI archive
# 2. Basic exploration of NWB file structure and metadata
# 3. Visualization of LFP data and electrode positions
# 4. Simple analysis of neural responses

# %% [markdown]
# ## Required Packages
# - pynwb
# - numpy
# - matplotlib
# - remfile
# - h5py

# %% [markdown]
# ## Loading the Dandiset

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
# ## Loading an NWB File
# We'll examine data from probe 0 of subject 692072:  
# `sub-692072/sub-692072_ses-1298465622_probe-0_ecephys.nwb`  
# Asset ID: ba8760f9-91fe-4c1c-97e6-590bed6a783b

# %%
import pynwb
import h5py
import remfile

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# ## NWB File Summary

# %%
print("Session Description:", nwb.session_description)
print("Session Start Time:", nwb.session_start_time)
print("Subject ID:", nwb.subject.subject_id)
print("Subject Age:", nwb.subject.age)
print("Subject Sex:", nwb.subject.sex)
print("Number of Electrodes:", len(nwb.electrodes.id[:]))
print("Probe Type:", nwb.devices["probeA"].description)

# %% [markdown]
# ## Exploring LFP Data

# %%
import numpy as np
import matplotlib.pyplot as plt

# Get LFP data and electrodes
lfp_data = nwb.acquisition['probe_0_lfp_data']
electrodes = nwb.electrodes.to_dataframe()

# Print data shape and sampling rate
print("LFP Data Shape:", lfp_data.data.shape)
print("Sampling Rate:", nwb.electrode_groups["probeA"].lfp_sampling_rate, "Hz")

# %% [markdown]
# ### LFP Traces (First 5 Channels)

# %%
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(lfp_data.timestamps[:1000], lfp_data.data[:1000, i] + i*100, 
             label=f'Channel {i}')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')
plt.title('LFP Traces (First 5 Channels)')
plt.legend()
plt.show()

# %% [markdown]
# ### Electrode Positions

# %%
plt.figure(figsize=(8, 8))
plt.scatter(electrodes['x'], electrodes['y'], c=electrodes['probe_vertical_position'])
plt.colorbar(label='Probe Vertical Position (um)')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Electrode Positions')
plt.show()

# %% [markdown]
# ## Basic Analysis
# Let's compute and visualize the power spectrum for one channel to see the frequency content.

# %%
from scipy import signal

# Compute power spectrum for first channel
fs = nwb.electrode_groups["probeA"].lfp_sampling_rate
f, Pxx = signal.welch(lfp_data.data[:10000, 0], fs=fs)

plt.figure(figsize=(10, 4))
plt.semilogy(f, Pxx)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [V**2/Hz]')
plt.title('Power Spectrum (Channel 0)')
plt.xlim(0, 100)  # Focus on lower frequencies
plt.show()

# %% [markdown]
# ## Summary and Next Steps
# We've demonstrated how to:
# - Load data from the Vision2Hippocampus Dandiset
# - Access and visualize LFP data from Neuropixels recordings
# - Examine electrode positions and basic signal properties

# %% [markdown]
# **Potential Next Steps:**
# - Analyze responses to specific stimuli
# - Compute cross-channel correlations
# - Examine spatial patterns in oscillatory activity
# - Investigate stimulus-locked responses

# %% [markdown]
# **Explore further on Neurosift:**  
# [Visualize this NWB file](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/&dandisetId=000690&dandisetVersion=draft)