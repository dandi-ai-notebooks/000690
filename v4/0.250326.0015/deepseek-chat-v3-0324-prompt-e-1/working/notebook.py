# %% [markdown]
# # Exploring Dandiset 000690: Vision2Hippocampus Project

# %% [markdown]
# **Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview
# This notebook explores data from the Allen Institute's OpenScope - Vision2Hippocampus project (Dandiset 000690). The project investigates how neural representations of visual stimuli evolve from the thalamus through visual cortex to hippocampus in mice.

# Key details:
# - **Dandiset URL:** [https://dandiarchive.org/dandiset/000690/0.250326.0015](https://dandiarchive.org/dandiset/000690/0.250326.0015)
# - **Subjects:** 3 mice (subject 692072 in this notebook)
# - **Stimuli:** Simple visual motion (bars of light) and complex naturalistic stimuli (movies)
# - **Techniques:** Multi-electrode extracellular electrophysiology recordings (Neuropixels 1.0 probes)
# - **Data types:** LFP, spike sorted units, stimulus information

# %% [markdown]
# ## Required Packages
# To run this notebook, you'll need:
# - dandi
# - pynwb
# - h5py
# - remfile 
# - numpy
# - matplotlib
# - pandas

# %% [markdown]
# ## Loading the Dandiset
# First we'll connect to the DANDI archive and load metadata about this Dandiset:

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("000690", "0.250326.0015")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset description: {metadata['description']}")
print("Contributors:", [c for c in metadata['contributor'] if isinstance(c, str)])

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Loading NWB File
# We'll examine data from the first probe (probe 0) of subject 692072. This contains LFP recordings from a Neuropixels probe.

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

# Print basic file info
print(f"Session ID: {nwb.session_id}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Probe: {nwb.devices['probeA'].description}")

# %% [markdown]
# ## NWB File Structure
# This NWB file contains LFP data recorded from a Neuropixels probe. Key components:

# - **acquisition/probe_0_lfp_data**: LFP data (10117092 timepoints × 95 channels)
# - **electrodes**: Table with metadata about each recording channel
# - **devices/probeA**: Information about the Neuropixels 1.0 probe
# - **subject**: Information about the mouse subject

# Explore this NWB file in Neurosift: [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/&dandisetId=000690&dandisetVersion=draft](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/&dandisetId=000690&dandisetVersion=draft)

# %% [markdown]
# ### Electrode Information
# Let's examine the electrode metadata to understand the recording setup:

# %%
# Get electrodes table as pandas DataFrame
electrodes_df = nwb.electrodes.to_dataframe()

# Show basic electrode statistics
print(f"Number of electrodes: {len(electrodes_df)}")
print("Electrode locations:")
print(electrodes_df['location'].value_counts())

# Visualize electrode positions
plt.figure(figsize=(10, 6))
plt.scatter(electrodes_df['y'], electrodes_df['z'], c=electrodes_df['x'], cmap='viridis')
plt.colorbar(label='X coordinate (posterior)')
plt.xlabel('Y (inferior)')
plt.ylabel('Z (right)')
plt.title('Electrode Positions in Brain')
plt.show()

# %% [markdown]
# ## Visualizing LFP Data
# We'll now load and visualize a small subset of the LFP data. Since the full dataset is very large (>10 million timepoints), we'll analyze a short segment from the middle of the recording.

# %%
# Get LFP data
lfp = nwb.acquisition['probe_0_lfp_data']
fs = nwb.electrode_groups['probeA'].lfp_sampling_rate  # Sampling rate (Hz)

# Plot a subset of the data (60 sec segment from middle of recording)
start_idx = len(lfp.timestamps) // 2  # Middle of recording
duration = 60  # seconds
n_samples = int(duration * fs)

# Load data for 10 channels (every 10th channel)
channel_idx = [i for i in range(0, 95, 10)]
data_samples = lfp.data[start_idx:start_idx+n_samples, channel_idx]
times = lfp.timestamps[start_idx:start_idx+n_samples]

# Create plot
plt.figure(figsize=(12, 8))
for i, ch in enumerate(channel_idx):
    offset = i * 0.5  # Offset traces for visualization
    plt.plot(times, data_samples[:, i] + offset, 
             label=f'Ch {ch} ({electrodes_df.iloc[ch]["location"]})')
    
plt.xlabel('Time (s)')
plt.ylabel('Voltage (offset)')
plt.title(f'LFP Traces ({duration}s segment)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary and Future Directions
# This notebook demonstrated how to:
# - Access data from the Vision2Hippocampus project on DANDI
# - Load and explore NWB files containing Neuropixels recordings
# - Visualize electrode positions and LFP data

# ### Potential Next Steps:
# - Analyze responses to specific visual stimuli (available in other NWB files in this dataset)
# - Compare activity patterns across different brain regions
# - Compute frequency-domain features from the LFP (e.g., power spectra)
# - Combine with spike data (available in other files) for multi-scale analysis