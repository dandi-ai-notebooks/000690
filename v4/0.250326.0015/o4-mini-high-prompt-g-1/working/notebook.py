# %% [markdown]
# Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project

**Disclaimer**: This notebook was AI-generated and has not been fully verified. Use caution when interpreting the code or results.

## Overview

This notebook covers loading the Dandiset 000690 version 0.250326.0015 using the DANDI API, exploring its metadata, loading an NWB file, and visualizing LFP data.

### Dataset Description

The Allen Institute Openscope - Vision2Hippocampus project aims to understand how neural representations of simple and natural visual stimuli evolve through the mouse brain, from the lateral geniculate nucleus (LGN) and primary visual cortex (V1) to hippocampal regions. Mice were presented with:
- **Simple visual motion**: moving bars of light (varied width, speed, contrast, “disco” bar).
- **Complex movie stimuli**: natural scenes involving eagles and squirrels.

The dataset includes high-density extracellular electrophysiology recordings (LFP, spike sorting) obtained with Neuropixels probes across multiple brain regions.

Dandiset link: https://dandiarchive.org/dandiset/000690/0.250326.0015

## Required Packages

The following Python packages are required and should be pre-installed:

- itertools
- dandi.dandiapi
- pynwb
- h5py
- remfile
- numpy
- pandas
- matplotlib
- seaborn

# %% [markdown]
## Loading the Dandiset

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
## Selecting an NWB File

We select one representative session and probe for demonstration:

- **File path**: `sub-692072/sub-692072_ses-1298465622_probe-1_ecephys.nwb`
- **Rationale**: Contains LFP recordings from 73 channels at 625 Hz, suitable for basic visualization.

Download URL:

https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/

# %% [markdown]
## Loading the NWB File

# %%
import pynwb
import h5py
import remfile

# Remote NWB file URL
url = "https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/"

# Load NWB file with error handling
try:
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
except Exception as e:
    print("Error loading NWB file:", e)
    raise

# Print high-level information
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Identifier: {nwb.identifier}")

# %% [markdown]
## NWB File Structure

This section shows top-level HDF5 groups and key NWBFile attributes.

# %%
# Explore HDF5 structure
print("HDF5 root groups:")
for key in h5_file.keys():
    print(f"- {key}")
print("\nNWB acquisition keys:", list(nwb.acquisition.keys()))
print("NWB processing modules:", list(nwb.processing.keys()))
print("NWB electrode groups:", list(nwb.electrode_groups.keys()))

# %% [markdown]
## Electrode Metadata

Display the first 10 electrode channels with location and 3D coordinates.

# %%
import pandas as pd
from IPython.display import display

elec_df = nwb.electrodes.to_dataframe()
elec_df.reset_index(inplace=True)
elec_df.rename(columns={'index': 'id'}, inplace=True)
elec_subset = elec_df[['id', 'x', 'y', 'z', 'location', 'group_name']].head(10)
display(elec_subset)

# %% [markdown]
## Units Table

Check for and display spike unit metadata if available.

# %%
units_table = getattr(nwb, 'units', None)
if units_table is not None:
    units_df = units_table.to_dataframe()
    print(f"Units table with {len(units_df)} entries")
    display(units_df.head())
else:
    print("No units table found in this NWB file.")

# %% [markdown]
## Neurosift Link

Explore this NWB file interactively in Neurosift:

https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/79686db3-e4ef-4214-89f6-f2589ddb4ffe/download/&dandisetId=000690&dandisetVersion=0.250326.0015

# %% [markdown]
## Visualizing LFP Data

Local Field Potentials (LFP) reflect aggregate synaptic activity. Below we plot raw waveforms and a spectrogram for channel 0 as examples.

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

# Access the LFP ElectricalSeries
lfp_series = nwb.acquisition['probe_1_lfp'].electrical_series['probe_1_lfp_data']

# Load a small subset: first 10000 samples for the first 3 channels
num_samples = 10000
num_channels = 3
timestamps = lfp_series.timestamps[:num_samples]
data = lfp_series.data[:num_samples, :num_channels]

# Plot the first 3 channels with offset
plt.figure(figsize=(10, 6))
for ch in range(num_channels):
    plt.plot(timestamps, data[:, ch] + ch * 0.005, label=f"Channel {ch}")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.title("First 3 Channels of LFP Data (Offset for Clarity)")
plt.legend()
plt.tight_layout()

# %% [markdown]
### LFP Spectrogram

# %%
# Compute sampling rate
dt = np.median(np.diff(timestamps))
fs = 1.0 / dt

plt.figure(figsize=(10, 4))
plt.specgram(data[:, 0], NFFT=256, Fs=fs, noverlap=128, cmap='viridis')
plt.colorbar(label='Power')
plt.title("Spectrogram of LFP Channel 0")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.tight_layout()

# %% [markdown]
## Summary and Future Directions

This notebook demonstrated how to:

- Connect to the DANDI archive and load metadata  
- List and select assets in a Dandiset  
- Load an NWB file remotely using PyNWB with error handling  
- Explore the NWB file structure and electrode metadata  
- Visualize raw LFP waveforms and a spectrogram

Future directions include:

- Exploring other probe recordings (e.g., probes 0, 2, 3)  
- Analyzing spike unit data  
- Examining imaging NWB files  
- Implementing advanced analyses such as event-related LFP averages or connectivity metrics