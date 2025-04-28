# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus Project

# %% [markdown]
# > **Disclaimer:** This notebook was AI-generated to help explore Dandiset 000690. It has not been rigorously verified by human experts. Please exercise caution when interpreting the code, results, or conclusions. Independent validation is recommended before drawing firm scientific conclusions based on this analysis.

# %% [markdown]
# ## Overview
#
# This notebook provides an introduction to exploring Dandiset 000690, titled "Allen Institute Openscope - Vision2Hippocampus project".
#
# **Dandiset Description:** Extensive research shows that visual cortical neurons respond to specific stimuli, e.g. the primary visual cortical neurons respond to bars of light with specific orientation. In contrast, the hippocampal neurons are thought to encode not specific stimuli but instead represent abstract concepts such as space, time and events. How is this abstraction computed in the mouse brain? Specifically, how does the representation of simple visual stimuli evolve from the thalamus, which is a synapse away from the retina, through primary visual cortex, higher order visual areas and all the way to hippocampus, that is farthest removed from the retina? The current OpenScope project aims to understand how the neural representations of simple and natural stimuli evolve from the LGN through V1, and most hippocampal regions, as well as some of the frontal areas. Data were collected using Neuropixels probes, measuring extracellular electrophysiology (LFP and potentially spikes, although spikes are usually in separate files) from mice presented with various visual stimuli (bars of light, movies).
#
# **Dandiset Link:** [https://dandiarchive.org/dandiset/000690](https://dandiarchive.org/dandiset/000690/draft)
#
# **Keywords:** mouse, neuropixel, extracellular electrophysiology, neocortex, hippocampus, Entorhinal cortex, excitatory, inhibitory, vision, movies
#
# **This notebook covers:**
# 1. Connecting to the DANDI archive and retrieving basic Dandiset metadata.
# 2. Listing assets within the Dandiset.
# 3. Loading a specific NWB (Neurodata Without Borders) file containing LFP data from the Dandiset using `pynwb`, `h5py`, and `remfile` for remote streaming.
# 4. Exploring basic metadata within the NWB file.
# 5. Examining the structure of the electrodes table (channel information and brain regions).
# 6. Loading and visualizing a short segment of LFP data from a few channels.

# %% [markdown]
# ## Required Packages
#
# This notebook requires the following Python packages. It is assumed they are already installed in your environment.
#
# *   `dandi` (for interacting with the DANDI Archive API)
# *   `pynwb` (for reading NWB files)
# *   `h5py` (NWB backend for HDF5 files)
# *   `remfile` (for streaming remote HDF5 files)
# *   `numpy` (for numerical operations)
# *   `pandas` (for data manipulation, especially the electrodes table)
# *   `matplotlib` (for plotting)
# *   `seaborn` (for enhanced plotting styles)

# %%
# Import necessary libraries
import os
import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dandi.dandiapi import DandiAPIClient

# Set plotting style
sns.set_theme()

# Set pandas display options for better table viewing
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Libraries imported.")

# %% [markdown]
# ## Connecting to DANDI and Loading Dandiset Info
#
# We can use the `dandi-cli` library to programmatically access information about the Dandiset.

# %%
# Connect to DANDI archive
try:
    client = DandiAPIClient()
    dandiset = client.get_dandiset("000690", "draft") # Specify version 'draft'

    # Print basic information about the Dandiset
    metadata = dandiset.get_raw_metadata()
    print(f"Dandiset name: {metadata.get('name', 'N/A')}")
    print(f"Dandiset URL: {metadata.get('url', 'https://dandiarchive.org/dandiset/000690/draft')}")
    print(f"Dandiset Description: {metadata.get('description', 'N/A')[:300]}...") # Print start of description

    # List the assets in the Dandiset (show first 5)
    print("\nListing assets...")
    assets = list(dandiset.get_assets())
    print(f"\nFound {len(assets)} assets in the dataset")
    print("\nFirst 5 assets:")
    for asset in assets[:5]:
        print(f"- Path: {asset.path}, Size: {asset.size / (1024**3):.2f} GB, ID: {asset.asset_id}")

except Exception as e:
    print(f"An error occurred while connecting to DANDI: {e}")


# %% [markdown]
# ## Loading a Specific NWB File
#
# This Dandiset contains multiple NWB files, often separated by subject, session, and data type (e.g., `_ecephys.nwb`, `_image.nwb`). The `_ecephys.nwb` files typically contain electrophysiology data like LFP or spikes, grouped by probe.
#
# For this demonstration, we will load an LFP data file from `probe-0` for subject `692072`, session `1298465622`.
#
# **File Path:** `sub-692072/sub-692072_ses-1298465622_probe-0_ecephys.nwb`
#
# **Asset ID:** `ba8760f9-91fe-4c1c-97e6-590bed6a783b`
#
# We will construct the DANDI API URL for this asset and use `remfile` along with `h5py` and `pynwb` to stream the data directly without downloading the entire (potentially large) file.

# %%
# Define the URL for the chosen NWB asset
nwb_asset_id = "ba8760f9-91fe-4c1c-97e6-590bed6a783b"
nwb_url = f"https://api.dandiarchive.org/api/assets/{nwb_asset_id}/download/"
print(f"NWB File URL: {nwb_url}")

# Initialize variables to None
remote_file = None
h5_file = None
io = None
nwb = None

try:
    print("\nOpening remote NWB file stream...")
    # Use remfile to open a stream to the remote file
    remote_file = remfile.File(nwb_url)
    # Open the HDF5 file stream using h5py
    h5_file = h5py.File(remote_file, 'r')
    # Use pynwb to read the NWB file structure
    io = pynwb.NWBHDF5IO(file=h5_file, mode='r', load_namespaces=True)
    nwb = io.read()
    print("NWB file loaded successfully.")
    # Display the basic NWB object representation
    print("\nNWB File Object:")
    print(nwb)

except Exception as e:
    print(f"\nAn error occurred while loading the NWB file: {e}")
    print("Please ensure the required libraries are installed and the URL is correct.")
    # Make sure to clean up if loading failed partially
    if io:
        io.close()
    elif h5_file:
        h5_file.close()
    elif remote_file:
        # remfile doesn't have an explicit close, relies on garbage collection
        pass


# %% [markdown]
# ### Explore NWB File on Neurosift
#
# You can interactively explore the structure and contents of this NWB file using Neurosift:
#
# [Explore on Neurosift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/&dandisetId=000690&dandisetVersion=draft)

# %% [markdown]
# ## Exploring NWB File Contents
#
# Now that the NWB file is loaded, we can explore its metadata and data structures. Based on the filename (`_probe-0_ecephys.nwb`) and preliminary checks, we expect this file to contain LFP data and associated electrode information for probe A.

# %%
# Print some basic metadata from the NWB file if loaded successfully
if nwb:
    print("--- Basic NWB File Metadata ---")
    print(f"Session Description: {getattr(nwb, 'session_description', 'N/A')}")
    print(f"Identifier: {getattr(nwb, 'identifier', 'N/A')}")
    print(f"Session Start Time: {getattr(nwb, 'session_start_time', 'N/A')}")
    print(f"Institution: {getattr(nwb, 'institution', 'N/A')}")
    print(f"Session ID: {getattr(nwb, 'session_id', 'N/A')}")

    print("\n--- Subject Information ---")
    if nwb.subject:
        print(f"Subject ID: {getattr(nwb.subject, 'subject_id', 'N/A')}")
        print(f"Age: {getattr(nwb.subject, 'age', 'N/A')}")
        print(f"Sex: {getattr(nwb.subject, 'sex', 'N/A')}")
        print(f"Species: {getattr(nwb.subject, 'species', 'N/A')}")
        print(f"Genotype: {getattr(nwb.subject, 'genotype', 'N/A')}")
    else:
        print("Subject information not available.")

    print("\n--- Available Data Structures ---")
    print("Acquisition Objects:", list(nwb.acquisition.keys()) if nwb.acquisition else "None")
    # Note: Processing modules, units etc. might be in other files or not present
    print("Processing Modules:", list(nwb.processing.keys()) if nwb.processing else "None")
    print("Units:", "Available" if nwb.units else "None") # Units are often in separate files
else:
    print("NWB file was not loaded successfully, skipping metadata exploration.")


# %% [markdown]
# ### Electrodes Table
#
# The `nwb.electrodes` table contains metadata about each recording channel, such as its location, impedance, and which probe it belongs to. Let's load it into a pandas DataFrame for easier viewing.

# %%
if nwb and nwb.electrodes:
    print("Loading electrodes table into pandas DataFrame...")
    try:
        electrodes_df = nwb.electrodes.to_dataframe()
        print("Electrodes table loaded.")

        print("\n--- Electrodes Table (First 5 Rows) ---")
        print(electrodes_df.head())

        # Print unique locations if the column exists
        print("\n--- Unique Brain Regions Recorded (Probe A) ---")
        if 'location' in electrodes_df.columns:
            # Filter for probeA if multiple probes were in the table (unlikely here)
            probe_a_locations = electrodes_df[electrodes_df['group_name'] == 'probeA']['location'].unique()
            print(probe_a_locations)
        else:
            print("'location' column not found in the table.")

    except Exception as e:
        print(f"An error occurred while processing the electrodes table: {e}")
else:
    print("NWB file not loaded or 'electrodes' table not found.")


# %% [markdown]
# ## Visualizing LFP Data
#
# This file contains Local Field Potential (LFP) data recorded from probe A. LFP reflects aggregate synaptic activity near the electrode. Let's access the `ElectricalSeries` containing the LFP data and plot a short segment for a few channels.
#
# The LFP data is typically found within `nwb.acquisition`. The path might vary slightly, but based on our earlier exploration, it's likely under `nwb.acquisition['probe_0_lfp'].electrical_series['probe_0_lfp_data']`.

# %%
if nwb and 'probe_0_lfp' in nwb.acquisition:
    print("Accessing LFP data...")
    try:
        # Access LFP ElectricalSeries
        lfp_electrical_series = nwb.acquisition['probe_0_lfp'].electrical_series['probe_0_lfp_data']
        lfp_data = lfp_electrical_series.data
        lfp_timestamps = lfp_electrical_series.timestamps
        print(f"LFP data shape: {lfp_data.shape}") # (time, channels)
        print(f"LFP timestamps shape: {lfp_timestamps.shape}")

        # Get sampling rate (important for selecting time segments)
        sampling_rate = None
        if hasattr(lfp_electrical_series, 'rate'):
            sampling_rate = lfp_electrical_series.rate
            print(f"Using LFP sampling rate from ElectricalSeries: {sampling_rate} Hz")
        elif 'probeA' in nwb.electrode_groups and hasattr(nwb.electrode_groups['probeA'], 'lfp_sampling_rate') and nwb.electrode_groups['probeA'].lfp_sampling_rate:
            sampling_rate = nwb.electrode_groups['probeA'].lfp_sampling_rate
            print(f"Using LFP sampling rate from ElectrodeGroup: {sampling_rate} Hz")
        else:
            # Estimate sampling rate from timestamps if possible
            if len(lfp_timestamps) > 1:
                 sampling_rate = 1.0 / (lfp_timestamps[1] - lfp_timestamps[0])
                 print(f"Estimated sampling rate from timestamps: {sampling_rate:.2f} Hz")

        if sampling_rate:
            # Define parameters for plotting
            num_channels_to_plot = 5
            start_time_s = 20.5  # Start time in seconds (chosen based on exploration)
            duration_s = 0.5    # Duration to plot in seconds

            # Calculate start and end indices
            start_index = int(start_time_s * sampling_rate)
            end_index = start_index + int(duration_s * sampling_rate)

            # Ensure indices are within bounds
            if start_index < 0: start_index = 0
            if end_index > len(lfp_timestamps): end_index = len(lfp_timestamps)
            if start_index >= end_index:
                 print("Warning: Calculated time range is invalid or too short.")

            print(f"\nLoading LFP data from {start_time_s:.2f}s to {start_time_s + duration_s:.2f}s for first {num_channels_to_plot} channels...")

            # Select the time segment for timestamps and data
            # Important: Load data segment directly using slicing for efficiency
            ts_segment = lfp_timestamps[start_index:end_index]
            data_segment = lfp_data[start_index:end_index, :num_channels_to_plot] # Slicing HDF5 dataset

            # Get actual channel IDs for the plotted channels from the DataFrame
            channel_ids = ['Unknown'] * num_channels_to_plot
            if 'electrodes_df' in locals():
                valid_indices = electrodes_df.index[:num_channels_to_plot]
                channel_ids = valid_indices.tolist()
                print(f"Plotting data for Channel IDs: {channel_ids}")


            # --- Plotting ---
            print("Generating plot...")
            plt.figure(figsize=(15, 6))

            # Offset traces for better visibility
            # Calculate offset based on std dev of the loaded segment
            if data_segment.size > 0 :
              offset_scale = 3 # Adjust this multiplier to control separation
              offset = np.std(data_segment) * offset_scale
              if offset == 0: offset = np.mean(np.abs(data_segment))*offset_scale*2 if np.mean(np.abs(data_segment)) > 0 else 1 # Handle zero std/mean case

              for i in range(data_segment.shape[1]):
                  plt.plot(ts_segment, data_segment[:, i] + i * offset, label=f'Channel {channel_ids[i]}')

              plt.title(f'LFP Data Segment ({duration_s}s, {num_channels_to_plot} Channels)')
              plt.xlabel('Time (s)')
              plt.ylabel(f'Voltage ({lfp_electrical_series.unit}) offsetted')
              plt.legend(loc='upper right')
              plt.grid(True)
              plt.show()
            else:
              print("No data loaded for the specified time range or channels.")

        else:
            print("Could not determine LFP sampling rate. Skipping LFP plot.")

    except KeyError as e:
        print(f"KeyError accessing LFP data: {e}. Structure might be different.")
    except Exception as e:
        print(f"An error occurred while accessing or plotting LFP data: {e}")
        import traceback
        traceback.print_exc()

else:
    print("NWB file not loaded or 'probe_0_lfp' not found in acquisition. Skipping LFP plot.")


# %% [markdown]
# ## Summary and Next Steps
#
# This notebook demonstrated how to:
# *   Connect to the DANDI Archive and retrieve metadata for Dandiset 000690.
# *   List assets within the Dandiset.
# *   Load a specific NWB file containing LFP data using remote streaming.
# *   Inspect basic metadata within the NWB file (session info, subject details).
# *   Examine the `electrodes` table to understand channel properties and recorded brain regions (including Thalamus, Hippocampus, Visual Cortex, RSP).
# *   Load and visualize a segment of LFP data, showing electrical activity traces over time.
#
# **Potential Next Steps:**
# *   **Explore other NWB files:** Analyze files from different subjects, sessions, or probes. Look for files potentially containing spike data (`Units` table) if available. There are also `_image.nwb` files which might contain visual stimulus information or imaging data not explored here.
# *   **Analyze different time segments:** Investigate LFP activity during specific visual stimulus presentations (if stimulus timing information is available in related files or metadata).
# *   **Frequency analysis:** Perform spectral analysis (e.g., using Welch's method) on the LFP data to examine power in different frequency bands (theta, gamma, etc.) across regions or conditions.
# *   **Cross-channel analysis:** Calculate coherence or correlation between LFP signals recorded on different channels or in different brain regions.
# *   **Relate to behavior/stimulus:** If stimulus tables or behavioral data are present (potentially in the main session file or `_image.nwb` file), align neural activity with specific events.
#
# Remember to close the NWB file connection when finished to release resources.

# %%
# Close the NWB file to release the stream
if io:
    print("Closing NWB file.")
    io.close()
    # Optionally, clear variables to save memory
    # nwb = None
    # h5_file = None
    # remote_file = None
    # io = None
print("Notebook execution finished.")