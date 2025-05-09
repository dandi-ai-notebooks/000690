# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
#
# **Dandiset Version: 0.250326.0015**

# %% [markdown]
# > ⚠️ **Disclaimer:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This Dandiset, titled "Allen Institute Openscope - Vision2Hippocampus project," contains data related to an experiment investigating how neural representations of visual stimuli evolve from the thalamus through visual cortex to the hippocampus in mice.
#
# The project aims to understand how the brain computes abstractions from simple visual stimuli to more complex representations. Mice were presented with simple visual motion (bars of light with varying properties) and complex visual stimuli (movies).
#
# **Key information:**
# - **Dandiset Identifier:** 000690
# - **Version:** 0.250326.0015
# - **Dandiset URL:** [https://dandiarchive.org/dandiset/000690/0.250326.0015](https://dandiarchive.org/dandiset/000690/0.250326.0015)
# - **Description:** Extensive research shows that visual cortical neurons respond to specific stimuli, e.g. the primary visual cortical neurons respond to bars of light with specific orientation. In contrast, the hippocampal neurons are thought to encode not specific stimuli but instead represent abstract concepts such as space, time and events. How is this abstraction computed in the mouse brain? Specifically, how does the representation of simple visual stimuli evolve from the thalamus, which is a synapse away from the retina, through primary visual cortex, higher order visual areas and all the way to hippocampus, that is farthest removed from the retina? The current OpenScope project aims to understand how the neural representations of simple and natural stimuli evolve from the LGN through V1, and most hippocampal regions, as well as some of the frontal areas.
# - **Keywords:** mouse, neuropixel, extracellular electrophysiology, neocortex, hippocampus, Entorhinal cortex, excitatory, inhibitory, vision, movies.
# - **Measurement Techniques:** multi electrode extracellular electrophysiology recording technique, signal filtering technique, spike sorting technique, analytical technique.

# %% [markdown]
# ## What this notebook covers
#
# This notebook will guide you through the following steps:
# 1. Listing the required Python packages.
# 2. Demonstrating how to load the Dandiset metadata using the DANDI API.
# 3. Showing how to load a specific Neurophysiology: Neurodata Without Borders (NWB) file from the Dandiset.
# 4. Displaying some metadata from the loaded NWB file.
# 5. Summarizing the contents of the NWB file.
# 6. Providing a link to explore the NWB file interactively using Neurosift.
# 7. Illustrating how to load and visualize some example data from the NWB file, such as eye tracking and running speed.
# 8. Summarizing the findings and suggesting potential future analysis directions.

# %% [markdown]
# ## Required Packages
#
# To run this notebook, you'll need the following Python packages. We assume these are already installed on your system.
#
# - `dandi` (for interacting with the DANDI Archive)
# - `pynwb` (for working with NWB files)
# - `h5py` (for HDF5 file access, a dependency of pynwb)
# - `remfile` (for accessing remote HDF5 files)
# - `numpy` (for numerical operations)
# - `matplotlib` (for plotting)
# - `pandas` (for data manipulation and viewing tables)
# - `seaborn` (for enhanced visualizations)

# %% [markdown]
# ## Loading the Dandiset using the DANDI API

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pynwb
import h5py
import remfile
import seaborn as sns

# Set seaborn style for plots (except for images)
sns.set_theme()

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset_id = "000690"
dandiset_version = "0.250326.0015"
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# %%
# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Dandiset description: {metadata.get('description', 'N/A')[:200]}...") # Print first 200 characters

# %%
# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.asset_id})") # Using asset_id as identifier

# %% [markdown]
# ## Loading an NWB File
#
# We will now load one of the NWB files from the Dandiset to explore its contents.
#
# For this demonstration, we will load the file: `sub-692072/sub-692072_ses-1298465622.nwb`.
#
# The asset ID for this file is `fbcd4fe5-7107-41b2-b154-b67f783f23dc`.
#
# We can construct the direct download URL for this asset as follows:
# `https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/`

# %%
# Define the URL for the NWB file
nwb_asset_id = "fbcd4fe5-7107-41b2-b154-b67f783f23dc"
nwb_file_url = f"https://api.dandiarchive.org/api/assets/{nwb_asset_id}/download/"
print(f"Loading NWB file from: {nwb_file_url}")

# Load the NWB file
# This uses remfile to stream the remote HDF5 file
remote_file = remfile.File(nwb_file_url)
h5_file = h5py.File(remote_file, 'r') # Ensure read-only mode
io = pynwb.NWBHDF5IO(file=h5_file, mode='r') # Ensure read-only mode for pynwb
nwbfile = io.read()

print("\nNWB file loaded successfully.")

# %% [markdown]
# ### NWB File Metadata
#
# Let's look at some basic metadata from the loaded NWB file.

# %%
print(f"Session Description: {nwbfile.session_description}")
print(f"Identifier: {nwbfile.identifier}")
print(f"Session Start Time: {nwbfile.session_start_time}")
if nwbfile.subject:
    print(f"Subject ID: {nwbfile.subject.subject_id}")
    print(f"Subject Age: {nwbfile.subject.age}")
    print(f"Subject Sex: {nwbfile.subject.sex}")
    print(f"Subject Species: {nwbfile.subject.species}")
else:
    print("Subject information: Not available")

# %% [markdown]
# ### Neurosift Link
#
# You can explore this NWB file interactively on Neurosift using the following link:
#
# [https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=draft](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=draft)
#
# *(Note: The Neurosift link uses `dandisetVersion=draft` as specific version linking might vary in behavior or availability)*

# %% [markdown]
# ### Summarizing NWB File Contents
#
# NWB files have a hierarchical structure. Let's explore some of the main groups and datasets within this file.

# %% [markdown]
# #### Acquisition Data
#
# This group typically contains raw acquired data.

# %%
if nwbfile.acquisition:
    print("Available data in 'acquisition':")
    for acq_name, acq_data in nwbfile.acquisition.items():
        print(f"- {acq_name} (type: {type(acq_data).__name__})")
        if hasattr(acq_data, 'data') and hasattr(acq_data.data, 'shape'):
             print(f"  Shape: {acq_data.data.shape}")
        if isinstance(acq_data, pynwb.base.TimeSeries) and hasattr(acq_data, 'timestamps') and hasattr(acq_data.timestamps, 'shape'):
             print(f"  Timestamps shape: {acq_data.timestamps.shape}")
        elif isinstance(acq_data, pynwb.behavior.Position): # For EyeTracking
            for spatial_series_name, spatial_series in acq_data.spatial_series.items():
                print(f"  - {spatial_series_name} (type: {type(spatial_series).__name__})")
                if hasattr(spatial_series, 'data') and hasattr(spatial_series.data, 'shape'):
                    print(f"    Data shape: {spatial_series.data.shape}")
                if hasattr(spatial_series, 'timestamps') and_link = hasattr(spatial_series.timestamps, 'shape'):  # Corrected: and_link to and
                    print(f"    Timestamps shape: {spatial_series.timestamps.shape if hasattr(spatial_series.timestamps, 'shape') else 'N/A'}")

else:
    print("'acquisition' group is empty or not present.")

# %% [markdown]
# #### Processing Data
#
# This group often contains processed data derived from the raw acquisition.

# %%
if nwbfile.processing:
    print("\nAvailable data in 'processing':")
    for proc_module_name, proc_module in nwbfile.processing.items():
        print(f"- Processing module: {proc_module_name}")
        for data_interface_name, data_interface in proc_module.data_interfaces.items():
            print(f"  - {data_interface_name} (type: {type(data_interface).__name__})")
            if hasattr(data_interface, 'data') and hasattr(data_interface.data, 'shape'):
                 print(f"    Shape: {data_interface.data.shape}")
            if isinstance(data_interface, pynwb.base.TimeSeries) and hasattr(data_interface, 'timestamps') and hasattr(data_interface.timestamps, 'shape'):
                 print(f"    Timestamps shape: {data_interface.timestamps.shape}")
else:
    print("\n'processing' group is empty or not present.")

# %% [markdown]
# #### Units Data (Spike Times)
#
# If spike sorting has been performed, this table contains information about putative neurons (units), including their spike times.

# %%
if nwbfile.units:
    print("\nUnits data (spike times):")
    # Displaying units as a pandas DataFrame can be very informative
    # We select a few representative columns to display.
    # Note: Accessing all columns with to_dataframe() might be slow for large unit tables.
    # It's better to select columns of interest.
    desired_columns = ['id', 'spike_times', 'electrodes', 'electrode_group', 'waveform_mean', 'quality']
    # Filter out columns that don't exist to avoid errors
    available_columns = [col for col in desired_columns if col in nwbfile.units.colnames]

    if available_columns:
        units_df = nwbfile.units[available_columns].to_dataframe()
        print(f"Number of units: {len(units_df)}")
        print("First 5 units (selected columns):")
        # For 'spike_times' and 'waveform_mean', which are ragged arrays,
        # the DataFrame will contain object arrays. We can show their shapes or a snippet.
        
        # Create a display DataFrame for better printing
        display_df = pd.DataFrame()
        for col in available_columns:
            if col == 'spike_times':
                display_df[col] = units_df[col].apply(lambda x: f"shape {x.shape}" if hasattr(x, 'shape') else x)
            elif col == 'waveform_mean':
                 display_df[col] = units_df[col].apply(lambda x: f"shape {x.shape}" if hasattr(x, 'shape') else x)
            elif col == 'electrodes': # Electrodes is a DynamicTableRegion, show indices
                display_df[col] = units_df[col].apply(lambda x: x[:] if hasattr(x, '__getitem__') else x)
            else:
                display_df[col] = units_df[col]
        print(display_df.head())
    else:
        print("Units table is present but no standard columns found for display.")

else:
    print("\n'units' data is not present in this file.")

# %% [markdown]
# #### Stimulus Presentations
#
# Information about presented stimuli is often stored in `nwbfile.intervals`.

# %%
if nwbfile.intervals:
    print("\nAvailable stimulus presentation intervals:")
    for interval_name, interval_data in nwbfile.intervals.items():
        print(f"- {interval_name} (type: {type(interval_data).__name__})")
        # Displaying a few rows of each interval table using pandas
        try:
            interval_df = interval_data.to_dataframe()
            print(f"  Number of intervals: {len(interval_df)}")
            print(f"  Columns: {list(interval_df.columns)}")
            print("  First 2 entries:")
            print(interval_df.head(2))
        except Exception as e:
            print(f"  Could not convert {interval_name} to DataFrame: {e}")
else:
    print("\n'intervals' data is not present.")

# %% [markdown]
# ## Visualizing Data from the NWB File
#
# Let's visualize some of the data. We'll be careful to load only subsets of large datasets to avoid excessive download times and memory usage.

# %% [markdown]
# ### Example 1: Pupil Area
#
# The file contains eye tracking data, including pupil area. Let's plot the pupil area over a short period.

# %%
if 'EyeTracking' in nwbfile.acquisition and \
   'pupil_tracking' in nwbfile.acquisition['EyeTracking'].spatial_series:
    pupil_tracking = nwbfile.acquisition['EyeTracking'].spatial_series['pupil_tracking']

    # Check if area and timestamps are available
    if hasattr(pupil_tracking, 'area') and pupil_tracking.area is not None and \
       hasattr(pupil_tracking, 'timestamps') and pupil_tracking.timestamps is not None:

        print(f"Pupil area data shape: {pupil_tracking.area.shape}")
        print(f"Pupil timestamps shape: {pupil_tracking.timestamps.shape}")

        # Determine a subset of data to plot (e.g., first 1000 points or first 10 seconds)
        num_points_to_plot = min(1000, len(pupil_tracking.timestamps))
        
        # Find the index for the first ~10 seconds of data if sampling rate is known
        # Assuming timestamps are in seconds.
        # If timestamps are regularly sampled, we can estimate. Otherwise, find index by time.
        # For simplicity, we plot the first num_points_to_plot
        
        time_slice = slice(0, num_points_to_plot)
        
        try:
            pupil_area_data = pupil_tracking.area[time_slice]
            timestamps_data = pupil_tracking.timestamps[time_slice]

            if timestamps_data.ndim > 1: # If timestamps are in intervals (start, stop)
                 timestamps_data = timestamps_data[:, 0] # Use start times

            plt.figure(figsize=(12, 6))
            plt.plot(timestamps_data, pupil_area_data)
            plt.xlabel(f"Time ({pupil_tracking.timestamps_unit})")
            plt.ylabel(f"Pupil Area ({pupil_tracking.unit if hasattr(pupil_tracking, 'unit') else 'unknown units'})")
            plt.title(f"Pupil Area (First {num_points_to_plot} samples)")
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f"Could not plot pupil area: {e}")
            print("This might be due to the data being very large or an issue with accessing a slice.")
            print("Consider checking the structure of pupil_tracking.area and pupil_tracking.timestamps in more detail.")
            print(f"pupil_tracking.area type: {type(pupil_tracking.area)}")
            print(f"pupil_tracking.timestamps type: {type(pupil_tracking.timestamps)}")

    else:
        print("Pupil area or timestamps data not available in 'EyeTracking/pupil_tracking'.")
else:
    print("'EyeTracking' or 'pupil_tracking' not found in acquisition data.")

# %% [markdown]
# ### Example 2: Running Speed
#
# The file contains running speed data in the `processing` module. Let's plot this.

# %%
if 'running' in nwbfile.processing and \
   'running_speed' in nwbfile.processing['running'].data_interfaces:
    running_speed_ts = nwbfile.processing['running'].data_interfaces['running_speed']

    if hasattr(running_speed_ts, 'data') and running_speed_ts.data is not None and \
       hasattr(running_speed_ts, 'timestamps') and running_speed_ts.timestamps is not None:

        print(f"Running speed data shape: {running_speed_ts.data.shape}")
        print(f"Running speed timestamps shape: {running_speed_ts.timestamps.shape}")

        # Determine a subset of data to plot
        num_points_to_plot = min(2000, len(running_speed_ts.timestamps))
        time_slice = slice(0, num_points_to_plot)

        try:
            running_speed_data = running_speed_ts.data[time_slice]
            timestamps_data = running_speed_ts.timestamps[time_slice]

            plt.figure(figsize=(12, 6))
            plt.plot(timestamps_data, running_speed_data)
            plt.xlabel(f"Time ({running_speed_ts.timestamps_unit})")
            plt.ylabel(f"Running Speed ({running_speed_ts.unit})")
            plt.title(f"Running Speed (First {num_points_to_plot} samples)")
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f"Could not plot running speed: {e}")
            print("This might be due to the data being very large or an issue with accessing a slice.")
    else:
        print("Running speed data or timestamps not available.")
else:
    print("'running/running_speed' not found in processing data.")

# %% [markdown]
# ### Example 3: Unit Spike Times (Raster Plot for a few units)
#
# If unit data is available, we can create a simple raster plot for a small number of units over a short time window.
# A full raster plot for all units and the entire session can be very dense and computationally intensive.

# %%
if nwbfile.units:
    print("\nAttempting to create a raster plot for a few units...")
    units_df = nwbfile.units.to_dataframe() # Get the full dataframe to access spike times

    num_units_to_plot = min(10, len(units_df)) # Plot up to 10 units
    plot_duration_seconds = 10 # Plot spikes within the first 10 seconds of the recording

    plt.figure(figsize=(12, 6))
    
    event_times_list = []
    unit_ids_for_plot = []

    for i in range(num_units_to_plot):
        unit_id = units_df.index[i] # This assumes default integer index; or use 'id' column if exists
        if 'id' in units_df.columns: # prefer actual unit IDs if available
            unit_id_val = units_df['id'].iloc[i]
        else:
            unit_id_val = unit_id

        spike_times_all = nwbfile.units['spike_times'][i] # Access spike times by index is more direct
        
        # Filter spike times for the desired duration
        spike_times_in_window = spike_times_all[spike_times_all < plot_duration_seconds]
        
        if len(spike_times_in_window) > 0:
            event_times_list.append(spike_times_in_window)
            unit_ids_for_plot.append(unit_id_val)

    if event_times_list:
        colors = plt.cm.viridis(np.linspace(0, 1, len(event_times_list)))
        plt.eventplot(event_times_list, linelengths=0.75, colors=colors)
        plt.yticks(np.arange(len(unit_ids_for_plot)), unit_ids_for_plot)
        plt.xlabel("Time (s)")
        plt.ylabel("Unit ID")
        plt.title(f"Spike Raster (First {num_units_to_plot} Units, First {plot_duration_seconds} s)")
        plt.grid(True, axis='x')
        plt.show()
    else:
        print("No spikes found in the selected time window for the chosen units, or no units to plot.")
else:
    print("No units data available for raster plot.")

# %% [markdown]
# ## Summary of Findings and Future Directions
#
# This notebook demonstrated how to:
# - Access and understand metadata for Dandiset 000690.
# - Load an NWB file from this Dandiset using its URL.
# - Explore the basic structure and metadata within the NWB file, including acquisition, processing, units, and stimulus interval data.
# - Visualize example data streams such as pupil area, running speed, and a simple spike raster.
#
# ### Potential Future Directions:
#
# 1.  **Detailed Stimulus-Response Analysis:** Correlate neural activity (spike times from `nwbfile.units`) with specific stimulus presentation times found in `nwbfile.intervals`. This could involve creating peri-stimulus time histograms (PSTHs) for different visual stimuli.
2.  **Behavioral Correlations:** Investigate how behavioral variables like running speed or pupil diameter modulate neural responses. For example, analyze if firing rates of certain units change when the animal is running versus stationary during visual stimulation.
3.  **Cross-Probe Analysis:** If the Dandiset contains data from multiple ephys probes (as suggested by `probe-X_ecephys.nwb` asset names, though this specific file `sub-692072_ses-1298465622.nwb` is a general session file), one could explore correlations or differential responses across brain regions targeted by these probes. This would involve loading the corresponding `_ecephys.nwb` files.
4.  **LFP Analysis:** The `electrode_groups` metadata mentions LFP (Local Field Potential) data. If LFP data is present (often in `nwbfile.acquisition` or linked `_ecephys.nwb` files), analyses such as power spectral density during different behavioral states or stimulus conditions could be performed.
5.  **Exploring Different NWB Files:** This notebook focused on one main session NWB file. The Dandiset contains other files, including `_image.nwb` (likely containing imaging data if applicable, though this Dandiset focuses on ephys) and `_probe-X_ecephys.nwb` files which typically contain the raw continuous electrophysiology data and LFP. Exploring these specialized files would provide deeper insights into specific data modalities.
6.  **Advanced Visualizations:** Create more sophisticated visualizations, such as heatmaps of neural activity across populations, or plots showing the relationship between multiple behavioral variables.
7.  **Comparative Analysis:** Compare neural responses to different types of visual stimuli (simple bars vs. movies) described in the Dandiset's protocol.
#
# Remember that working with remote NWB files requires careful consideration of data sizes. Always try to load only necessary subsets of data for initial exploration and visualization to manage download times and memory usage effectively. Refer to the `tools_cli.py nwb-file-info` output for detailed paths to specific datasets within the NWB file structure.

# %% [markdown]
# ---
# End of Notebook.
#
# To close the NWB file and release the remote file object if you were to run this interactively and wanted to manage resources explicitly (though not strictly necessary for a script that terminates):
# ```python
# # io.close()
# # remote_file.close() # If remfile has a close method
# # h5_file.close()
# ```
# However, for this automated notebook execution, explicit closing is handled upon script completion.

# %%
print("Notebook execution cell - final check.")