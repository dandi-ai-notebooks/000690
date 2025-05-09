# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
#
# **Note:** This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset
#
# This notebook explores Dandiset [000690/0.250326.0015](https://dandiarchive.org/dandiset/000690/0.250326.0015), titled "Allen Institute Openscope - Vision2Hippocampus project".
#
# **Description from DANDI:**
#
# Extextensive research shows that visual cortical neurons respond to specific stimuli, e.g. the primary visual cortical neurons respond to bars of light with specific orientation. In contrast, the hippocampal neurons are thought to encode not specific stimuli but instead represent abstract concepts such as space, time and events. How is this abstraction computed in the mouse brain? Specifically, how does the representation of simple visual stimuli evolve from the thalamus, which is a synapse away from the retina, through primary visual cortex, higher order visual areas and all the way to hippocampus, that is farthest removed from the retina?
#
# The current OpenScope project aims to understand how the neural representations of simple and natural stimuli evolve from the LGN through V1, and most hippocampal regions, as well as some of the frontal areas.
#
# **Stimuli presented:**
# Two main categories of visual stimuli were presented–
# 1. Simple visual motion, elicited by basic stimuli, like bars of light.
# 2. Complex, potentially ethologically valid visual stimuli, elicited by movies involving eagles (and squirrels).
#
# This Dandiset contains data related to extracellular electrophysiology recordings in mice, focusing on how visual stimuli are processed across different brain regions, including the neocortex, hippocampus, and entorhinal cortex. It includes measurements of electrical series, LFP, processing modules, and units.
#
# **Keywords:** mouse, neuropixel, extracellular electrophysiology, neocortex, hippocampus, Entorhinal cortex, excitatory, inhibitory, vision, movies.

# %% [markdown]
# ## What this notebook covers
#
# This notebook will guide you through:
# 1. Listing the required Python packages.
# 2. Connecting to the DANDI archive and loading the Dandiset metadata.
# 3. Listing some assets (files) within the Dandiset.
# 4. Loading a specific NWB (Neurodata Without Borders) file from the Dandiset.
# 5. Exploring the metadata and structure of the loaded NWB file.
# 6. Loading and visualizing some example data series from the NWB file, such as running speed and pupil area.
# 7. Summarizing findings and suggesting potential future analysis directions.

# %% [markdown]
# ## Required Packages
#
# To run this notebook, you'll need the following Python packages installed. This notebook assumes they are already present in your environment.
#
# * `dandi` (for interacting with the DANDI Archive)
# * `pynwb` (for reading NWB files)
# * `h5py` (dependency for pynwb for HDF5 file access)
# * `remfile` (for streaming remote HDF5 files)
# * `numpy` (for numerical operations)
# * `matplotlib` (for plotting)
# * `pandas` (for data manipulation, especially with NWB interval data)
# * `itertools` (used for slicing iterators)
# * `seaborn` (for enhanced visualizations)

# %%
# Import necessary packages
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Apply seaborn styling for plots (except images)
sns.set_theme()

# %% [markdown]
# ## Loading the Dandiset using DANDI API

# %%
# Connect to DANDI archive
client = DandiAPIClient()
dandiset_id = "000690"
dandiset_version = "0.250326.0015"
dandiset = client.get_dandiset(dandiset_id, dandiset_version)

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}") # Note: This URL might point to the base dandiset, not the specific version.
print(f"Dandiset description: {metadata.get('description', 'N/A')[:200]}...") # Print first 200 chars of description

# List some assets in the Dandiset
assets = dandiset.get_assets()
print("\nFirst 5 assets:")
for asset in islice(assets, 5):
    print(f"- {asset.path} (ID: {asset.identifier})")

# %% [markdown]
# ## Loading an NWB file from the Dandiset
#
# We will load one of the NWB files to explore its contents. For this demonstration, we'll use the file:
# `sub-692072/sub-692072_ses-1298465622.nwb`
#
# Its asset ID is `fbcd4fe5-7107-41b2-b154-b67f783f23dc`.
#
# The URL for downloading this asset is:
# `https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/`
#
# We will use `remfile` and `h5py` to stream the NWB file directly from the DANDI archive without downloading it locally.
#
# **Note:** Loading large NWB files, especially over a network connection, can be time-consuming. Please be patient.

# %%
# Hardcoded URL for the specific NWB file
nwb_asset_id = "fbcd4fe5-7107-41b2-b154-b67f783f23dc"
nwb_file_url = f"https://api.dandiarchive.org/api/assets/{nwb_asset_id}/download/"
nwb_file_path_in_dandiset = "sub-692072/sub-692072_ses-1298465622.nwb"

print(f"Loading NWB file: {nwb_file_path_in_dandiset}")
print(f"From URL: {nwb_file_url}")

# Load the NWB file
# This uses the exact code structure provided by `tools_cli.py nwb-file-info`
remote_nwb_file = remfile.File(nwb_file_url)
h5_nwb_file = h5py.File(remote_nwb_file, 'r') # Ensure read-only mode
nwb_io = pynwb.NWBHDF5IO(file=h5_nwb_file, mode='r', load_namespaces=True) # Ensure read-only, load namespaces
nwbfile = nwb_io.read()

print("\nNWB file loaded successfully.")

# %% [markdown]
# ### NWB File Metadata
# Let's look at some basic metadata from the loaded NWB file.

# %%
print(f"NWBFile Identifier: {nwbfile.identifier}")
print(f"Session Description: {nwbfile.session_description}")
print(f"Session Start Time: {nwbfile.session_start_time}")
print(f"Experimenter(s): {nwbfile.experimenter}")
print(f"Institution: {nwbfile.institution}")
print(f"Lab: {nwbfile.lab}")
print(f"Subject ID: {nwbfile.subject.subject_id if nwbfile.subject else 'N/A'}")
print(f"Subject Age: {nwbfile.subject.age if nwbfile.subject else 'N/A'}")
print(f"Subject Species: {nwbfile.subject.species if nwbfile.subject else 'N/A'}")

# %% [markdown]
# ### Neurosift Link
#
# You can explore this NWB file interactively using Neurosift:
# [Open in Neurosift](https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=0.250326.0015)
#
# (Note: The `dandisetVersion` in the Neurosift link is set to `0.250326.0015` which is the version we are exploring. The asset itself is version-agnostic within the Dandiset version context for download purposes.)

# %% [markdown]
# ### Summarizing NWB File Contents
#
# NWB files are structured hierarchically. Here's a brief overview of some top-level groups found in this NWB file:
#
# *   **`acquisition`**: Contains raw acquired data that has not been processed.
#     *   `EyeTracking`: Data related to eye movements, pupil size, etc.
#         * `corneal_reflection_tracking`: EllipseSeries for corneal reflection.
#         * `eye_tracking`: EllipseSeries for eye position.
#         * `pupil_tracking`: EllipseSeries for pupil.
#         * `likely_blink`: TimeSeries for blink detection.
#     *   `raw_running_wheel_rotation`: TimeSeries for raw wheel rotation.
#     *   `running_wheel_signal_voltage`: TimeSeries for wheel signal voltage.
#     *   `running_wheel_supply_voltage`: TimeSeries for wheel supply voltage.
# *   **`processing`**: Contains processed data. This NWB file contains:
#     *   `running`: Processing module for running speed.
#         * `running_speed`: TimeSeries for computed running speed.
#     *   `stimulus`: Processing module for stimulus presentation times.
#         *   `timestamps`: TimeSeries for stimulus frame timestamps.
# *   **`intervals`**: Contains time intervals for various experimental epochs or events.
#     *   E.g., `Disco2SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations`, `Disk_Wd15_Vel2_Bndry1_Cntst0_loop_presentations`, etc. These tables contain detailed information about stimulus presentations.
# *   **`units`**: Contains information about sorted spike units, including spike times and waveforms.
# *   **`electrodes`**: A table describing the properties of each electrode.
# *   **`electrode_groups`**: Describes groups of electrodes (e.g., probes).
# *   **`devices`**: Information about data acquisition devices (e.g., Neuropixels probes).
# *   **`subject`**: Information about the experimental subject.
#
# To see all available data, you can explore the `nwbfile` object interactively or refer to the Neurosift link. For example, to list all items in `acquisition`:
# `print(list(nwbfile.acquisition.keys()))`
# And to list all processing modules:
# `print(list(nwbfile.processing.keys()))`

# %%
print("Items in nwbfile.acquisition:")
print(list(nwbfile.acquisition.keys()))

print("\nProcessing modules in nwbfile.processing:")
print(list(nwbfile.processing.keys()))

if "running" in nwbfile.processing:
    print("\nData interfaces in nwbfile.processing['running']:")
    print(list(nwbfile.processing["running"].data_interfaces.keys()))

if "EyeTracking" in nwbfile.acquisition:
    print("\nSpatial series in nwbfile.acquisition['EyeTracking']:")
    print(list(nwbfile.acquisition["EyeTracking"].spatial_series.keys()))

# %% [markdown]
# ## Loading and Visualizing Data from the NWB File
#
# Let's load and visualize a few examples of data. We'll be careful to load only subsets of data if they are very large, given that we are streaming over the network.

# %% [markdown]
# ### Example 1: Running Speed
#
# The running speed is stored in `nwbfile.processing['running']['running_speed']`. This is a `TimeSeries` object. We can access its data and timestamps.

# %%
if "running" in nwbfile.processing and "running_speed" in nwbfile.processing["running"].data_interfaces:
    running_speed_ts = nwbfile.processing["running"]["running_speed"]

    # Load a subset of data for plotting (e.g., first 1000 points)
    num_points_to_plot = 1000
    if len(running_speed_ts.data) > num_points_to_plot:
        print(f"Plotting the first {num_points_to_plot} points of running speed data.")
        running_speed_data = running_speed_ts.data[:num_points_to_plot]
        running_speed_timestamps = running_speed_ts.timestamps[:num_points_to_plot]
    else:
        print("Plotting all running speed data points.")
        running_speed_data = running_speed_ts.data[:]
        running_speed_timestamps = running_speed_ts.timestamps[:]

    plt.figure(figsize=(12, 4))
    plt.plot(running_speed_timestamps, running_speed_data)
    plt.xlabel(f"Time ({running_speed_ts.timestamps_unit})")
    plt.ylabel(f"Running Speed ({running_speed_ts.unit})")
    plt.title(f"Running Speed (First {len(running_speed_data)} points)")
    plt.show()
else:
    print("Running speed data not found in the expected location.")

# %% [markdown]
# ### Example 2: Pupil Area
#
# Pupil tracking data is often stored within `acquisition` under an `EyeTracking` module. Let's try to plot the pupil area from `nwbfile.acquisition['EyeTracking'].spatial_series['pupil_tracking']`. This is an `EllipseSeries` which has an `area` field.

# %%
if "EyeTracking" in nwbfile.acquisition and \
   "pupil_tracking" in nwbfile.acquisition["EyeTracking"].spatial_series:

    pupil_tracking_series = nwbfile.acquisition["EyeTracking"].spatial_series["pupil_tracking"]

    if hasattr(pupil_tracking_series, "area") and hasattr(pupil_tracking_series, "timestamps"):
        # Load a subset of data for plotting (e.g., first 1000 points or as available)
        num_points_to_plot_pupil = 1000
        
        # Ensure timestamps are loaded correctly. pupil_tracking_series.timestamps might be a dataset or another series
        if hasattr(pupil_tracking_series.timestamps, 'data'): # If timestamps is a TimeSeries itself
             timestamps_data_pupil = pupil_tracking_series.timestamps.data
        else: # If timestamps is a direct dataset
             timestamps_data_pupil = pupil_tracking_series.timestamps

        # Check lengths before slicing
        available_points_pupil = min(len(pupil_tracking_series.area), len(timestamps_data_pupil))

        if available_points_pupil > num_points_to_plot_pupil:
            print(f"Plotting the first {num_points_to_plot_pupil} points of pupil area data.")
            pupil_area_data = pupil_tracking_series.area[:num_points_to_plot_pupil]
            pupil_timestamps = timestamps_data_pupil[:num_points_to_plot_pupil]
        elif available_points_pupil > 0:
            print(f"Plotting all {available_points_pupil} available pupil area data points.")
            pupil_area_data = pupil_tracking_series.area[:available_points_pupil]
            pupil_timestamps = timestamps_data_pupil[:available_points_pupil]
        else:
            print("No pupil area data points available to plot.")
            pupil_area_data = [] # ensure it's an empty list

        if len(pupil_area_data) > 0:
            plt.figure(figsize=(12, 4))
            plt.plot(pupil_timestamps, pupil_area_data)
            plt.xlabel(f"Time ({pupil_tracking_series.timestamps_unit if hasattr(pupil_tracking_series, 'timestamps_unit') else 'seconds'})")
            plt.ylabel(f"Pupil Area (units: {pupil_tracking_series.unit})") # EllipseSeries might not have a specific unit for area
            plt.title(f"Pupil Area (First {len(pupil_area_data)} points)")
            plt.show()
    else:
        print("Pupil area or timestamps data not found in the pupil_tracking series.")
else:
    print("EyeTracking or pupil_tracking data not found in the expected location.")

# %% [markdown]
# ### Example 3: Spike Times from Units
#
# The `nwbfile.units` is a table containing information about sorted neuronal units. Each row typically represents a unit, and there's a `spike_times` column which is a ragged array (each unit has its own array of spike times).
#
# Let's get the spike times for the first few units and create a simple raster plot for a short time window.
# Accessing spike data directly can be more efficient than converting the entire units table to a DataFrame if it's very large.

# %%
if nwbfile.units is not None and 'spike_times' in nwbfile.units.colnames and 'id' in nwbfile.units.colnames:
    num_total_units = len(nwbfile.units.id[:]) # Get total number of units
    num_units_to_plot = min(3, num_total_units) # Plot first 3 units or fewer if not available
    time_window_plot = (0, 5) # Plot spikes in the first 5 seconds

    if num_units_to_plot > 0:
        print(f"Attempting to plot spike raster for the first {num_units_to_plot} units.")
        plt.figure(figsize=(12, 6))
        event_data = []
        ytick_labels = []
        
        try:
            # Get all unit IDs at once to minimize repeated access if possible
            all_unit_ids = nwbfile.units.id[:]

            for i in range(num_units_to_plot):
                # It's often more direct to get spike_times for unit i using nwbfile.units['spike_times'][i]
                # or through VectorIndex: nwbfile.units['spike_times'].data[i]
                # The exact access method for ragged array data can vary slightly with PyNWB versions or file specifics.
                # Let's try indexing the VectorIndex directly.
                spike_times_for_unit = nwbfile.units['spike_times'][i]
                unit_id = all_unit_ids[i]
                ytick_labels.append(f"Unit {unit_id}") # G for original ID from table
                
                # Filter spike times for the plotting window
                spikes_in_window = spike_times_for_unit[
                    (spike_times_for_unit >= time_window_plot[0]) & (spike_times_for_unit <= time_window_plot[1])
                ]
                event_data.append(spikes_in_window)
            
            # Seaborn's eventplot is good for rasters
            # Reset default matplotlib style for this plot if seaborn theme affects it undesirably
            with plt.style.context('default'):
                 colors = plt.cm.viridis(np.linspace(0, 1, num_units_to_plot))
                 plt.eventplot(event_data, linelengths=0.75, colors=colors)
            
            plt.yticks(ticks=np.arange(num_units_to_plot), labels=ytick_labels)
            plt.xlabel("Time (seconds)")
            plt.ylabel("Unit ID")
            plt.title(f"Spike Raster for First {num_units_to_plot} Units ({time_window_plot[0]}-{time_window_plot[1]} s)")
            plt.show()

        except Exception as e:
            print(f"Could not generate spike raster plot: {e}")
            print("This might be due to the size of the units table or specific data access patterns.")
            print("Consider exploring nwbfile.units directly or using Neurosift for large datasets.")
            
    else:
        print("No units found to plot or 'id' column missing from units table.")
else:
    print("Units data, spike_times, or id column not found.")

# %% [markdown]
# ### Example 4: Stimulus Presentation Intervals
#
# NWB files often store information about stimulus presentations or experimental trials in `TimeIntervals` tables within `nwbfile.intervals`. Let's inspect one such table. The output of `nwb-file-info` showed many stimulus presentation tables like `Disk_Wd15_Vel2_Bndry1_Cntst0_loop_presentations`.

# %%
if nwbfile.intervals is not None:
    # Let's pick one of the interval tables mentioned in the nwb-file-info output
    # For example: 'Disk_Wd15_Vel2_Bndry1_Cntst0_loop_presentations'
    interval_table_name = 'Disk_Wd15_Vel2_Bndry1_Cntst0_loop_presentations'
    if interval_table_name in nwbfile.intervals:
        stim_table = nwbfile.intervals[interval_table_name]
        # Convert to pandas DataFrame for easy viewing
        stim_df = stim_table.to_dataframe()
        print(f"\nFirst 5 rows of stimulus interval table: '{interval_table_name}'")
        # Use display for richer output if in a full Jupyter environment, otherwise print
        try:
            from IPython.display import display
            display(stim_df.head())
        except ImportError:
            print(stim_df.iloc[:5])

        # We could, for example, plot the distribution of a stimulus parameter
        if 'orientation' in stim_df.columns:
            plt.figure(figsize=(8, 4))
            # sns.histplot might not be suitable if orientation is categorical or has few unique values
            # sns.countplot is better for categorical data
            unique_orientations = stim_df['orientation'].nunique()
            if unique_orientations < 20: # Heuristic for using countplot
                 sns.countplot(data=stim_df, x='orientation')
            else: # Use histplot if many unique values, though may still not be ideal
                 sns.histplot(stim_df['orientation'], kde=False, bins=30)
            plt.title(f"Distribution of 'orientation' in {interval_table_name}")
            plt.xlabel("Orientation (degrees)")
            plt.ylabel("Count")
            plt.show()
        else:
            print(f"'orientation' column not found in {interval_table_name}.")

    else:
        print(f"Interval table '{interval_table_name}' not found. Available tables: {list(nwbfile.intervals.keys())}")
else:
    print("No interval tables found in nwbfile.intervals.")


# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to:
# *   Load a Dandiset from the DANDI Archive.
# *   Access and inspect an NWB file within the Dandiset, specifically `sub-692072/sub-692072_ses-1298465622.nwb`.
# *   View basic metadata and understand the general structure of the NWB file.
# *   Load and visualize example data, including:
#     *   Running speed over time.
#     *   Pupil area over time.
#     *   Spike times for a few units as a raster plot.
#     *   Properties from a stimulus presentation table.
#
# **Future Directions:**
#
# This is just a starting point. Researchers can expand on this by:
# 1.  **Exploring other NWB files:** This Dandiset contains many NWB files. Similar analyses can be applied to other files, including those with `_ecephys` (containing raw ephys data) or `_image` (potentially containing imaging data if applicable, though this one seems focused on ephys behavior).
# 2.  **Detailed E LFP and Spike Analysis:** For files containing ephys data (e.g., ending in `_ecephys.nwb` or if units are present in general files), one could perform more in-depth LFP analysis (e.g., power spectra, coherence) or spike train analysis (e.g., firing rate statistics, inter-spike intervals, correlations with behavior or stimuli). *Note: The main NWB file we explored here already contains sorted units.*
# 3.  **Relating Neural Activity to Behavior and Stimuli:** A key goal would be to correlate neural activity (spike rates, LFP features) with behavioral variables (running speed, pupil size) and stimulus presentations (using the `intervals` tables).
# 4.  **Analyzing data across different stimulus conditions:** The `intervals` tables contain rich information about various visual stimuli presented. One could analyze neural responses conditioned on these different stimuli.
# 5.  **Population Analysis:** Investigate how populations of neurons collectively represent information.
# 6.  **Cross-session/subject comparisons:** If applicable, compare findings across different recording sessions or subjects within the Dandiset.
#
# Remember to consult the Dandiset documentation and related publications for more context on the experimental design and data.

# %% [markdown]
# ---
# End of AI-Generated Notebook.
# %%
# Final check: close the HDF5 file and remfile object
# This is good practice, although not strictly required by the prompt for this specific notebook's execution flow.
try:
    nwb_io.close()
    # h5_nwb_file is managed by NWBHDF5IO, closing io should close it.
    # If remfile needs explicit closing and is not managed by h5py.File upon its closure:
    if hasattr(remote_nwb_file, 'close'):
         remote_nwb_file.close()
    print("\nNWB file and remote file object closed.")
except Exception as e:
    print(f"Error closing files: {e}")