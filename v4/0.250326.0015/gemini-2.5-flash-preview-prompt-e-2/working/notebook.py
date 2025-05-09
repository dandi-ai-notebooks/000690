# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
#
# **Caution**: This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.

# %% [markdown]
# ## Dandiset Overview
#
# This notebook explores Dandiset 000690, which contains data from the Allen Institute Openscope - Vision2Hippocampus project. This project aims to understand how visual representations evolve from the LGN through visual cortex and into hippocampal regions in mice. The data includes Neuropixels extracellular electrophysiology from various brain regions while mice were presented with simple and complex visual stimuli.
#
# You can find more information about this Dandiset on the DANDI Archive: https://dandiarchive.org/dandiset/000690/0.250326.0015

# %% [markdown]
# ## Notebook Contents
#
# This notebook will cover:
#
# *   Connecting to the DANDI archive and loading the Dandiset metadata.
# *   Loading a specific NWB file from the Dandiset.
# *   Examining the structure and contents of the NWB file.
# *   Loading and visualizing samples of the Eye Tracking and Running Wheel data.
# *   Summarizing the findings and suggesting future analysis directions.

# %% [markdown]
# ## Required Packages
#
# To run this notebook, you need the following Python packages installed:
#
# *   `dandi`
# *   `pynwb`
# *   `remfile`
# *   `h5py`
# *   `numpy`
# *   `pandas`
# *   `matplotlib`
# *   `seaborn`

# %%
from itertools import islice
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
# ## Loading an NWB file
#
# We will load the NWB file `sub-692072/sub-692072_ses-1298465622.nwb` to explore its contents. This file has the asset ID `fbcd4fe5-7107-41b2-b154-b67f783f23dc`.
#
# The URL for this asset is: https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/

# %%
# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Show some basic metadata from the NWB file
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject Species: {nwb.subject.species}")
print(f"Subject Genotype: {nwb.subject.genotype}")

# %% [markdown]
# ## NWB File Contents
#
# The chosen NWB file contains various types of data related to the experimental session:
#
# *   **Acquisition**: Raw data acquired during the experiment. This includes Eye Tracking data (corneal_reflection_tracking, eye_tracking, pupil_tracking, likely_blink) and raw running wheel rotation/voltage data.
# *   **Processing**: Processed data derived from the raw acquisition data. This includes processed running wheel speed and rotation. Stimulus timing information is also stored here.
# *   **Electrodes**: Metadata about the extracellular electrodes used in the experiment, including location and grouping.
# *   **Devices**: Information about the recording devices (Neuropixels probes).
# *   **Intervals**: Time intervals corresponding to different experimental phases or stimulus presentations.
# *   **Subject**: Metadata about the experimental subject (mouse).
# *   **Units**: Spike times, amplitudes, and waveforms for identified units (neurons). This table also includes various quality metrics for the units.
#
# Here's a simplified view of the NWB file structure:
#
# ```
# NWBFile
# ├── acquisition
# │   ├── EyeTracking (EllipseEyeTracking)
# │   │   ├── corneal_reflection_tracking (EllipseSeries)
# │   │   ├── eye_tracking (EllipseSeries)
# │   │   ├── likely_blink (TimeSeries)
# │   │   └── pupil_tracking (EllipseSeries)
# │   ├── raw_running_wheel_rotation (TimeSeries)
# │   ├── running_wheel_signal_voltage (TimeSeries)
# │   └── running_wheel_supply_voltage (TimeSeries)
# ├── processing
# │   ├── running (ProcessingModule)
# │   │   ├── running_speed (TimeSeries)
# │   │   ├── running_speed_end_times (TimeSeries)
# │   │   └── running_wheel_rotation (TimeSeries)
# │   └── stimulus (ProcessingModule)
# │       └── timestamps (TimeSeries)
# ├── electrode_groups (LabelledDict)
# ├── devices (LabelledDict)
# ├── intervals (LabelledDict)
# ├── electrodes (DynamicTable)
# ├── subject (EcephysSpecimen)
# ├── invalid_times (TimeIntervals)
# └── units (Units)
# ```
#
# You can explore this NWB file directly in neurosift: https://neurosift.app/nwb?url=https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/&dandisetId=000690&dandisetVersion=draft

# %% [markdown]
# ## Exploring Eye Tracking Data
#
# The `EyeTracking` module in the `acquisition` group contains data related to the subject's eye movements. This includes tracking of the pupil, corneal reflection, and the entire eye. There is also a timeseries indicating whether a blink is likely.

# %%
# Access the EyeTracking data
eye_tracking_module = nwb.acquisition['EyeTracking']

# Get the data for pupil tracking position, area, and timestamps (subset)
N_POINTS_TO_PLOT = 10000
pupil_position = eye_tracking_module['pupil_tracking'].data[:N_POINTS_TO_PLOT, :]
pupil_area = eye_tracking_module['pupil_tracking'].area[:N_POINTS_TO_PLOT]
timestamps_eye = eye_tracking_module['pupil_tracking'].timestamps[:N_POINTS_TO_PLOT] # Load only subset of timestamps

# Print shapes of the data subsets
print(f"Shape of pupil_position (subset): {pupil_position.shape}")
print(f"Shape of pupil_area (subset): {pupil_area.shape}")
print(f"Shape of timestamps_eye (subset): {timestamps_eye.shape}")

# Plot the subset of the pupil area data
plt.figure(figsize=(12, 4))
plt.plot(timestamps_eye, pupil_area)
plt.xlabel('Time (s)')
plt.ylabel('Pupil Area')
plt.title('Subset of Pupil Area over Time')
sns.set_theme() # Apply seaborn style
plt.show()

# Plot the subset of the pupil position data (x and y)
plt.figure(figsize=(12, 4))
plt.plot(timestamps_eye, pupil_position[:, 0], label='X Position')
plt.plot(timestamps_eye, pupil_position[:, 1], label='Y Position')
plt.xlabel('Time (s)')
plt.ylabel('Pupil Position')
plt.title('Subset of Pupil Position (X and Y) over Time')
sns.set_theme() # Apply seaborn style
plt.legend()
plt.show()

# Access the likely_blink data (subset) and timestamps (subset)
likely_blink = eye_tracking_module.likely_blink.data[:N_POINTS_TO_PLOT] # Load only subset of data
timestamps_blink = eye_tracking_module.likely_blink.timestamps[:N_POINTS_TO_PLOT] # Load only subset of timestamps

# Print shape of likely_blink data subset
print(f"Shape of likely_blink (subset): {likely_blink.shape}")
print(f"Shape of timestamps_blink (subset): {timestamps_blink.shape}")

# Plot the subset of the likely_blink data
plt.figure(figsize=(12, 2))
plt.plot(timestamps_blink, likely_blink)
plt.xlabel('Time (s)')
plt.ylabel('Likely Blink (boolean)')
plt.title('Subset of Likely Blink Indicator over Time')
sns.set_theme() # Apply seaborn style
plt.show()


# %% [markdown]
# ## Exploring Running Wheel Data
#
# The processed running speed is available in the `running` processing module.

# %%
# Access the running data
running_module = nwb.processing['running']

# Get the running speed data (subset) and timestamps (subset)
N_POINTS_TO_PLOT_RUNNING = 10000 # Using a different variable name just in case
running_speed = running_module['running_speed'].data[:N_POINTS_TO_PLOT_RUNNING]
timestamps_running = running_module['running_speed'].timestamps[:N_POINTS_TO_PLOT_RUNNING] # load subset of timestamps

# Print shape of running speed data subset
print(f"Shape of running_speed (subset): {running_speed.shape}")
print(f"Shape of timestamps_running (subset): {timestamps_running.shape}")

# Plot the subset of the running speed data
plt.figure(figsize=(12, 4))
plt.plot(timestamps_running, running_speed)
plt.xlabel('Time (s)')
plt.ylabel('Running Speed (cm/s)')
plt.title('Subset of Running Speed over Time')
sns.set_theme() # Apply seaborn style
plt.show()


# %% [markdown]
# ## Exploring Stimulus Presentation Intervals
#
# The `intervals` section contains tables describing the presentation times and properties of different stimuli. We can load these tables into pandas DataFrames for easier analysis.

# %%
# List available stimulus interval tables
print("Available stimulus interval tables:")
for interval_name in nwb.intervals:
    if interval_name.endswith('_presentations'):
        print(f"- {interval_name}")

# Load one of the stimulus presentation tables and display the first few rows
try:
    # Display the first 5 rows of the dataframe without loading the entire table
    print("\nFirst 5 rows of 'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations' table:")
    # Accessing a slice and then converting to dataframe should be faster
    stimulus_presentations_slice = nwb.intervals['SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations'][0:5]
    stimulus_presentations_df_head = pd.DataFrame(stimulus_presentations_slice.to_dict()) # Convert to dict then DataFrame
    print(stimulus_presentations_df_head)
except KeyError:
    print("\n'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations' table not found in this NWB file.")
except Exception as e:
    print(f"\nCould not load stimulus presentation table head: {e}")


# %% [markdown]
# ## Exploring Units Data
#
# The `units` table contains information about the sorted spikes. We can examine the structure and glimpse the contents without loading the entire table.

# %%
# Access the units table and display the first few rows and column names
try:
    # Display the first 5 rows of the units table without loading the entire table
    print("First 5 rows of the units table:")
    # Accessing a slice and then converting to dataframe should be faster
    units_slice = nwb.units[0:5]
    units_df_head = pd.DataFrame(units_slice.to_dict()) # Convert to dict then DataFrame
    print(units_df_head)

    # Print all column names in the units table
    print("\nColumns in the units table:")
    print(nwb.units.colnames)

    # Select a subset of columns and display the first few rows
    subset_colnames = ['quality', 'firing_rate', 'isolation_distance', 'peak_channel_id', 'location']
    try:
        subset_units_slice = nwb.units.to_dataframe(columns=subset_colnames).head()
        print("\nSubset of units table with key columns (first 5 rows):")
        print(subset_units_slice)
    except Exception as e:
         print(f"\nCould not display subset of units table columns: {e}")


    # Get spike times for the first unit (first few spikes)
    if len(nwb.units) > 0:
        first_unit_id = nwb.units.id[0]
        # Accessing spike times for a specific unit - load only first 10 spikes
        try:
            spike_times_first_unit = nwb.units['spike_times'][0][0:10]
            print(f"\nFirst 10 spike times for unit with ID {first_unit_id}: {spike_times_first_unit}")
        except Exception as e:
            print(f"\nCould not retrieve spike times for the first unit: {e}")
    else:
        print("\nNo units found in the units table.")

except Exception as e:
    print(f"\nCould not access or process units table: {e}")


# %% [markdown]
# ## Summary and Future Directions
#
# This notebook provided a basic introduction to exploring Dandiset 000690 and examining data within one of its NWB files. We demonstrated how to load the Dandiset, access an NWB file, view its structure, and visualize samples of eye tracking and running behavior data. We also showed how to access stimulus interval information and unit data.
#
# Possible future directions for analysis include:
#
# *   Analyzing the relationship between visual stimuli and neural activity (spikes/waveforms).
# *   Investigating how running speed or eye movements correlate with neural responses.
# *   Exploring the LFP data available in other NWB files within the Dandiset (e.g., `_ecephys.nwb` files for each probe).
# *   Examining the image data (e.g., functional imaging) available in the `_image.nwb` files.
# *   Performing spike analysis, such as inter-spike intervals or cross-correlations.
#
# Remember to consult the Dandiset metadata and the NWB file structure details for a complete understanding of the available data.

# %%
# Close the NWB file
io.close()
h5_file.close()
remote_file.close()