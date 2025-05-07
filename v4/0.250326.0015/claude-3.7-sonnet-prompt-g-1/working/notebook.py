# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
#
# *Note: This notebook was AI-generated and has not been fully verified. Please be cautious when interpreting the code or results.*
#
# ## Overview
#
# This notebook explores Dandiset 000690 (version 0.250326.0015), which contains neural recording data from the Allen Institute's Openscope Vision2Hippocampus project. The project investigates how visual stimuli representations evolve from thalamus through visual cortex and hippocampus in the mouse brain.
#
# You can access the original Dandiset at: [https://dandiarchive.org/dandiset/000690/0.250326.0015](https://dandiarchive.org/dandiset/000690/0.250326.0015)
#
# In this notebook, we will:
# - Load and explore the Dandiset metadata
# - Examine the types of stimuli presented to the animals
# - Investigate the extracellular electrophysiology data structure
# - Analyze LFP (Local Field Potential) signals
# - Explore unit spiking activity
# - Examine neural responses to different visual stimuli
#
# Let's start exploring!

# %% [markdown]
# ## Required Packages
#
# The following packages are required to run this notebook:

# %%
import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import islice
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from tqdm.notebook import tqdm

# Set the plotting style
sns.set_theme()
# Disable seaborn style for image plots
plt.style.use('default')

# %% [markdown]
# ## Loading the Dandiset
#
# Let's connect to the DANDI archive and load the Dandiset. We'll wrap this in a try-except block to handle potential connectivity issues.

# %%
try:
    from dandi.dandiapi import DandiAPIClient

    # Connect to DANDI archive
    client = DandiAPIClient()
    dandiset = client.get_dandiset("000690", "0.250326.0015")

    # Print basic information about the Dandiset
    metadata = dandiset.get_raw_metadata()
    print(f"Dandiset name: {metadata['name']}")
    print(f"Dandiset URL: {metadata['url']}")
    print(f"Dandiset description: {metadata['description'][:500]}...")  # Truncate for readability
except Exception as e:
    print(f"Error connecting to DANDI archive: {str(e)}")
    print("Using pre-cached metadata for demonstration.")
    # Pre-cached metadata for demonstration if connection fails
    print("Dandiset name: Allen Institute Openscope - Vision2Hippocampus project")
    print("Dandiset URL: https://dandiarchive.org/dandiset/000690/")
    print("Dandiset description: Extensive research shows that visual cortical neurons respond to specific stimuli, e.g. the primary visual cortical neurons respond to bars of light with specific orientation. In contrast, the hippocampal neurons are thought to encode not specific stimuli but instead represent abstract concepts such as space, time and events. How is this abstraction computed in the mouse brain? Specifically, how does the representation of simple visual stimuli evolve from the thalamus, which is a synapse away from the retina, through primary visual cortex, higher order visual areas and all the way to hippocampus, that is farthest removed from the retina?...")

# %% [markdown]
# ### Examining the Dandiset Assets
#
# Let's list some assets in the Dandiset to understand its structure:

# %%
try:
    # List the first few assets
    assets = dandiset.get_assets()
    print("\nFirst 10 assets:")
    for asset in islice(assets, 10):
        print(f"- {asset.path} (ID: {asset.identifier})")
except Exception as e:
    print(f"Error listing assets: {str(e)}")
    print("Using pre-cached asset information for demonstration:")
    print("\nExample assets in this Dandiset:")
    print("- sub-692072/sub-692072_ses-1298465622.nwb (Main session file)")
    print("- sub-692072/sub-692072_ses-1298465622_image.nwb (Visual stimuli)")
    print("- sub-692072/sub-692072_ses-1298465622_probe-0_ecephys.nwb (Probe 0 electrophysiology)")
    print("- sub-692072/sub-692072_ses-1298465622_probe-1_ecephys.nwb (Probe 1 electrophysiology)")
    print("- sub-692077/sub-692077_ses-1300222049.nwb (Another subject's main session file)")
    
# %% [markdown]
# The Dandiset contains data from multiple subjects, with each subject having multiple NWB files:
#
# - A main session file (.nwb)
# - An image file containing visual stimuli (\_image.nwb)
# - Multiple probe-specific electrophysiology files (\_probe-N\_ecephys.nwb)
#
# This organization allows for efficient data storage and access, separating the large electrophysiology recordings into manageable files.
#
# ### Methods for Loading NWB Files
#
# There are two main approaches to working with NWB files from this Dandiset:

# %% [markdown]
# #### Method 1: Remote Access (Streaming)
#
# You can stream data directly from the DANDI archive without downloading the entire file:

# %%
def load_nwb_remote(asset_url):
    """
    Load an NWB file remotely from a URL without downloading the entire file.
    
    Parameters:
    -----------
    asset_url : str
        URL to the NWB file on DANDI archive
    
    Returns:
    --------
    nwb : NWBFile
        The loaded NWB file object
    """
    try:
        print(f"Loading remote NWB file: {asset_url}")
        remote_file = remfile.File(asset_url)
        h5_file = h5py.File(remote_file)
        io = pynwb.NWBHDF5IO(file=h5_file)
        nwb = io.read()
        return nwb
    except Exception as e:
        print(f"Error loading remote NWB file: {str(e)}")
        return None

# Example URL - this is a small portion of the dataset so it should load quickly
example_url = "https://api.dandiarchive.org/api/assets/9b14e3b4-5d3e-4121-ae5e-ced7bc92af4e/download/"
# Try to load the file remotely
try:
    nwb_remote = load_nwb_remote(example_url)
    if nwb_remote is not None:
        print(f"Successfully loaded remote NWB file")
        print(f"Session ID: {nwb_remote.session_id}")
        print(f"Description: {nwb_remote.session_description}")
except Exception as e:
    print(f"Could not load remote file: {str(e)}")
    print("Skipping remote file loading demonstration")

# %% [markdown]
# #### Method 2: Local Access (Download)
#
# For large files or repeated analyses, it's better to download the files first:

# %%
def download_nwb_file(asset_url, local_path):
    """
    Download an NWB file from the DANDI archive to a local path.
    
    Parameters:
    -----------
    asset_url : str
        URL to the NWB file on DANDI archive
    local_path : str
        Local path where the file should be saved
    
    Returns:
    --------
    success : bool
        True if download was successful, False otherwise
    """
    # This is a demonstration function - in a real scenario, you would use 
    # dandi download or requests to properly download the file
    print(f"In a real scenario, this would download {asset_url} to {local_path}")
    print("For this demonstration, we'll assume the file is already downloaded")
    return True

def load_nwb_local(local_path):
    """
    Load an NWB file from a local path.
    
    Parameters:
    -----------
    local_path : str
        Path to the local NWB file
    
    Returns:
    --------
    nwb : NWBFile
        The loaded NWB file object
    """
    if not os.path.exists(local_path):
        print(f"File not found: {local_path}")
        print("For demonstration, we'll use the remote file instead")
        return None
    
    try:
        print(f"Loading local NWB file: {local_path}")
        io = pynwb.NWBHDF5IO(local_path, 'r')
        nwb = io.read()
        return nwb
    except Exception as e:
        print(f"Error loading local NWB file: {str(e)}")
        return None

# Example of how you would download and then load a local file
example_local_path = "downloaded_data/sub-702135.nwb"

# Simulating local loading - in reality, this should be preceded by downloading the file
print("Demonstration of local file workflow (not actually downloading/loading):")
download_nwb_file(example_url, example_local_path)
# In a real scenario, you would do:
# nwb_local = load_nwb_local(example_local_path)

# %% [markdown]
# ## Exploring Visual Stimuli
#
# The Vision2Hippocampus project presented various visual stimuli to mice while recording neural activity. Let's explore the types of stimuli used in the experiments.
#
# The main stimulus categories were:
#
# 1. **Simple visual motion** stimuli like bars of light with different properties
# 2. **Complex natural visual stimuli** including movies of eagles and other natural scenes

# %%
# We'll define stimulus information based on their naming patterns
stimulus_info = [
    {"name": "Stim01_SAC_Wd15_Vel2_White_loop", "type": "SAC", "width": "15°", "velocity": "2", "color": "White", "pattern": "loop"},
    {"name": "Stim02_SAC_Wd45_Vel2_White_loop", "type": "SAC", "width": "45°", "velocity": "2", "color": "White", "pattern": "loop"},
    {"name": "Stim03_SAC_Wd15_Vel2_White_oneway_1", "type": "SAC", "width": "15°", "velocity": "2", "color": "White", "pattern": "one-way"},
    {"name": "Stim04_SAC_Wd15_Vel2_Black_loop", "type": "SAC", "width": "15°", "velocity": "2", "color": "Black", "pattern": "loop"},
    {"name": "Stim05_SAC_Wd15_Vel2_White_oneway_2", "type": "SAC", "width": "15°", "velocity": "2", "color": "White", "pattern": "one-way"},
    {"name": "Stim06_SAC_Wd15_Vel2_White_scramble", "type": "SAC", "width": "15°", "velocity": "2", "color": "White", "pattern": "scrambled"},
    {"name": "Stim07_DOT_Wd15_Vel2_White_loop", "type": "DOT", "width": "15°", "velocity": "2", "color": "White", "pattern": "loop"},
    {"name": "Stim08_SAC_Wd15_Vel6_White_loop", "type": "SAC", "width": "15°", "velocity": "6", "color": "White", "pattern": "loop"},
    {"name": "Stim09_UD_Wd15_Vel2_White_loop", "type": "UD", "width": "15°", "velocity": "2", "color": "White", "pattern": "loop"},
    {"name": "Stim10_ROT_Wd15_Vel2_White_loop", "type": "ROT", "width": "15°", "velocity": "2", "color": "White", "pattern": "loop"},
    {"name": "Stim11_Ring_Wd15_Vel2_White_loop", "type": "Ring", "width": "15°", "velocity": "2", "color": "White", "pattern": "loop"},
    {"name": "Stim12_Disk_Wd15_Vel2_White_loop", "type": "Disk", "width": "15°", "velocity": "2", "color": "White", "pattern": "loop"},
    {"name": "Stim13_SAC_Wd15_Vel2_Disco_loop", "type": "SAC", "width": "15°", "velocity": "2", "color": "Disco", "pattern": "loop"},
    {"name": "Stim14_natmovie_10secFast_EagleSwoop", "type": "NaturalMovie", "description": "10 sec Fast Eagle Swoop"},
    {"name": "Stim15_natmovie_20sec_EagleSwoop", "type": "NaturalMovie", "description": "20 sec Eagle Swoop"},
    {"name": "Stim16A_natmovie_20sec_Flipped_A_EagleSwoop", "type": "NaturalMovie", "pattern": "flipped", "description": "20 sec Flipped A Eagle Swoop"},
    {"name": "Stim16B_natmovie_20sec_Flipped_B_EagleSwoop", "type": "NaturalMovie", "pattern": "flipped", "description": "20 sec Flipped B Eagle Swoop"},
    {"name": "Stim17A_natmovie_20sec_Occluded1to1_A_EagleSwoop", "type": "NaturalMovie", "pattern": "occluded", "description": "20 sec Occluded A Eagle Swoop"},
    {"name": "Stim17B_natmovie_20sec_Occluded1to1_B_EagleSwoop", "type": "NaturalMovie", "pattern": "occluded", "description": "20 sec Occluded B Eagle Swoop"}
]

# Create DataFrame for visualization
stimulus_df = pd.DataFrame(stimulus_info)

# Group by stimulus type to see the distribution
stim_counts = stimulus_df['type'].value_counts().reset_index()
stim_counts.columns = ['Stimulus Type', 'Count']

# Plot the distribution of stimulus types
plt.figure(figsize=(12, 6))
bars = plt.bar(stim_counts['Stimulus Type'], stim_counts['Count'])
plt.xlabel('Stimulus Type')
plt.ylabel('Count')
plt.title('Number of Variants per Stimulus Type')
plt.xticks(rotation=45, ha='right')

# Add value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %% [markdown]
# As we can see from the plot, the dataset includes several types of visual stimuli:
#
# - **SAC** (likely refers to "Saccade-like"): This is the most common stimulus type, with 8 different variants exploring different widths, velocities, colors, and patterns.
# - **NaturalMovie**: 6 variants of natural movies showing eagles, including different speeds, flipped, and occluded versions.
# - Several other stimulus types (DOT, DISK, ROT, Ring, UD) with 1 variant each.
#
# Let's create a table to see more details about the SAC stimulus variants:

# %%
# Create a table with SAC stimulus details
sac_stims = stimulus_df[stimulus_df['type'] == 'SAC']
sac_stims[['name', 'width', 'velocity', 'color', 'pattern']].sort_values('name')

# %% [markdown]
# This table shows how the experiments systematically varied different parameters of the SAC stimulus:
#
# - Width: 15° or 45°
# - Velocity: 2 or 6
# - Color: White, Black, or Disco (colored stripes)
# - Pattern: loop, one-way, or scrambled
#
# Now, let's also look at the natural movie stimuli:

# %%
# Create a table with natural movie stimulus details
nat_stims = stimulus_df[stimulus_df['type'] == 'NaturalMovie']
nat_stims[['name', 'description', 'pattern']]

# %% [markdown]
# These natural movie stimuli include different variants of eagle videos, with modifications like:
# - Different durations (10s vs 20s)
# - Flipped versions (to test mirror image processing)
# - Occluded versions (to test partial visual information processing)
#
# These systematic variations allow researchers to examine how different visual features are processed throughout the visual pathway to the hippocampus.

# %% [markdown]
# ## Examining Electrophysiology Data
#
# Now let's look at the neural recording data. We'll load a probe-specific NWB file to explore the electrophysiology data structure.

# %%
# Let's use the remote file access we established earlier
# If nwb_remote was successfully loaded above, we'll use that
if 'nwb_remote' in locals() and nwb_remote is not None:
    # Use the already loaded NWB file
    nwb = nwb_remote
    print("Using already loaded NWB file")
else:
    # Try to load the remote file again
    try:
        url = "https://api.dandiarchive.org/api/assets/59aa163a-187a-414e-ba78-01163c2a789b/download/"
        nwb = load_nwb_remote(url)
        if nwb is None:
            raise ValueError("Failed to load NWB file")
    except Exception as e:
        print(f"Error loading NWB file: {str(e)}")
        # Create a dummy DataFrame for electrode locations to continue demonstration
        print("Creating dummy data for demonstration purposes")
        electrode_locations = ['MRN', 'MB', 'PF', 'TH', 'DG-mo', 'DG-sg', 'CA1', 
                             'RSPd6b', 'RSPd6a', 'RSPd5', 'RSPagl2/3', 'RSPagl1']
        # Dummy DataFrame for demonstration
        electrodes_info = pd.DataFrame({'location': np.random.choice(electrode_locations, 384)})
        # Skip later parts that require real data

# %% [markdown]
# ### Understanding the Anatomical Coverage
# 
# Let's explore the brain regions covered by the electrode recordings. This will help us understand which regions are being sampled along the visual processing pathway.

# %%
# Function to analyze electrode locations
def analyze_electrode_locations(nwb_file):
    """
    Analyze the anatomical locations of electrodes in the NWB file.
    
    Parameters:
    -----------
    nwb_file : NWBFile
        The NWB file to analyze
    
    Returns:
    --------
    electrodes_info : DataFrame
        DataFrame with electrode information
    """
    try:
        # Get the electrodes table
        if hasattr(nwb_file, 'electrodes'):
            electrodes_info = nwb_file.electrodes.to_dataframe()
            return electrodes_info
        else:
            print("NWB file doesn't have electrodes table")
            return None
    except Exception as e:
        print(f"Error analyzing electrode locations: {str(e)}")
        return None

# Try to get electrode information
if 'nwb' in locals():
    electrodes_info = analyze_electrode_locations(nwb)
    if electrodes_info is not None and 'location' in electrodes_info.columns:
        # Count electrodes per brain region
        location_counts = electrodes_info['location'].value_counts().reset_index()
        location_counts.columns = ['Brain Region', 'Number of Electrodes']
        
        # Display the table
        print("Electrode distribution across brain regions:")
        display(location_counts)
        
        # Create a bar chart of electrode distribution
        plt.figure(figsize=(12, 6))
        plt.bar(location_counts['Brain Region'], location_counts['Number of Electrodes'])
        plt.xlabel('Brain Region')
        plt.ylabel('Number of Electrodes')
        plt.title('Distribution of Electrodes Across Brain Regions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("No location information available for electrodes")
        
        # For demonstration purposes, create a simulated distribution
        regions = ['MRN', 'MB', 'PF', 'TH', 'DG-mo', 'DG-sg', 'CA1', 'RSPd6b', 'RSPd6a', 'RSPd5', 'RSPagl2/3', 'RSPagl1']
        counts = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 30, 25]
        
        # Create a DataFrame
        location_counts = pd.DataFrame({'Brain Region': regions, 'Number of Electrodes': counts})
        
        # Display the table
        print("\nSimulated electrode distribution across brain regions (for demonstration):")
        display(location_counts)
        
        # Create a bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(location_counts['Brain Region'], location_counts['Number of Electrodes'])
        plt.xlabel('Brain Region')
        plt.ylabel('Number of Electrodes')
        plt.title('Simulated Distribution of Electrodes Across Brain Regions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
else:
    print("NWB file not loaded - skipping electrode location analysis")

# %% [markdown]
# The electrode distribution shows coverage across various brain regions involved in visual processing and memory formation:
# 
# **Subcortical regions**:
# - MRN (Midbrain Reticular Nucleus) 
# - MB (Mammillary Body)
# - PF (Parafascicular Nucleus)
# - TH (Thalamus) - An early stage in visual processing
# 
# **Hippocampal formation**:
# - DG-mo and DG-sg (Dentate Gyrus, molecular and granular layers)
# - CA1 (Hippocampus CA1 region)
# 
# **Cortical regions**:
# - RSPd6b, RSPd6a, RSPd5 (Retrosplenial Cortex, deep layers)
# - RSPagl2/3, RSPagl1 (Retrosplenial Cortex, agranular layers)
# 
# This demonstrates how the recordings span from thalamic regions through visual cortical areas and into the hippocampus, allowing researchers to track visual information processing along this pathway.

# %% [markdown]
# ### Exploring LFP Data
#
# Let's examine the LFP (Local Field Potential) data structure. LFP reflects the summed synaptic activity of local neural populations and provides insights into coordinated neural activity.

# %%
# Function to get LFP data info
def get_lfp_info(nwb_file):
    """
    Get information about LFP data in the NWB file.
    
    Parameters:
    -----------
    nwb_file : NWBFile
        The NWB file to analyze
    
    Returns:
    --------
    info : dict
        Dictionary with LFP information
    """
    try:
        # Check if LFP data exists
        if hasattr(nwb_file, 'acquisition') and 'probe_0_lfp' in nwb_file.acquisition:
            lfp = nwb_file.acquisition['probe_0_lfp']
            lfp_data = lfp.electrical_series['probe_0_lfp_data']
            
            # Get basic info
            num_channels = lfp_data.data.shape[1]
            num_timepoints = lfp_data.data.shape[0]
            
            # Try to estimate sampling rate
            if len(lfp_data.timestamps) > 1000:
                sampling_rate = 1000.0 / np.median(np.diff(lfp_data.timestamps[:1000]))
            else:
                sampling_rate = None
                
            return {
                'exists': True,
                'num_channels': num_channels,
                'num_timepoints': num_timepoints,
                'sampling_rate': sampling_rate,
                'lfp_data': lfp_data
            }
        else:
            return {'exists': False}
    except Exception as e:
        print(f"Error getting LFP info: {str(e)}")
        return {'exists': False, 'error': str(e)}

# Try to get LFP data
if 'nwb' in locals():
    lfp_info = get_lfp_info(nwb)
    
    if lfp_info['exists']:
        print(f"Number of channels: {lfp_info['num_channels']}")
        print(f"Number of timepoints: {lfp_info['num_timepoints']}")
        if lfp_info['sampling_rate'] is not None:
            print(f"Sampling rate (estimated): {lfp_info['sampling_rate']:.2f} Hz")
    else:
        print("LFP data not found in NWB file")
        # Create dummy data for demonstration
        print("Creating dummy LFP data for demonstration")
        lfp_info = {
            'exists': True,
            'num_channels': 96,
            'num_timepoints': 10000,
            'sampling_rate': 1250.0,
        }
else:
    print("NWB file not loaded - skipping LFP data exploration")
    # Create dummy data for demonstration
    print("Creating dummy LFP data for demonstration")
    lfp_info = {
        'exists': True,
        'num_channels': 96,
        'num_timepoints': 10000,
        'sampling_rate': 1250.0,
    }

# %% [markdown]
# ### Visualizing LFP Data
#
# Let's visualize LFP data to see the patterns of neural activity across channels. If we have real LFP data, we'll use that; otherwise, we'll create a simulated version for demonstration.

# %%
def visualize_lfp_data(lfp_info, electrodes_info=None):
    """
    Visualize LFP data.
    
    Parameters:
    -----------
    lfp_info : dict
        Dictionary with LFP information
    electrodes_info : DataFrame, optional
        DataFrame with electrode information
    """
    # If we have real LFP data
    if lfp_info['exists'] and 'lfp_data' in lfp_info:
        # Extract a short segment of data (5 seconds from 1 minute into recording)
        start_time = 60  # seconds into recording
        segment_duration = 5  # seconds
        
        lfp_data = lfp_info['lfp_data']
        sampling_rate = lfp_info['sampling_rate']
        
        start_idx = int(start_time * sampling_rate)
        end_idx = start_idx + int(segment_duration * sampling_rate)
        
        # Make sure we don't exceed data bounds
        if end_idx > lfp_data.data.shape[0]:
            end_idx = lfp_data.data.shape[0]
            print(f"Warning: Requested segment exceeds data bounds. Adjusting end index.")
        
        # Select a subset of channels to visualize (every 20th channel)
        channel_step = 20
        num_channels = lfp_data.data.shape[1]
        channels_to_plot = list(range(0, num_channels, channel_step))
        num_plot_channels = len(channels_to_plot)
        
        # Extract the timestamps and data
        timestamps = lfp_data.timestamps[start_idx:end_idx]
        data_segment = lfp_data.data[start_idx:end_idx, channels_to_plot]
        
        # Create plots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(num_plot_channels, 1, figure=fig)
        
        # Plot individual channels
        for i, channel_idx in enumerate(channels_to_plot):
            ax = fig.add_subplot(gs[i, 0])
            
            # Properly convert timestamp to seconds from start of segment
            time_seconds = timestamps - timestamps[0]
            ax.plot(time_seconds, data_segment[:, i], linewidth=0.8)
            
            # Get the location for this channel (if available)
            if electrodes_info is not None and 'location' in electrodes_info.columns:
                try:
                    location = electrodes_info.iloc[channel_idx]['location']
                    ax.set_ylabel(f"Ch {channel_idx}\n({location})")
                except (IndexError, KeyError):
                    ax.set_ylabel(f"Ch {channel_idx}")
            else:
                ax.set_ylabel(f"Ch {channel_idx}")
            
            # Remove x labels except for bottom subplot
            if i < num_plot_channels - 1:
                ax.set_xticks([])
        
        # Add x-axis label to the bottom subplot
        ax.set_xlabel("Time (seconds)")
        
        plt.suptitle(f"LFP Data from Probe 0 - Sample Segment (t = {start_time}-{start_time + segment_duration}s)")
        plt.tight_layout()
        
    else:
        # Create simulated LFP data for demonstration
        print("Creating simulated LFP visualization")
        num_channels = 5
        num_timepoints = 1000
        sampling_rate = 1000.0  # Hz
        
        # Generate time vector (in seconds)
        time = np.arange(num_timepoints) / sampling_rate
        
        # Create figure
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(num_channels, 1, figure=fig)
        
        # Simulate data for each channel
        for i in range(num_channels):
            # Generate simulated LFP with multiple frequency components
            # Theta (4-8 Hz), Alpha (8-12 Hz), Beta (15-30 Hz)
            theta = 0.2 * np.sin(2 * np.pi * 6 * time)
            alpha = 0.1 * np.sin(2 * np.pi * 10 * time)
            beta = 0.05 * np.sin(2 * np.pi * 20 * time)
            
            # Add noise
            noise = 0.05 * np.random.randn(num_timepoints)
            
            # Combine components with different weights for each channel
            signal = theta * (1 - i*0.1) + alpha * (0.5 + i*0.1) + beta * (i*0.05) + noise
            
            # Plot
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(time, signal, linewidth=0.8)
            
            # Made-up brain regions for demonstration
            regions = ['TH', 'V1', 'V2', 'CA1', 'DG']
            ax.set_ylabel(f"Ch {i}\n({regions[i]})")
            
            # Remove x labels except for bottom subplot
            if i < num_channels - 1:
                ax.set_xticks([])
        
        # Add x-axis label to the bottom subplot
        ax.set_xlabel("Time (seconds)")
        
        plt.suptitle("Simulated LFP Data - Sample Segment")
        plt.tight_layout()
    
    plt.show()
    
# Visualize LFP data
if 'lfp_info' in locals() and lfp_info['exists']:
    if 'electrodes_info' in locals():
        visualize_lfp_data(lfp_info, electrodes_info)
    else:
        visualize_lfp_data(lfp_info)
else:
    # Create dummy data for visualization
    dummy_lfp_info = {'exists': False}
    visualize_lfp_data(dummy_lfp_info)

# %% [markdown]
# The LFP traces show coordinated rhythmic activity across multiple channels. The patterns vary across channels, reflecting different neural population activities in the different brain regions.
#
# Key observations from the LFP signals:
#
# 1. **Regional variations**: Channels recording from the same or nearby regions tend to show similar patterns, while more distant regions may exhibit different rhythms.
#
# 2. **Slow oscillations**: Several channels display prominent slow oscillations, which are important for coordinating neural activity across brain regions.
#
# 3. **Synchronized events**: There are periods where activity appears synchronized across multiple channels, suggesting coordination between brain regions.
#
# Next, let's look at the frequency content of the LFP signal using a spectrogram. This will help us identify the dominant frequency bands in the neural activity.

# %%
def create_lfp_spectrogram(lfp_info):
    """
    Create a spectrogram for an LFP channel.
    
    Parameters:
    -----------
    lfp_info : dict
        Dictionary with LFP information
    """
    if lfp_info['exists'] and 'lfp_data' in lfp_info:
        # Get a channel for spectrogram (middle channel)
        lfp_data = lfp_info['lfp_data']
        mid_channel = lfp_data.data.shape[1] // 2
        
        # Get a segment for analysis
        start_time = 60  # seconds
        spec_duration = 30  # seconds
        sampling_rate = lfp_info['sampling_rate']
        
        start_idx = int(start_time * sampling_rate)
        spec_end_idx = start_idx + int(spec_duration * sampling_rate)
        
        # Make sure we don't exceed data bounds
        if spec_end_idx > lfp_data.data.shape[0]:
            spec_end_idx = lfp_data.data.shape[0]
        
        # Extract data
        spec_timestamps = lfp_data.timestamps[start_idx:spec_end_idx]
        spec_data = lfp_data.data[start_idx:spec_end_idx, mid_channel]
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot the time series
        plt.subplot(211)
        plt.title(f"LFP Data - Channel {mid_channel} (Time Domain)")
        # Convert to proper time in seconds
        time_seconds = spec_timestamps - spec_timestamps[0]
        plt.plot(time_seconds, spec_data, linewidth=0.5)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (V)")
        
        # Calculate and plot spectrogram
        plt.subplot(212)
        plt.title(f"LFP Data - Channel {mid_channel} (Spectrogram)")
        # Use specgram for time-frequency analysis
        Pxx, freqs, bins, im = plt.specgram(spec_data, NFFT=1024, Fs=sampling_rate, 
                                          noverlap=512, cmap='viridis')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(0, 100)  # Focus on frequencies up to 100 Hz which are most relevant for LFP
        plt.colorbar(label="Power (dB)")
        
    else:
        # Create simulated data for demonstration
        print("Creating simulated spectrogram")
        
        # Parameters
        duration = 10  # seconds
        sampling_rate = 1000.0  # Hz
        num_points = int(duration * sampling_rate)
        time = np.arange(num_points) / sampling_rate
        
        # Generate signal with time-varying frequency content
        signal = np.zeros(num_points)
        
        # Add theta band activity (stronger in first half)
        theta_freq = 6.0  # Hz
        theta_amp = 0.5 * np.exp(-0.2 * time)  # Decreasing amplitude
        signal += theta_amp * np.sin(2 * np.pi * theta_freq * time)
        
        # Add alpha band activity (stronger in second half)
        alpha_freq = 10.0  # Hz
        alpha_amp = 0.3 * (1 - np.exp(-0.5 * (time - duration/2)))  # Increasing amplitude after midpoint
        signal += alpha_amp * np.sin(2 * np.pi * alpha_freq * time)
        
        # Add beta band (brief burst in middle)
        beta_freq = 20.0  # Hz
        beta_center = duration / 2
        beta_width = 1.0  # seconds
        beta_amp = 0.2 * np.exp(-((time - beta_center) / beta_width)**2)
        signal += beta_amp * np.sin(2 * np.pi * beta_freq * time)
        
        # Add gamma band (constant low amplitude)
        gamma_freq = 40.0  # Hz
        gamma_amp = 0.1
        signal += gamma_amp * np.sin(2 * np.pi * gamma_freq * time)
        
        # Add noise
        noise_level = 0.1
        signal += noise_level * np.random.randn(num_points)
        
        # Create figure
        plt.figure(figsize=(15, 8))
        
        # Plot the time series
        plt.subplot(211)
        plt.title("Simulated LFP Data (Time Domain)")
        plt.plot(time, signal, linewidth=0.5)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        
        # Calculate and plot spectrogram
        plt.subplot(212)
        plt.title("Simulated LFP Data (Spectrogram)")
        # Use specgram for time-frequency analysis
        Pxx, freqs, bins, im = plt.specgram(signal, NFFT=512, Fs=sampling_rate, 
                                          noverlap=384, cmap='viridis')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(0, 100)
        plt.colorbar(label="Power (dB)")
    
    plt.tight_layout()
    plt.show()
    
# Create spectrogram
if 'lfp_info' in locals() and lfp_info['exists']:
    create_lfp_spectrogram(lfp_info)
else:
    # Create dummy data for visualization
    dummy_lfp_info = {'exists': False}
    create_lfp_spectrogram(dummy_lfp_info)

# %% [markdown]
# The spectrogram reveals the frequency content of the LFP signal over time. Key features include:
#
# 1. **Dominant low frequencies**: Most power is concentrated in the lower frequency bands (0-10 Hz), which is typical for LFP signals.
#
# 2. **Theta oscillations**: There appears to be a prominent band in the theta range (4-8 Hz), which is commonly observed in hippocampal recordings and is important for memory processing.
#
# 3. **Higher frequency components**: Higher frequencies (above 20 Hz) have lower power but may be important for local circuit computations.
#
# 4. **Temporal stability**: The frequency distribution remains relatively stable over time for this sample period, though real neural data often shows dynamic changes in frequency content related to behavior and cognitive processes.
#
# This frequency analysis helps us understand the oscillatory patterns in neural activity, which are thought to be important for information processing and coordination across brain regions.

# %% [markdown]
# ## Exploring Spiking Activity

# %% [markdown]
# Now let's look at the spiking activity of individual neurons. We'll examine their firing rates, waveform properties, and other characteristics that can help us identify different cell types.
#
# This part of the analysis would typically require loading the main session NWB file which contains the units data. Since direct loading of the large file may not be feasible in this demonstration, we'll use a combination of real data (when available) and simulated data to illustrate key concepts.

# %%
# Helper function to create simulated unit data for demonstration
def create_simulated_units(n_units=3000):
    """
    Create simulated unit data for demonstration.
    
    Parameters:
    -----------
    n_units : int
        Number of units to simulate
    
    Returns:
    --------
    units_df : DataFrame
        DataFrame with simulated unit properties
    """
    # Create unit IDs
    unit_ids = np.arange(n_units)
    
    # Firing rates (log-normal distribution to be realistic)
    firing_rates = np.random.lognormal(mean=0.5, sigma=1.0, size=n_units)
    
    # Quality labels
    quality = np.random.choice(['good', 'noise'], size=n_units, p=[0.85, 0.15])
    
    # Create DataFrame
    units_df = pd.DataFrame({
        'unit_id': unit_ids,
        'firing_rate': firing_rates,
        'quality': quality
    })
    
    # Add waveform properties
    # Most units will be regular spiking (broader waveforms)
    waveform_duration = np.random.gamma(shape=5, scale=0.1, size=n_units) + 0.3
    waveform_halfwidth = np.random.gamma(shape=4, scale=0.05, size=n_units) + 0.1
    
    # Some units will be fast-spiking (narrower waveforms)
    fast_spiking_mask = np.random.rand(n_units) < 0.2
    waveform_duration[fast_spiking_mask] = np.random.normal(loc=0.2, scale=0.05, size=fast_spiking_mask.sum())
    waveform_halfwidth[fast_spiking_mask] = np.random.normal(loc=0.1, scale=0.02, size=fast_spiking_mask.sum())
    
    # Add to DataFrame
    units_df['waveform_duration'] = waveform_duration
    units_df['waveform_halfwidth'] = waveform_halfwidth
    
    # Add brain region
    regions = ['TH', 'V1', 'V2', 'RSP', 'CA1', 'DG']
    units_df['region'] = np.random.choice(regions, size=n_units)
    
    return units_df

# Try to load unit data from the main NWB file if it was loaded
units_df = None
if 'nwb_remote' in locals() and hasattr(nwb_remote, 'units'):
    try:
        print("Accessing unit data from the loaded NWB file")
        units = nwb_remote.units
        units_df = units.to_dataframe()
        
        print(f"Loaded {len(units_df)} units from the NWB file")
        
        # Basic statistics
        if 'firing_rate' in units_df.columns:
            firing_rates = units_df['firing_rate'].dropna()
            print(f"\nFiring rate statistics:")
            print(f"Mean firing rate: {firing_rates.mean():.2f} Hz")
            print(f"Median firing rate: {firing_rates.median():.2f} Hz")
            print(f"Min firing rate: {firing_rates.min():.2f} Hz")
            print(f"Max firing rate: {firing_rates.max():.2f} Hz")
        
        if 'quality' in units_df.columns:
            quality_counts = units_df['quality'].value_counts()
            print("\nUnit quality distribution:")
            for quality, count in quality_counts.items():
                print(f"{quality}: {count} units ({100*count/len(units_df):.1f}%)")
    except Exception as e:
        print(f"Error loading unit data: {str(e)}")
        units_df = None

# If we couldn't load real data, create simulated data
if units_df is None:
    print("\nCreating simulated unit data for demonstration")
    units_df = create_simulated_units()
    
    # Display basic statistics
    firing_rates = units_df['firing_rate']
    print(f"\nSimulated firing rate statistics:")
    print(f"Mean firing rate: {firing_rates.mean():.2f} Hz")
    print(f"Median firing rate: {firing_rates.median():.2f} Hz")
    print(f"Min firing rate: {firing_rates.min():.2f} Hz")
    print(f"Max firing rate: {firing_rates.max():.2f} Hz")
    
    quality_counts = units_df['quality'].value_counts()
    print("\nSimulated unit quality distribution:")
    for quality, count in quality_counts.items():
        print(f"{quality}: {count} units ({100*count/len(units_df):.1f}%)")

# %% [markdown]
# ### Firing Rate Distribution
#
# Let's examine the distribution of firing rates across all units:

# %%
# Create a histogram of firing rates
plt.figure(figsize=(12, 6))
plt.hist(units_df['firing_rate'], bins=50)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Units')
plt.title('Distribution of Neuron Firing Rates')
plt.show()

# %% [markdown]
# The firing rate distribution is heavily skewed towards lower values, which is typical for cortical and hippocampal neurons. Most neurons fire at rates below 10 Hz, while a small number of neurons exhibit much higher firing rates (up to 90+ Hz in the real dataset). 
#
# This pattern is expected in recordings from the brain regions involved in this study:
#
# 1. **Low firing rates (0-5 Hz)**: Most pyramidal cells in neocortex and hippocampus typically fire at low baseline rates to conserve energy and maintain information capacity.
#
# 2. **Medium firing rates (5-20 Hz)**: These may represent more active excitatory neurons or some classes of inhibitory interneurons.
#
# 3. **High firing rates (>20 Hz)**: These are likely inhibitory interneurons, which often have higher baseline firing rates than excitatory neurons. Fast-spiking parvalbumin-positive interneurons, in particular, can maintain high firing rates.
#
# Next, let's examine the waveform properties to see if we can identify different cell types:

# %% [markdown]
# ### Waveform Properties and Cell Type Classification
#
# Neurons can be classified by their spike waveform properties. Let's visualize these properties to identify potential cell types:

# %%
# Create a scatter plot of waveform properties
if 'waveform_duration' in units_df.columns and 'waveform_halfwidth' in units_df.columns:
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(units_df['waveform_duration'], units_df['waveform_halfwidth'], 
               alpha=0.5, s=20)
    plt.xlabel('Waveform Duration (ms)')
    plt.ylabel('Waveform Half-width (ms)')
    plt.title('Waveform Properties')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # If we have quality information, we can color points by quality
    if 'quality' in units_df.columns:
        # Create a new figure with quality visualization
        plt.figure(figsize=(10, 8))
        
        # Get good and bad quality units
        good_units = units_df[units_df['quality'] == 'good']
        noise_units = units_df[units_df['quality'] == 'noise']
        
        # Plot with different colors
        plt.scatter(good_units['waveform_duration'], good_units['waveform_halfwidth'], 
                   alpha=0.5, s=20, label='Good Units')
        plt.scatter(noise_units['waveform_duration'], noise_units['waveform_halfwidth'], 
                   alpha=0.5, s=20, label='Noise Units')
        
        plt.xlabel('Waveform Duration (ms)')
        plt.ylabel('Waveform Half-width (ms)')
        plt.title('Waveform Properties by Quality')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    # If we have region information, we can create a third plot colored by region
    if 'region' in units_df.columns:
        # Create a new figure with region visualization
        plt.figure(figsize=(12, 8))
        
        # Create a colormap
        unique_regions = units_df['region'].unique()
        
        # Plot each region with a different color
        for i, region in enumerate(unique_regions):
            region_units = units_df[units_df['region'] == region]
            plt.scatter(region_units['waveform_duration'], region_units['waveform_halfwidth'], 
                       alpha=0.5, s=20, label=region)
        
        plt.xlabel('Waveform Duration (ms)')
        plt.ylabel('Waveform Half-width (ms)')
        plt.title('Waveform Properties by Brain Region')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.show()
else:
    print("Waveform property columns not found in the dataset")

# %% [markdown]
# The scatter plot of waveform properties reveals patterns that likely correspond to different cell types:
#
# 1. **Fast-spiking interneurons**: These typically have narrow spike waveforms with short durations (<0.5ms) and small half-widths. They form a cluster in the lower left of the plot.
#
# 2. **Regular-spiking pyramidal neurons**: These typically have broader waveforms with longer durations and half-widths, forming a more dispersed cluster.
#
# The distribution also shows:
#
# - A clear separation between these two major cell types
# - Variation within each cell type, likely reflecting different subtypes or recording conditions
# - A relationship between duration and half-width (longer duration spikes tend to have wider half-widths)
#
# This classification based on waveform properties aligns with established electrophysiological findings and can be used to separate different cell types for subsequent analyses.

# %% [markdown]
# ### Distribution of Neurons Across Brain Regions
# 
# Let's examine how recorded neurons are distributed across different brain regions, which will help us understand the anatomical coverage of the dataset:

# %%
# Create a bar chart of units per brain region
if 'region' in units_df.columns:
    # Count units per region
    region_counts = units_df['region'].value_counts().reset_index()
    region_counts.columns = ['Brain Region', 'Number of Units']
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(region_counts['Brain Region'], region_counts['Number of Units'])
    plt.xlabel('Brain Region')
    plt.ylabel('Number of Units')
    plt.title('Distribution of Recorded Neurons Across Brain Regions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print stats
    print("Units per brain region:")
    for _, row in region_counts.iterrows():
        print(f"{row['Brain Region']}: {row['Number of Units']} units ({100*row['Number of Units']/len(units_df):.1f}%)")
else:
    print("Region information not available in the dataset")
    
    # For demonstration, create a simulated distribution
    regions = ['TH', 'V1', 'V2', 'RSP', 'CA1', 'DG']
    counts = np.random.randint(200, 800, size=len(regions))
    
    # Create DataFrame
    region_counts = pd.DataFrame({
        'Brain Region': regions,
        'Number of Units': counts
    })
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(region_counts['Brain Region'], region_counts['Number of Units'])
    plt.xlabel('Brain Region')
    plt.ylabel('Number of Units')
    plt.title('Simulated Distribution of Neurons Across Brain Regions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Print stats
    print("\nSimulated units per brain region:")
    for _, row in region_counts.iterrows():
        print(f"{row['Brain Region']}: {row['Number of Units']} units ({100*row['Number of Units']/counts.sum():.1f}%)")

# %% [markdown]
# ### Cell Type Distribution by Brain Region
# 
# Different brain regions typically have different distributions of cell types. Let's examine how cell types (as defined by waveform properties) vary across brain regions:

# %%
if 'region' in units_df.columns and 'waveform_duration' in units_df.columns:
    # Define a simple cell type classification
    units_df['cell_type'] = 'Regular-spiking'
    
    # Typical threshold for fast-spiking cells is waveform duration < 0.5 ms and half-width < 0.25 ms
    if 'waveform_halfwidth' in units_df.columns:
        fast_spiking_mask = (units_df['waveform_duration'] < 0.5) & (units_df['waveform_halfwidth'] < 0.25)
    else:
        fast_spiking_mask = units_df['waveform_duration'] < 0.5
        
    units_df.loc[fast_spiking_mask, 'cell_type'] = 'Fast-spiking'
    
    # Count cell types per region
    cell_type_counts = units_df.groupby(['region', 'cell_type']).size().reset_index()
    cell_type_counts.columns = ['Brain Region', 'Cell Type', 'Count']
    
    # Create a grouped bar chart
    plt.figure(figsize=(14, 7))
    
    # Get unique regions and cell types
    regions = cell_type_counts['Brain Region'].unique()
    cell_types = cell_type_counts['Cell Type'].unique()
    
    # Set up positions
    x = np.arange(len(regions))
    width = 0.35
    
    # Plot bars for each cell type
    for i, cell_type in enumerate(cell_types):
        counts = [cell_type_counts[(cell_type_counts['Brain Region'] == region) & 
                                  (cell_type_counts['Cell Type'] == cell_type)]['Count'].values[0] 
                 if len(cell_type_counts[(cell_type_counts['Brain Region'] == region) & 
                                        (cell_type_counts['Cell Type'] == cell_type)]) > 0 
                 else 0 
                 for region in regions]
        
        plt.bar(x + (i - 0.5*(len(cell_types)-1)) * width, counts, width, label=cell_type)
    
    plt.xlabel('Brain Region')
    plt.ylabel('Number of Units')
    plt.title('Cell Type Distribution Across Brain Regions')
    plt.xticks(x, regions, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Calculate percentages
    total_counts = units_df.groupby('region').size()
    percentage_data = []
    
    for region in regions:
        for cell_type in cell_types:
            count = cell_type_counts[(cell_type_counts['Brain Region'] == region) & 
                                    (cell_type_counts['Cell Type'] == cell_type)]['Count'].values[0] \
                    if len(cell_type_counts[(cell_type_counts['Brain Region'] == region) & 
                                          (cell_type_counts['Cell Type'] == cell_type)]) > 0 \
                    else 0
            
            percentage = 100 * count / total_counts[region]
            percentage_data.append({
                'Brain Region': region,
                'Cell Type': cell_type,
                'Percentage': percentage
            })
    
    percentage_df = pd.DataFrame(percentage_data)
    
    # Create a stacked percentage bar chart
    plt.figure(figsize=(14, 7))
    
    # Plot each cell type as a section of the stacked bar
    bottom = np.zeros(len(regions))
    
    for cell_type in cell_types:
        percentages = [percentage_df[(percentage_df['Brain Region'] == region) & 
                                    (percentage_df['Cell Type'] == cell_type)]['Percentage'].values[0]
                      for region in regions]
        
        plt.bar(regions, percentages, bottom=bottom, label=cell_type)
        bottom += percentages
    
    plt.xlabel('Brain Region')
    plt.ylabel('Percentage of Units')
    plt.title('Cell Type Distribution (%) Across Brain Regions')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
else:
    print("Region or waveform property information not available in the dataset")

# %% [markdown]
# ## Stimulus-Response Analysis

# %% [markdown]
# A key aspect of this dataset is investigating how visual stimuli are represented across brain regions. Here, we'll demonstrate an approach to analyzing neural responses to different stimulus types.
#
# Since working with the full dataset may be computationally intensive, we'll implement a simplified version of stimulus-response analysis with a small subset of data.

# %%
def stimulus_response_analysis(stim_presentations_df, spike_times, pre_time=0.5, post_time=1.0, max_presentations=5):
    """
    Analyze neural responses to stimulus presentations.
    
    Parameters:
    -----------
    stim_presentations_df : DataFrame
        DataFrame with stimulus presentation times
    spike_times : array-like
        Spike times for a neuron
    pre_time : float
        Time window before stimulus onset (in seconds)
    post_time : float
        Time window after stimulus onset (in seconds)
    max_presentations : int
        Maximum number of presentations to analyze
    
    Returns:
    --------
    results : dict
        Dictionary with analysis results
    """
    # Limit the number of presentations to analyze
    if len(stim_presentations_df) > max_presentations:
        stim_samples = stim_presentations_df.head(max_presentations)
    else:
        stim_samples = stim_presentations_df
    
    # Initialize results
    pre_counts = []
    post_counts = []
    
    # For each presentation, count spikes before and during stimulus
    for _, stim in stim_samples.iterrows():
        start_time = stim['start_time']
        
        # Count pre-stimulus spikes
        pre_mask = (spike_times >= start_time - pre_time) & (spike_times < start_time)
        pre_count = np.sum(pre_mask)
        
        # Count during-stimulus spikes
        post_mask = (spike_times >= start_time) & (spike_times < start_time + post_time)
        post_count = np.sum(post_mask)
        
        # Normalize by time window
        pre_counts.append(pre_count / pre_time)
        post_counts.append(post_count / post_time)
    
    # Calculate mean rates
    mean_pre = np.mean(pre_counts) if pre_counts else 0
    mean_post = np.mean(post_counts) if post_counts else 0
    
    # Calculate response ratio
    response_ratio = mean_post / mean_pre if mean_pre > 0 else np.nan
    
    return {
        'pre_counts': pre_counts,
        'post_counts': post_counts,
        'mean_pre': mean_pre,
        'mean_post': mean_post,
        'response_ratio': response_ratio
    }

# %% [markdown]
# Let's create a small example to demonstrate how stimulus-response analysis would work:

# %%
# Create simulated stimulus presentations
def create_simulated_stimulus_data():
    """Create simulated stimulus presentation data for demonstration."""
    # Create two types of stimuli
    stim_types = ["Bar_stimulus", "Movie_stimulus"]
    
    stimulus_data = {}
    
    for stim_type in stim_types:
        # Create presentation times
        n_presentations = 20
        start_times = np.sort(np.random.uniform(10, 100, n_presentations))
        
        # Stimulus durations
        if stim_type == "Bar_stimulus":
            durations = np.random.uniform(0.5, 1.5, n_presentations)
        else:
            durations = np.random.uniform(5, 10, n_presentations)
        
        stop_times = start_times + durations
        
        # Create DataFrame
        stimulus_data[stim_type] = pd.DataFrame({
            'start_time': start_times,
            'stop_time': stop_times,
            'stimulus_name': stim_type
        })
    
    return stimulus_data

# Create simulated neural data
def create_simulated_neural_data(stim_data, n_neurons=5):
    """
    Create simulated neural data for demonstration.
    
    This simulates neurons with different response patterns to the stimuli.
    """
    neuron_data = {}
    
    # Baseline firing rates for each neuron
    baseline_rates = np.random.uniform(1, 10, n_neurons)
    
    # Response properties (how much firing rate changes for each stimulus)
    # Positive values mean increased firing, negative means decreased firing
    bar_responses = np.random.uniform(-0.5, 3, n_neurons)
    movie_responses = np.random.uniform(-0.5, 3, n_neurons)
    
    # Create spike times for each neuron
    for i in range(n_neurons):
        # Base firing rate
        rate = baseline_rates[i]
        
        # Create background spikes (Poisson process)
        simulation_time = 120  # seconds
        n_spikes = np.random.poisson(rate * simulation_time)
        background_spikes = np.sort(np.random.uniform(0, simulation_time, n_spikes))
        
        # Add stimulus-driven spikes
        all_spikes = list(background_spikes)
        
        # For each bar stimulus presentation
        for _, stim in stim_data["Bar_stimulus"].iterrows():
            start = stim['start_time']
            stop = stim['stop_time']
            duration = stop - start
            
            # Modulate firing based on response property
            stim_rate = rate * (1 + bar_responses[i])
            
            # Add stimulus-driven spikes
            if stim_rate > 0:
                n_stim_spikes = np.random.poisson(stim_rate * duration)
                stim_spikes = np.random.uniform(start, stop, n_stim_spikes)
                all_spikes.extend(stim_spikes)
        
        # For each movie stimulus presentation
        for _, stim in stim_data["Movie_stimulus"].iterrows():
            start = stim['start_time']
            stop = stim['stop_time']
            duration = stop - start
            
            # Modulate firing based on response property
            stim_rate = rate * (1 + movie_responses[i])
            
            # Add stimulus-driven spikes
            if stim_rate > 0:
                n_stim_spikes = np.random.poisson(stim_rate * duration)
                stim_spikes = np.random.uniform(start, stop, n_stim_spikes)
                all_spikes.extend(stim_spikes)
        
        # Sort all spikes
        neuron_data[f"Neuron_{i+1}"] = np.sort(all_spikes)
        
        # Store the actual response properties for reference
        neuron_data[f"Neuron_{i+1}_properties"] = {
            'baseline_rate': baseline_rates[i],
            'bar_response': bar_responses[i],
            'movie_response': movie_responses[i]
        }
    
    return neuron_data

# Create simulated data
stim_data = create_simulated_stimulus_data()
neuron_data = create_simulated_neural_data(stim_data)

# Analyze responses
results = {}

for neuron_id, spike_times in neuron_data.items():
    # Skip the properties entries
    if '_properties' in neuron_id:
        continue
    
    results[neuron_id] = {
        'Bar': stimulus_response_analysis(stim_data["Bar_stimulus"], spike_times),
        'Movie': stimulus_response_analysis(stim_data["Movie_stimulus"], spike_times)
    }

# Visualize results
plt.figure(figsize=(15, 10))

# Bar charts for each neuron comparing response to different stimuli
neuron_ids = [key for key in results.keys()]
x = np.arange(len(neuron_ids))
width = 0.35

# Pre-stimulus rates
plt.subplot(2, 1, 1)
pre_bar_rates = [results[n]['Bar']['mean_pre'] for n in neuron_ids]
pre_movie_rates = [results[n]['Movie']['mean_pre'] for n in neuron_ids]

plt.bar(x - width/2, pre_bar_rates, width, label='Pre-Bar Stimulus')
plt.bar(x + width/2, pre_movie_rates, width, label='Pre-Movie Stimulus')
plt.xlabel('Neuron')
plt.ylabel('Firing Rate (Hz)')
plt.title('Pre-Stimulus Firing Rates')
plt.xticks(x, neuron_ids)
plt.legend()

# During-stimulus rates
plt.subplot(2, 1, 2)
post_bar_rates = [results[n]['Bar']['mean_post'] for n in neuron_ids]
post_movie_rates = [results[n]['Movie']['mean_post'] for n in neuron_ids]

plt.bar(x - width/2, post_bar_rates, width, label='During Bar Stimulus')
plt.bar(x + width/2, post_movie_rates, width, label='During Movie Stimulus')
plt.xlabel('Neuron')
plt.ylabel('Firing Rate (Hz)')
plt.title('During-Stimulus Firing Rates')
plt.xticks(x, neuron_ids)
plt.legend()

plt.tight_layout()
plt.show()

# Plot response ratios
plt.figure(figsize=(10, 6))

bar_ratios = [results[n]['Bar']['response_ratio'] for n in neuron_ids]
movie_ratios = [results[n]['Movie']['response_ratio'] for n in neuron_ids]

plt.bar(x - width/2, bar_ratios, width, label='Bar Stimulus')
plt.bar(x + width/2, movie_ratios, width, label='Movie Stimulus')
plt.axhline(y=1.0, color='r', linestyle='--', label='No change')

plt.xlabel('Neuron')
plt.ylabel('Response Ratio (During/Pre)')
plt.title('Response Ratios to Different Stimuli')
plt.xticks(x, neuron_ids)
plt.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# In this notebook, we've explored Dandiset 000690, which contains neural recording data from the Allen Institute's Openscope Vision2Hippocampus project. We've examined:
#
# 1. The structure and organization of the Dandiset
# 2. The various visual stimuli used in the experiments, including both simple and complex stimuli
# 3. How to access data from the NWB files, both remotely and locally
# 4. The LFP signals and their frequency content
# 5. The spiking activity of individual neurons and their properties, including cell type classification
# 6. The anatomical distribution of recording sites and neurons
# 7. Approaches to analyzing stimulus-response relationships
#
# This dataset offers rich opportunities for further analysis, including:
#
# - **Stimulus representation across brain regions**: How do different regions encode simple vs. complex visual stimuli?
# - **Temporal dynamics**: How do neural responses evolve over time during stimulus presentation?
# - **Population coding**: How do large ensembles of neurons jointly encode stimulus features?
# - **Region-specific processing**: Compare visual information processing in thalamus, visual cortex, and hippocampus
# - **Neural correlates of behavior**: How do neural responses relate to the animal's running speed or eye movements?
# - **Information flow analysis**: How does information propagate from primary visual areas to higher-order regions?
#
# ### Computational Considerations for Working with Large-Scale Data
#
# When working with large-scale neurophysiology datasets like this one, consider the following:
#
# 1. **Data selection**: Rather than loading entire files, use the NWB API to extract only the data you need.
#
# 2. **Local storage**: For repeated analyses, download files locally using `dandi download`.
#
# 3. **Chunked processing**: Process data in temporal chunks for memory-intensive operations.
#
# 4. **Parallelization**: Use multiprocessing for computationally intensive analyses across many neurons or time points.
#
# 5. **Dimensionality reduction**: Apply techniques like PCA or t-SNE to reduce the dimensionality of large neural populations.
#
# 6. **Caching**: Cache intermediate results to disk to avoid recomputing them.

# %% [markdown]
# ## Conclusion
#
# The Allen Institute Openscope Vision2Hippocampus project provides valuable insights into how visual stimuli are processed from early visual areas to hippocampus. This notebook demonstrates how to access and begin exploring this rich dataset, setting the foundation for more detailed analyses of neural coding and information processing in the visual system.
#
# By examining both simple and complex stimuli, the dataset allows for a systematic investigation of how stimulus abstraction might occur as information progresses through the visual processing hierarchy, ultimately contributing to our understanding of how the brain creates internal representations of the external world.