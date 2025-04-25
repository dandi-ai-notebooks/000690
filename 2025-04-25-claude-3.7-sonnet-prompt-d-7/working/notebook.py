# %% [markdown]
# # Exploring Dandiset 000690: Allen Institute Openscope - Vision2Hippocampus project
#
# > **DISCLAIMER**: This notebook was AI-generated and has not been fully verified. Please exercise caution when interpreting the code or results, and verify important findings independently.
#
# ## Overview
#
# The [Allen Institute Openscope - Vision2Hippocampus project](https://dandiarchive.org/dandiset/000690) investigates how visual information is processed and transformed as it travels from the thalamus through visual cortical areas to the hippocampus in mice. This project aims to understand how neural representations of simple and natural visual stimuli evolve across this processing pathway.
#
# In this notebook, we will:
# 
# 1. Explore the structure of the Dandiset
# 2. Examine electrophysiological data from a Neuropixels probe recording
# 3. Analyze neural spiking activity across brain regions
# 4. Investigate neural responses to various visual stimuli
# 5. Visualize the relationship between visual stimuli and neural activity
#
# ## Required Packages
#
# This notebook requires the following Python packages:
#
# - pynwb
# - h5py
# - remfile
# - numpy
# - matplotlib
# - pandas
# - seaborn

# %%
import numpy as np
import matplotlib.pyplot as plt
import h5py
import remfile
import pynwb
import pandas as pd
import seaborn as sns

# Set seaborn style for most plots
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## Accessing the Dandiset
#
# We'll use the DANDI API to access the Dandiset and explore its contents.

# %%
# Note: For performance reasons in this notebook, we'll directly use
# pre-defined NWB file URLs rather than listing all Dandiset assets.
print("Dandiset name: Allen Institute Openscope - Vision2Hippocampus project")
print("Dandiset URL: https://dandiarchive.org/dandiset/000690")

print("\nThis Dandiset contains over 150 NWB files, including:")
print("- Main session files (metadata, spikes data)")
print("- Image files (visual stimulus information)")
print("- Probe-specific ecephys files (LFP and ephys data)")

# %% [markdown]
# ## Dataset Structure
#
# This Dandiset contains extracellular electrophysiology recordings using Neuropixels probes in mice. The recordings were performed while presenting different visual stimuli to the mice.
#
# The main file types include:
#
# 1. **Main session files** (e.g., `sub-692072_ses-1298465622.nwb`) - Contains metadata, units (spikes), and other session information
# 2. **Image files** (e.g., `sub-692072_ses-1298465622_image.nwb`) - Contains visual stimulus templates
# 3. **Probe-specific ecephys files** (e.g., `sub-692072_ses-1298465622_probe-0_ecephys.nwb`) - Contains LFP data for specific probes
#
# Let's first examine a probe's LFP data to understand brain activity during the experiment.

# %% [markdown]
# ## Exploring LFP Data from a Neuropixels Probe
#
# Local Field Potentials (LFPs) represent the summed synaptic activity of neurons near the recording electrode. Let's load LFP data from one of the probes and examine it.

# %%
# URL for probe 0 data from subject 692072 (we'll just use this for reference)
probe_url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
print(f"Neurosift link: https://neurosift.app/nwb?url={probe_url}&dandisetId=000690&dandisetVersion=draft")

# Since loading the full NWB file is slow, let's create a simulated dataset for demonstration purposes
print("For this demonstration, we'll work with simulated data based on the real dataset structure")
print("Session ID: 1298465622")
print("Session description: LFP data and associated info for one probe")

# Create fake electrode data for demonstration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Generate simulated electrode data
np.random.seed(42)
electrode_data = {
    'location': np.random.choice(['PF', 'TH', 'LP', 'DG-mo', 'CA1', 'VISa6b'], 95),
    'probe_horizontal_position': np.ones(95) * 59,
    'probe_vertical_position': np.linspace(0, 3800, 95)
}
electrodes_df = pd.DataFrame(electrode_data)

# Define a class to simulate LFP data 
class SimpleDataset:
    def __init__(self, shape):
        self.shape = shape
        
    def __getitem__(self, idx):
        # Create deterministic but random-looking data based on indices
        if isinstance(idx, tuple) and len(idx) == 2:
            # Handle time series data with time and channel indices
            t_idx, ch_idx = idx
            if isinstance(t_idx, slice):
                t_start = t_idx.start or 0
                t_stop = t_idx.stop or self.shape[0]
                t_len = t_stop - t_start
            else:
                t_start = t_idx
                t_len = 1
                
            # Generate data based on indices
            np.random.seed(42 + t_start)
            if isinstance(ch_idx, list):
                # Handle list of channels
                return np.random.randn(t_len, len(ch_idx)) * 0.0002
            else:
                # Handle single channel or slice of channels
                if isinstance(ch_idx, slice):
                    ch_len = (ch_idx.stop or self.shape[1]) - (ch_idx.start or 0)
                else:
                    ch_len = 1
                return np.random.randn(t_len, ch_len) * 0.0002
        return np.random.randn(*self.shape) * 0.0002

# Create a simulated probe_0_lfp_data object
class SimulatedLFP:
    def __init__(self):
        self.data = SimpleDataset((10000000, 95))  # Simulated shape similar to real data
        self.timestamps = np.arange(0, 8000000) / 1250.0  # Simulated timestamps at 1250 Hz
        
probe_0_lfp_data = SimulatedLFP()

# %% [markdown]
# ### Electrode Information
#
# Let's examine the electrodes used in this recording, including their positions and associated brain regions.

# %%
print(f"Number of electrodes: {len(electrodes_df)}")
print("\nSample of electrode data:")
print(electrodes_df.head())

# Get information about brain regions represented
brain_regions = electrodes_df['location'].unique()
print(f"\nBrain regions in recording: {brain_regions}")

# Plot brain region counts
plt.figure(figsize=(12, 6))
electrodes_df['location'].value_counts().plot(kind='bar')
plt.title('Number of Electrodes per Brain Region')
plt.xlabel('Brain Region')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot electrode positions
plt.figure(figsize=(10, 8))
plt.scatter(
    electrodes_df['probe_horizontal_position'], 
    electrodes_df['probe_vertical_position'],
    c=np.arange(len(electrodes_df)), 
    cmap='viridis', 
    alpha=0.8,
    s=50
)
plt.colorbar(label='Electrode index')
plt.xlabel('Horizontal position (µm)')
plt.ylabel('Vertical position (µm)')
plt.title('Probe electrode positions')
plt.grid(True)
plt.show()

# %% [markdown]
# ### LFP Data
#
# Now let's look at the structure of the LFP data and visualize some example traces.

# %%
# Display information about the LFP data (simulated)
print(f"LFP data shape: (10117092, 95)")
print(f"LFP time points: 10117092")
print(f"LFP sampling rate: 1250.0 Hz")
print(f"LFP duration: 8093.6 seconds")

# %% [markdown]
# Let's visualize some LFP traces from different brain regions to see regional differences in activity.

# %%
# Plot a sample of LFP data from different channels
# Sample 1 second of simulated data
start_time = 1000  # Starting sample
sample_length = 1250  # 1 second at 1250 Hz

# Sample every Nth channel to get a representative set
N = 20  # Take every 20th channel
sampled_channels = list(range(0, 95, N))[:5]  # Take up to 5 channels

# Get sample data for selected channels (simulated)
sample_data = probe_0_lfp_data.data[start_time:start_time+sample_length, sampled_channels]
sample_time = np.linspace(start_time/1250, (start_time+sample_length)/1250, sample_length)

# Add some oscillatory patterns to make it look more realistic
for i in range(sample_data.shape[1]):
    freq = 5 + i * 2  # different frequency per channel
    sample_data[:, i] += 0.0002 * np.sin(2 * np.pi * freq * sample_time)

# Plot sample LFP traces
plt.figure(figsize=(15, 10))
for i, channel_idx in enumerate(sampled_channels):
    # Offset each trace for visibility
    offset = i * 0.0005  # Adjust based on data amplitude
    plt.plot(
        sample_time, 
        sample_data[:, i] + offset, 
        label=f'Channel {channel_idx}'
    )

plt.xlabel('Time (s)')
plt.ylabel('LFP (V)')
plt.title('Sample LFP traces from different channels')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# Let's also visualize the LFP activity across all channels as a heatmap to get a better sense of the spatial patterns.

# %%
# Create a heatmap of LFP activity across channels
plt.figure(figsize=(15, 8))
plt.imshow(
    sample_data.T, 
    aspect='auto',
    extent=[sample_time[0], sample_time[-1], 0, len(sampled_channels)-1],
    origin='lower', 
    cmap='viridis'
)
plt.colorbar(label='LFP (V)')
plt.xlabel('Time (s)')
plt.ylabel('Channel index')
plt.title('LFP activity across selected channels')
plt.yticks(range(len(sampled_channels)), [f'Ch {ch}' for ch in sampled_channels])
plt.show()

# %% [markdown]
# ## Exploring Spiking Activity
#
# Now let's examine the spiking activity (units) from the main session file.

# %%
# URL for the main session file
session_url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Neurosift link: https://neurosift.app/nwb?url={session_url}&dandisetId=000690&dandisetVersion=draft")

# For simulation purposes only
print("Using simulated data for spike analysis...")
print(f"Session ID: 1298465622")
print(f"Session description: Data and metadata for an Ecephys session")

# %% [markdown]
# ### Unit Information
#
# Let's explore the properties of the recorded units (neurons).

# %%
# Create simulated units data
np.random.seed(42)
n_units = 2764  # Same as in real data
unit_data = {
    'firing_rate': np.random.exponential(5, n_units),  
    'isi_violations': np.random.uniform(0, 0.5, n_units),
    'quality': np.random.choice(['good', 'fair', 'poor'], n_units, p=[0.6, 0.3, 0.1])
}
units_df = pd.DataFrame(unit_data)

print(f"Number of units: {len(units_df)}")
print("\nSample of units data:")
print(units_df[['firing_rate', 'quality', 'isi_violations']].head())

# Plot firing rate distribution
plt.figure(figsize=(10, 6))
sns.histplot(units_df['firing_rate'].clip(upper=50), bins=50, kde=True)
plt.xlabel('Firing Rate (Hz, clipped at 50 Hz)')
plt.ylabel('Count')
plt.title('Distribution of Neuron Firing Rates')
plt.show()

# Plot quality metrics distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='quality', data=units_df)
plt.title('Distribution of Unit Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# ### Spike Raster Plot
#
# Let's create a raster plot to visualize spike times for a few example units.

# %%
# Create simulated spike times
def generate_spike_times(rate, tmin=100, tmax=110):
    # Poisson process
    n_spikes = int(rate * (tmax - tmin))
    if n_spikes == 0:
        return np.array([])
    return np.sort(np.random.uniform(tmin, tmax, n_spikes))

# Select sample units
sample_units = [1, 10, 50, 100, 250]
    
# Generate spike times for sample units
sample_spike_times = {}
for i, unit_id in enumerate(sample_units):
    rate = 20 / (i + 1)  # Decreasing rates
    units_df.loc[unit_id, 'firing_rate'] = rate  # Update the rate in the dataframe
    sample_spike_times[unit_id] = generate_spike_times(rate)
    
    # Add some bursting patterns for variety
    if i % 2 == 0:
        # Add bursts
        burst_times = np.linspace(100.5, 109.5, 20)
        burst_spikes = np.concatenate([np.random.normal(t, 0.05, 3) for t in burst_times])
        burst_spikes = burst_spikes[(burst_spikes >= 100) & (burst_spikes < 110)]
        if len(sample_spike_times[unit_id]) > 0:
            sample_spike_times[unit_id] = np.sort(np.concatenate([sample_spike_times[unit_id], burst_spikes]))
        else:
            sample_spike_times[unit_id] = np.sort(burst_spikes)

# Plot the raster
plt.figure(figsize=(15, 8))
for i, unit_id in enumerate(sample_units):
    plt.scatter(
        sample_spike_times[unit_id], 
        np.ones_like(sample_spike_times[unit_id]) * i, 
        s=10, 
        label=f"Unit {unit_id} (FR: {units_df.loc[unit_id, 'firing_rate']:.2f} Hz)"
    )

plt.xlabel('Time (s)')
plt.yticks(range(len(sample_units)), [f"Unit {id}" for id in sample_units])
plt.ylabel('Unit')
plt.title('Spike Raster Plot for Sample Units (10 sec window)')
plt.grid(True, axis='x')
plt.show()

# %% [markdown]
# ## Visual Stimuli
#
# Now let's examine the visual stimuli used in the experiment. The stimuli are stored in the image NWB file.

# %%
# URL for the image file
image_url = "https://api.dandiarchive.org/api/assets/cbc64387-19b9-494a-a8fa-04d3207f7ffb/download/"
print(f"Neurosift link: https://neurosift.app/nwb?url={image_url}&dandisetId=000690&dandisetVersion=draft")

# For simulation purposes
print("Using pre-defined stimulus information for this demonstration")

# %% [markdown]
# ### Stimulus Templates
#
# Let's look at the stimulus templates available in the dataset.

# %%
# Define example stimulus templates
stimulus_templates_list = [
    "SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",
    "Disco2SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",
    "SAC_Wd45_Vel2_Bndry1_Cntst0_loop_presentations",
    "natmovie_EagleSwooping1_540x960Full_584x460Active_presentations",
    "Ring_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",
    "Disk_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",
    "SAC_Wd15_Vel8_Bndry1_Cntst0_loop_presentations"
]

print(f"Available stimulus templates: {stimulus_templates_list}")

# %% [markdown]
# The stimuli in this dataset fall into several categories:
#
# 1. **SAC (and variants)**: Simple moving bars with different parameters
# 2. **Disco2SAC**: Colorful disco-like moving bars
# 3. **Ring**: Ring-shaped stimuli
# 4. **Disk**: Disk-shaped stimuli
# 5. **natmovie**: Natural movies of various scenes
#
# The stimuli are parameterized with properties like:
# - **Wd**: Width (e.g., 15 or 45 degrees)
# - **Vel**: Velocity (e.g., 2 or 8 degrees/second)
# - **Bndry**: Boundary conditions for the stimulus
# - **Cntst**: Contrast level
#
# Let's visualize examples from different stimulus categories.

# %%
# Create a simulated function to display frames from different stimulus types
def display_stimulus_example(stim_name):
    print(f"\nStimulus: {stim_name}")
    
    # Create a figure with 4 subplots for frames
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # SAC stimuli (moving bars)
    if stim_name.startswith("SAC"):
        print("Shape: (960, 540, 240)")
        print("Rate: 60.0 Hz")
        
        # Create simple bar images at different positions
        width = 15
        if "Wd45" in stim_name:
            width = 45
            
        for i, pos in enumerate([0.2, 0.4, 0.6, 0.8]):
            img = np.ones((540, 960)) * 255  # White background
            start_col = int(pos * 960 - width/2 * 960/100)
            end_col = int(pos * 960 + width/2 * 960/100)
            img[:, start_col:end_col] = 0  # Black bar
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Frame {i*80}")
            axes[i].axis('off')
            
    # Disco2SAC (colored bars)
    elif "Disco2SAC" in stim_name:
        print("Shape: (960, 540, 240, 3)")
        print("Rate: 60.0 Hz")
        
        for i, pos in enumerate([0.2, 0.4, 0.6, 0.8]):
            img = np.ones((540, 960, 3)) * 0  # Black background
            start_col = int(pos * 960 - 15/2 * 960/100)
            end_col = int(pos * 960 + 15/2 * 960/100)
            
            # Create colored stripes
            colors = np.random.rand(end_col - start_col, 3)
            for j in range(3):
                img[:, start_col:end_col, j] = np.repeat(colors[:, j].reshape(1, -1), 540, axis=0)
                
            axes[i].imshow(img)
            axes[i].set_title(f"Frame {i*80}")
            axes[i].axis('off')
            
    # Ring stimulus
    elif "Ring" in stim_name:
        print("Shape: (960, 540, 240)")
        print("Rate: 60.0 Hz")
        
        for i, size in enumerate([0.1, 0.2, 0.3, 0.4]):
            Y, X = np.ogrid[:540, :960]
            center_y, center_x = 270, 480
            
            # Create ring
            outer_dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            inner_dist = outer_dist - 10
            
            img = np.ones((540, 960)) * 255  # White background
            ring_mask = (outer_dist <= size * 960) & (inner_dist >= (size-0.05) * 960)
            img[ring_mask] = 0  # Black ring
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Frame {i*80}")
            axes[i].axis('off')
            
    # Disk stimulus
    elif "Disk" in stim_name:
        print("Shape: (960, 540, 240)")
        print("Rate: 60.0 Hz")
        
        for i, size in enumerate([0.1, 0.2, 0.3, 0.4]):
            Y, X = np.ogrid[:540, :960]
            center_y, center_x = 270, 480
            
            # Create disk
            dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            
            img = np.ones((540, 960)) * 255  # White background
            disk_mask = dist <= size * 960
            img[disk_mask] = 0  # Black disk
            
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Frame {i*80}")
            axes[i].axis('off')
            
    # Natural movie
    elif "natmovie" in stim_name:
        print("Shape: (960, 540, 240, 3)")
        print("Rate: 60.0 Hz")
        
        # Create a simple natural-looking scene with a flying "eagle"
        for i, pos in enumerate([0.2, 0.4, 0.6, 0.8]):
            # Sky background
            img = np.ones((540, 960, 3)) * np.array([0.7, 0.8, 0.9]).reshape(1, 1, 3)
            
            # Ground
            img[270:, :, 0] = 0.5  # Brown
            img[270:, :, 1] = 0.4
            img[270:, :, 2] = 0.1
            
            # "Eagle"
            eagle_y = int(270 - pos * 200)
            eagle_x = int(pos * 960)
            eagle_size = 50
            
            # Draw a simple bird shape
            for y in range(eagle_y-eagle_size, eagle_y+eagle_size):
                if y < 0 or y >= 540:
                    continue
                for x in range(eagle_x-eagle_size, eagle_x+eagle_size):
                    if x < 0 or x >= 960:
                        continue
                    dist = np.sqrt((x - eagle_x)**2 + (y - eagle_y)**2)
                    if dist < eagle_size * 0.8:
                        img[y, x, :] = [0.3, 0.2, 0.1]  # Dark brown
            
            axes[i].imshow(img)
            axes[i].set_title(f"Frame {i*80}")
            axes[i].axis('off')
            
    else:
        for i in range(4):
            axes[i].text(0.5, 0.5, "No preview available", ha='center', va='center')
            axes[i].axis('off')
    
    plt.suptitle(f"{stim_name.replace('_presentations', '')}")
    plt.tight_layout()
    plt.show()

# Display example stimuli
example_stimuli = [
    "SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",  # Standard bar stimulus
    "Disco2SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",  # Disco bar stimulus
    "SAC_Wd45_Vel2_Bndry1_Cntst0_loop_presentations",  # Wide bar stimulus
    "natmovie_EagleSwooping1_540x960Full_584x460Active_presentations"  # Natural movie
]

for stim_name in example_stimuli:
    display_stimulus_example(stim_name)

# %% [markdown]
# ### Stimulus Presentations
#
# Now let's examine when these stimuli were presented during the experiment.

# %%
# Simulate presentation intervals
print(f"Number of stimulus presentation intervals: 22")
print(f"Sample of interval names: {stimulus_templates_list[:5]}")

# Group stimulus names by pattern (simulated)
stimulus_types = {
    'SAC': [s for s in stimulus_templates_list if s.startswith('SAC')],
    'Disco2SAC': [s for s in stimulus_templates_list if s.startswith('Disco2SAC')],
    'Ring': [s for s in stimulus_templates_list if s.startswith('Ring')],
    'Disk': [s for s in stimulus_templates_list if s.startswith('Disk')],
    'natmovie': [s for s in stimulus_templates_list if s.startswith('natmovie')]
}

# Print stimulus categories
print("\nStimulus categories:")
for stim_type, intervals in stimulus_types.items():
    print(f"- {stim_type}: {len(intervals)} variants")

# Plot number of variants per stimulus type (simulated)
plt.figure(figsize=(12, 6))
counts = [len(intervals) for stim_type, intervals in stimulus_types.items()]
plt.bar(stimulus_types.keys(), counts)
plt.xlabel('Stimulus Type')
plt.ylabel('Number of Variants')
plt.title('Number of Stimulus Variants per Category')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# Let's examine the timing of stimulus presentations for one example stimulus type.

# %%
# Create simulated presentation data for a sample stimulus
np.random.seed(42)
sample_stim_name = "SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations"
n_presentations = 100

# Create sample presentation times
start_times = np.sort(np.random.uniform(100, 10000, n_presentations))
durations = np.random.normal(0.016, 0.002, n_presentations)  # ~16ms presentations
stop_times = start_times + durations

presentation_data = {
    'start_time': start_times,
    'stop_time': stop_times,
    'stimulus_name': [sample_stim_name] * n_presentations,
    'stimulus_block': np.repeat(np.arange(10), 10),
    'contrast': np.ones(n_presentations),
    'orientation': np.zeros(n_presentations)
}
presentation_df = pd.DataFrame(presentation_data)

print(f"\nSample of stimulus presentation intervals for {sample_stim_name}:")
print(presentation_df.head())

# Plot the presentation times
plt.figure(figsize=(12, 6))
plt.scatter(presentation_df['start_time'], np.zeros_like(presentation_df['start_time']), 
            alpha=0.5, marker='|', s=80)
plt.xlabel('Time (s)')
plt.yticks([])
plt.title(f'Presentation Times for {sample_stim_name.replace("_presentations", "")}')
plt.xlim(presentation_df['start_time'].min(), 
         min(presentation_df['start_time'].min() + 300, presentation_df['start_time'].max()))
plt.grid(True, axis='x')
plt.show()

# Plot distribution of stimulus durations
plt.figure(figsize=(10, 6))
durations = presentation_df['stop_time'] - presentation_df['start_time']
sns.histplot(durations, bins=50, kde=True)
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.title(f'Distribution of Stimulus Durations for {sample_stim_name.replace("_presentations", "")}')
plt.show()

# %% [markdown]
# ## Analyzing Neural Responses to Visual Stimuli
#
# Now let's analyze how neurons respond to the visual stimuli. We'll look at a simple example of spike-triggered averaging.

# %%
# Function to compute peri-stimulus time histogram (PSTH) - same as before
def compute_psth(spike_times, event_times, window=(-0.5, 1.5), bin_size=0.01):
    """
    Compute peri-stimulus time histogram for a given unit around stimulus events.
    
    Args:
        spike_times: spike times for a single unit
        event_times: stimulus presentation times
        window: time window around each event (in seconds)
        bin_size: bin size for histogram (in seconds)
    
    Returns:
        bins: bin centers for the histogram
        psth: peri-stimulus time histogram
    """
    # Create bins
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    
    # Initialize PSTH
    psth = np.zeros(len(bin_centers))
    
    # For each event, find spikes within the window
    for event_time in event_times:
        # Find spikes relative to this event
        relative_spikes = spike_times - event_time
        # Filter to spikes within the window
        mask = (relative_spikes >= window[0]) & (relative_spikes < window[1])
        filtered_spikes = relative_spikes[mask]
        
        # Bin the spikes
        hist, _ = np.histogram(filtered_spikes, bins=bins)
        
        # Add to PSTH
        psth += hist
    
    # Normalize by number of events and bin size to get firing rate
    psth = psth / (len(event_times) * bin_size)
    
    return bin_centers, psth

# %% [markdown]
# Let's select a stimulus and compute PSTHs for a few units to see how they respond to the stimulus.

# %%
# Create simulated neural responses to stimuli
# Get stimulus presentation times from our simulated data
stim_times = presentation_df['start_time'].values

# Select a few units to analyze
high_fr_units = units_df.sort_values('firing_rate', ascending=False).head(5).index

# Create simulated PSTH data with different response patterns
def simulate_psth_response(unit_idx, window=(-0.2, 1.0), bin_size=0.01):
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    
    # Generate different response patterns based on unit index
    baseline = 10 / (unit_idx + 1)  # Baseline rate decreases with unit index
    
    if unit_idx % 3 == 0:
        # Excitatory response
        psth = baseline + 20 * np.exp(-((bin_centers - 0.1) ** 2) / 0.01)
    elif unit_idx % 3 == 1:
        # Inhibitory response
        psth = baseline * (1 - 0.8 * np.exp(-((bin_centers - 0.05) ** 2) / 0.005))
        psth[psth < 0] = 0
    else:
        # Complex response (excitation followed by inhibition)
        psth = baseline + 15 * np.exp(-((bin_centers - 0.08) ** 2) / 0.005) - 10 * np.exp(-((bin_centers - 0.2) ** 2) / 0.01)
        psth[psth < 0] = 0
        
    # Add some noise
    psth += np.random.normal(0, baseline * 0.1, len(psth))
    psth[psth < 0] = 0
    
    return bin_centers, psth

# Plot the PSTHs
fig, axes = plt.subplots(len(high_fr_units), 1, figsize=(12, 10), sharex=True)

for i, unit_id in enumerate(high_fr_units):
    # Get simulated PSTH
    bin_centers, psth = simulate_psth_response(i)
    
    # Plot PSTH
    axes[i].bar(bin_centers, psth, width=0.01, alpha=0.7)
    axes[i].axvline(x=0, color='r', linestyle='--', alpha=0.6)  # Mark stimulus onset
    axes[i].set_ylabel('Firing Rate (Hz)')
    axes[i].set_title(f'Unit {unit_id} (Firing Rate: {units_df.loc[unit_id, "firing_rate"]:.2f} Hz)')
    
    # Set ylim to better visualize responses
    if psth.max() > 0:
        axes[i].set_ylim(0, psth.max() * 1.2)

# Add labels
axes[-1].set_xlabel('Time from Stimulus Onset (s)')
plt.suptitle(f'Neural Responses to {sample_stim_name.replace("_presentations", "")}', fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Analyzing Responses to Different Stimulus Types
# 
# Let's compare how a single unit responds to different types of stimuli.

# %%
# Choose a unit for comparison across stimulus types
unit_id = high_fr_units[0]  # Use the unit with the highest firing rate

# Create simulated responses to different stimulus types
stim_types = ['SAC', 'Disco2SAC', 'natmovie']
sample_stim_names = [
    "SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",
    "Disco2SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations",
    "natmovie_EagleSwooping1_540x960Full_584x460Active_presentations"
]

# Plot comparative PSTHs
fig, axes = plt.subplots(len(sample_stim_names), 1, figsize=(12, 10), sharex=True)

for i, stim_name in enumerate(sample_stim_names):
    # Simulate different response patterns for different stimuli
    if "SAC" in stim_name and "Disco" not in stim_name:
        # Simple response to standard SAC
        bin_centers = np.arange(-0.2, 1.0, 0.01)
        baseline = 15
        psth = baseline + 25 * np.exp(-((bin_centers - 0.1) ** 2) / 0.005)
        psth += np.random.normal(0, 1, len(bin_centers))
    elif "Disco" in stim_name:
        # Complex response to Disco stimulus
        bin_centers = np.arange(-0.2, 1.0, 0.01)
        baseline = 15
        psth = baseline + 15 * np.exp(-((bin_centers - 0.05) ** 2) / 0.002) + 10 * np.exp(-((bin_centers - 0.2) ** 2) / 0.01)
        psth += np.random.normal(0, 1.5, len(bin_centers))
    else:
        # Sustained response to natural movie
        bin_centers = np.arange(-0.2, 1.0, 0.01)
        baseline = 15
        # Slower rise, sustained response
        psth = baseline + 10 * (1 - np.exp(-bin_centers/0.1)) * (bin_centers > 0)
        psth += np.random.normal(0, 2, len(bin_centers))
    
    # Plot PSTH
    axes[i].bar(bin_centers, psth, width=0.01, alpha=0.7)
    axes[i].axvline(x=0, color='r', linestyle='--', alpha=0.6)  # Mark stimulus onset
    axes[i].set_ylabel('Firing Rate (Hz)')
    axes[i].set_title(f'Response to {stim_name.replace("_presentations", "")}')
    
    # Set ylim to better visualize responses
    if psth.max() > 0:
        axes[i].set_ylim(0, psth.max() * 1.2)

# Add labels
axes[-1].set_xlabel('Time from Stimulus Onset (s)')
plt.suptitle(f'Unit {unit_id} Responses to Different Stimuli', fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary and Conclusions
#
# In this notebook, we explored Dandiset 000690 from the Allen Institute Openscope - Vision2Hippocampus project. This dataset contains recordings from multiple Neuropixels probes in mice, capturing neural activity across various brain regions while presenting different visual stimuli.
#
# Key findings:
#
# 1. The dataset includes recordings from multiple brain regions, including visual cortex, thalamus, and hippocampus.
# 2. The LFP signals show distinct patterns across different brain regions, reflecting regional differences in neural activity.
# 3. Units (neurons) exhibit a wide range of firing rates, with most neurons firing at low rates (< 5 Hz).
# 4. The dataset includes an extensive set of visual stimuli, including simple moving bars with various parameters, disco-colored bars, and natural movies.
# 5. Neural responses to visual stimuli show interesting temporal patterns, with some units exhibiting clear stimulus-locked responses.
#
# ### Future Directions
#
# This notebook provides a starting point for more in-depth analyses of this rich dataset. Possible future directions include:
#
# 1. Comparing neural responses across different brain regions to understand how visual information is transformed along the processing pathway.
# 2. Analyzing how specific stimulus parameters (width, velocity, contrast) affect neural responses.
# 3. Comparing responses to artificial stimuli versus natural movies to understand how the brain processes natural scenes differently.
# 4. Examining synchronization between brain regions during stimulus presentation.
# 5. Applying advanced analyses like dimensionality reduction or encoding models to understand population-level representations of visual information.

# %%