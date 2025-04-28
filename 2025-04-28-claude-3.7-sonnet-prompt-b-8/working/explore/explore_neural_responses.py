# Script to explore the relationship between neural activity and visual stimuli presentations
# This script analyzes how neurons respond to different visual stimulus presentations

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from scipy import stats

# Configure matplotlib to save rather than display
import seaborn as sns
sns.set_theme()
plt.rcParams['figure.figsize'] = (12, 8)

print("Loading NWB files...")
# Load the ecephys data
url_ecephys = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file_ecephys = remfile.File(url_ecephys)
h5_file_ecephys = h5py.File(remote_file_ecephys)
io_ecephys = pynwb.NWBHDF5IO(file=h5_file_ecephys)
nwb_ecephys = io_ecephys.read()

# Load the session data to get stimulus presentation times
url_session = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file_session = remfile.File(url_session)
h5_file_session = h5py.File(remote_file_session)
io_session = pynwb.NWBHDF5IO(file=h5_file_session)
nwb_session = io_session.read()

print("Files loaded successfully")

# Get units information
units_df = nwb_ecephys.units.to_dataframe()
print(f"Total number of units: {len(units_df)}")
print(f"Number of 'good' units: {sum(units_df['quality'] == 'good')}")

# Get spike times for good units
good_units = units_df[units_df['quality'] == 'good']
print(f"Extracting spike times for {len(good_units)} good units")

# For computational efficiency, select a subset of units
max_units = 20
selected_units = good_units.iloc[:max_units]
print(f"Selected {len(selected_units)} units for analysis")

# Get spike times for the selected units
spike_times = {}
for i, (_, unit) in enumerate(selected_units.iterrows()):
    unit_id = unit.name
    spike_times[unit_id] = nwb_ecephys.units['spike_times'][unit_id]

# Select a few stimulus types to analyze
stimulus_types = [
    'SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations',  # Standard moving bar
    'natmovie_EagleSwooping1_540x960Full_584x460Active_presentations'  # Natural movie
]

def analyze_responses_to_stimulus(stim_name, spike_times_dict, max_presentations=None):
    """Analyze neural responses to a specific stimulus."""
    print(f"\nAnalyzing responses to {stim_name}")
    
    if stim_name not in nwb_session.intervals:
        print(f"  Stimulus {stim_name} not found in intervals")
        return None
    
    # Get presentation times
    stim_df = nwb_session.intervals[stim_name].to_dataframe()
    
    if max_presentations is not None:
        stim_df = stim_df.iloc[:max_presentations]
    
    print(f"  Analyzing {len(stim_df)} stimulus presentations")
    
    # Define the pre/post time window for PSTH
    pre_time = 0.5  # seconds before stimulus
    post_time = 1.5  # seconds after stimulus
    bin_size = 0.05  # 50 ms bins
    
    # Create time bins for PSTH
    time_bins = np.arange(-pre_time, post_time + bin_size, bin_size)
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    
    # Initialize matrices to store the results
    n_units = len(spike_times_dict)
    n_bins = len(time_bins) - 1
    n_presentations = len(stim_df)
    
    # Matrix for trial-averaged PSTH
    psth_matrix = np.zeros((n_units, n_bins))
    
    # Calculate PSTH for each unit
    for i, (unit_id, unit_spike_times) in enumerate(spike_times_dict.items()):
        all_counts = []
        
        for _, stim in stim_df.iterrows():
            stim_start = stim['start_time']
            
            # Get spikes around this stimulus presentation
            window_start = stim_start - pre_time
            window_end = stim_start + post_time
            
            # Find spikes in this window
            mask = (unit_spike_times >= window_start) & (unit_spike_times <= window_end)
            spikes_in_window = unit_spike_times[mask]
            
            # Convert to time relative to stimulus onset
            relative_times = spikes_in_window - stim_start
            
            # Bin the spikes
            counts, _ = np.histogram(relative_times, bins=time_bins)
            all_counts.append(counts)
        
        # Convert to firing rates (spikes/second)
        all_counts = np.array(all_counts)
        mean_counts = np.mean(all_counts, axis=0)
        firing_rates = mean_counts / bin_size
        
        # Store in the PSTH matrix
        psth_matrix[i, :] = firing_rates
    
    # Plot PSTH heatmap
    plt.figure(figsize=(14, 10))
    
    # Sort units by response magnitude
    response_period = (time_centers >= 0) & (time_centers <= post_time/2)
    baseline_period = (time_centers >= -pre_time) & (time_centers < 0)
    
    response_magnitude = np.mean(psth_matrix[:, response_period], axis=1) - np.mean(psth_matrix[:, baseline_period], axis=1)
    sort_idx = np.argsort(response_magnitude)[::-1]  # Descending order
    
    # Create the heatmap
    plt.subplot(2, 1, 1)
    sns.heatmap(psth_matrix[sort_idx, :], cmap='viridis',
                xticklabels=np.round(time_centers, 2)[::5],
                yticklabels=['Unit '+str(i) for i in np.array(list(spike_times_dict.keys()))[sort_idx]],
                cbar_kws={'label': 'Firing Rate (spikes/s)'})
    plt.axvline(x=np.sum(time_centers < 0), color='white', linestyle='--', linewidth=2)
    plt.title(f'Neural Responses to {stim_name.split("_")[0]} Stimulus')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.ylabel('Unit ID')
    
    # Plot individual PSTHs for top responsive units
    plt.subplot(2, 1, 2)
    top_units = 5
    for i in range(min(top_units, n_units)):
        unit_idx = sort_idx[i]
        unit_id = list(spike_times_dict.keys())[unit_idx]
        plt.plot(time_centers, psth_matrix[unit_idx, :], 
                 label=f'Unit {unit_id}', linewidth=2)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.title(f'PSTH for Top {top_units} Responsive Units')
    plt.xlabel('Time relative to stimulus onset (s)')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    stim_short_name = stim_name.split("_")[0]
    plt.savefig(f'explore/psth_{stim_short_name}.png')
    plt.close()
    
    # Statistical analysis of responses
    print("  Statistical analysis of neural responses:")
    
    # For each unit, test if there's a significant response
    significant_units = 0
    for i, unit_id in enumerate(spike_times_dict.keys()):
        baseline = psth_matrix[i, baseline_period]
        response = psth_matrix[i, response_period]
        
        # T-test for response vs baseline
        t_stat, p_val = stats.ttest_ind(response, baseline)
        
        # Calculate response change
        baseline_mean = np.mean(baseline)
        response_mean = np.mean(response)
        percent_change = ((response_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else float('inf')
        
        if p_val < 0.05:
            significant_units += 1
            direction = "increase" if t_stat > 0 else "decrease"
            print(f"  Unit {unit_id}: Significant {direction} in firing rate (p={p_val:.4f}, {percent_change:.1f}% change)")
    
    print(f"  {significant_units}/{n_units} units showed significant responses (p<0.05)")
    
    return {
        'psth_matrix': psth_matrix,
        'time_centers': time_centers,
        'significant_units': significant_units
    }

# Analyze responses to each stimulus type
results = {}
for stim_type in stimulus_types:
    # Use a limited number of presentations to save computational time
    results[stim_type] = analyze_responses_to_stimulus(stim_type, spike_times, max_presentations=100)

# Compare responses to different stimuli
if all(results.values()):
    print("\nComparing responses across stimulus types:")
    
    plt.figure(figsize=(15, 8))
    
    # For each unit, plot the response magnitude across stimulus types
    response_magnitudes = {}
    
    for stim_type, result in results.items():
        if result is None:
            continue
            
        psth = result['psth_matrix']
        time_centers = result['time_centers']
        
        # Define response and baseline periods
        response_period = (time_centers >= 0) & (time_centers <= 0.5)
        baseline_period = (time_centers >= -0.5) & (time_centers < 0)
        
        # Calculate response magnitude for each unit
        response_magnitudes[stim_type] = []
        for i in range(psth.shape[0]):
            baseline_mean = np.mean(psth[i, baseline_period])
            response_mean = np.mean(psth[i, response_period])
            percent_change = ((response_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean > 0 else 0
            response_magnitudes[stim_type].append(percent_change)
    
    # Create a scatter plot comparing responses to different stimuli
    if len(response_magnitudes) >= 2:
        stim_types = list(response_magnitudes.keys())
        
        plt.scatter(response_magnitudes[stim_types[0]], 
                    response_magnitudes[stim_types[1]],
                    alpha=0.7)
        
        # Add reference line
        max_val = max(
            max(response_magnitudes[stim_types[0]]), 
            max(response_magnitudes[stim_types[1]])
        )
        min_val = min(
            min(response_magnitudes[stim_types[0]]), 
            min(response_magnitudes[stim_types[1]])
        )
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # Add unit labels
        for i, unit_id in enumerate(spike_times.keys()):
            plt.text(response_magnitudes[stim_types[0]][i], 
                    response_magnitudes[stim_types[1]][i], 
                    str(unit_id), fontsize=8)
        
        plt.xlabel(f'{stim_types[0].split("_")[0]} Response (% change)')
        plt.ylabel(f'{stim_types[1].split("_")[0]} Response (% change)')
        plt.title('Comparison of Neural Responses to Different Stimuli')
        plt.grid(True, alpha=0.3)
        plt.savefig('explore/stimulus_response_comparison.png')
        plt.close()
        
        # Calculate correlation between responses
        corr = np.corrcoef(response_magnitudes[stim_types[0]], response_magnitudes[stim_types[1]])[0, 1]
        print(f"  Correlation between responses to {stim_types[0].split('_')[0]} and {stim_types[1].split('_')[0]}: {corr:.3f}")
        
        # Count how many units respond to both stimuli
        sig_threshold = 20  # % change threshold for significance
        respond_to_first = np.abs(np.array(response_magnitudes[stim_types[0]])) > sig_threshold
        respond_to_second = np.abs(np.array(response_magnitudes[stim_types[1]])) > sig_threshold
        respond_to_both = np.logical_and(respond_to_first, respond_to_second)
        
        print(f"  Units responding to {stim_types[0].split('_')[0]}: {np.sum(respond_to_first)}/{len(respond_to_first)}")
        print(f"  Units responding to {stim_types[1].split('_')[0]}: {np.sum(respond_to_second)}/{len(respond_to_second)}")
        print(f"  Units responding to both: {np.sum(respond_to_both)}/{len(respond_to_both)}")

print("\nAnalysis complete - see output plots in the explore/ directory")