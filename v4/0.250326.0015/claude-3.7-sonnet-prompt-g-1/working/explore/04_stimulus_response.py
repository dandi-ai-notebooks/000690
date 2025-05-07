# This script explores the relationship between neural activity and visual stimuli
# by analyzing neural responses to specific stimulus presentations

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the main session NWB file
url = "https://api.dandiarchive.org/api/assets/9b14e3b4-5d3e-4121-ae5e-ced7bc92af4e/download/"
print(f"Loading main session NWB file from {url}")
print("This might take a moment as we're accessing a remote file...")

remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Basic info
print(f"Main session file loaded.")
print(f"Session ID: {nwb.session_id}")

# Let's get the stimulus presentations and unit spiking data
print("\nExtracting stimulus presentation information...")

# First, identify the available stimulus types
stimuli = [name for name in nwb.intervals.keys() if "presentations" in name]
print(f"Found {len(stimuli)} different stimulus types:")
for i, stim in enumerate(stimuli[:5], 1):  # Show first 5 as example
    print(f"{i}. {stim}")
if len(stimuli) > 5:
    print(f"...and {len(stimuli) - 5} more.")

# Choose one stimulus type for detailed analysis
# Let's look at the standard white bar stimulus and a natural movie for comparison
stim_bar = "Stim01_SAC_Wd15_Vel2_White_loop_presentations"
stim_movie = "Stim14_natmovie_10secFast_EagleSwoop_presentations"

# Extract stimulus presentation times for these two stimuli
print(f"\nAnalyzing responses to:")
print(f"1. {stim_bar}")
print(f"2. {stim_movie}")

try:
    bar_stim_df = nwb.intervals[stim_bar].to_dataframe()
    movie_stim_df = nwb.intervals[stim_movie].to_dataframe()
    
    print(f"\nBar stimulus: {len(bar_stim_df)} presentations")
    print(f"Movie stimulus: {len(movie_stim_df)} presentations")
    
    # Get a sample of presentation times for each stimulus
    bar_samples = bar_stim_df.iloc[:5][['start_time', 'stop_time']]
    movie_samples = movie_stim_df.iloc[:5][['start_time', 'stop_time']]
    
    print("\nSample bar stimulus presentations (first 5):")
    print(bar_samples)
    
    print("\nSample movie stimulus presentations (first 5):")
    print(movie_samples)
    
    # Get the units data
    units_df = nwb.units.to_dataframe()
    
    # Select a subset of high-firing, good-quality units
    if 'quality' in units_df.columns:
        good_units = units_df[(units_df['quality'] == 'good') & 
                             (units_df['firing_rate'] > 5)].sort_values('firing_rate', ascending=False)
        print(f"\nSelected {len(good_units)} high-firing, good-quality units for analysis")
    else:
        good_units = units_df[units_df['firing_rate'] > 5].sort_values('firing_rate', ascending=False)
        print(f"\nSelected {len(good_units)} high-firing units for analysis")
    
    # Take top 5 units for detailed analysis
    top_units = good_units.head(5)
    print("Top 5 units for analysis:")
    print(top_units[['firing_rate']])
    
    # Function to get spikes within a time window
    def get_spikes_in_window(unit_id, start_time, stop_time):
        """Get spikes for a given unit within the specified time window"""
        spike_times = nwb.units.spike_times[unit_id]
        mask = (spike_times >= start_time) & (spike_times <= stop_time)
        return spike_times[mask]
    
    # Function to analyze responses across multiple presentations
    def analyze_stimulus_responses(stim_df, unit_ids, pre_time=0.5, post_time=1.0, max_presentations=10):
        """Analyze unit responses to stimulus presentations"""
        results = []
        
        # Limit to a reasonable number of presentations for analysis
        if len(stim_df) > max_presentations:
            stim_df = stim_df.iloc[:max_presentations]
        
        for unit_id in unit_ids:
            unit_resp = {
                'unit_id': unit_id,
                'firing_rate': units_df.loc[unit_id, 'firing_rate'],
                'pre_counts': [],
                'post_counts': []
            }
            
            for _, stim in stim_df.iterrows():
                start_time = stim['start_time']
                stop_time = stim['stop_time']
                
                # Get spikes before and during/after stimulus
                pre_spikes = get_spikes_in_window(unit_id, start_time - pre_time, start_time)
                post_spikes = get_spikes_in_window(unit_id, start_time, start_time + post_time)
                
                # Count spikes and normalize by time window
                unit_resp['pre_counts'].append(len(pre_spikes) / pre_time)
                unit_resp['post_counts'].append(len(post_spikes) / post_time)
            
            # Calculate mean response
            unit_resp['mean_pre'] = np.mean(unit_resp['pre_counts']) if unit_resp['pre_counts'] else 0
            unit_resp['mean_post'] = np.mean(unit_resp['post_counts']) if unit_resp['post_counts'] else 0
            unit_resp['response_ratio'] = (unit_resp['mean_post'] / unit_resp['mean_pre']) if unit_resp['mean_pre'] > 0 else np.nan
            
            results.append(unit_resp)
        
        return results
    
    # Analyze responses to both stimulus types
    print("\nAnalyzing neural responses to stimuli...")
    bar_responses = analyze_stimulus_responses(bar_stim_df, top_units.index)
    movie_responses = analyze_stimulus_responses(movie_stim_df, top_units.index)
    
    # Create bar plot comparing responses
    plt.figure(figsize=(15, 10))
    
    # Bar stimulus responses
    plt.subplot(2, 1, 1)
    unit_ids = [resp['unit_id'] for resp in bar_responses]
    pre_rates = [resp['mean_pre'] for resp in bar_responses]
    post_rates = [resp['mean_post'] for resp in bar_responses]
    
    x = np.arange(len(unit_ids))
    width = 0.35
    
    plt.bar(x - width/2, pre_rates, width, label='Pre-stimulus')
    plt.bar(x + width/2, post_rates, width, label='During stimulus')
    plt.xlabel('Unit ID')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Neural Responses to Bar Stimulus')
    plt.xticks(x, unit_ids)
    plt.legend()
    
    # Movie stimulus responses
    plt.subplot(2, 1, 2)
    unit_ids = [resp['unit_id'] for resp in movie_responses]
    pre_rates = [resp['mean_pre'] for resp in movie_responses]
    post_rates = [resp['mean_post'] for resp in movie_responses]
    
    plt.bar(x - width/2, pre_rates, width, label='Pre-stimulus')
    plt.bar(x + width/2, post_rates, width, label='During stimulus')
    plt.xlabel('Unit ID')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Neural Responses to Natural Movie Stimulus')
    plt.xticks(x, unit_ids)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('explore/stimulus_responses.png', dpi=300)
    print("Stimulus response plot saved to explore/stimulus_responses.png")
    
    # Create plot of response ratios for comparison
    plt.figure(figsize=(10, 6))
    
    bar_ratios = [resp['response_ratio'] for resp in bar_responses]
    movie_ratios = [resp['response_ratio'] for resp in movie_responses]
    
    plt.bar(x - width/2, bar_ratios, width, label='Bar Stimulus')
    plt.bar(x + width/2, movie_ratios, width, label='Movie Stimulus')
    plt.axhline(y=1.0, color='r', linestyle='--', label='No change')
    
    plt.xlabel('Unit ID')
    plt.ylabel('Response Ratio (During/Pre)')
    plt.title('Comparison of Neural Response Ratios Between Stimulus Types')
    plt.xticks(x, unit_ids)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('explore/response_ratios.png', dpi=300)
    print("Response ratio plot saved to explore/response_ratios.png")
    
except Exception as e:
    print(f"Error during analysis: {str(e)}")

print("\nExploration complete!")