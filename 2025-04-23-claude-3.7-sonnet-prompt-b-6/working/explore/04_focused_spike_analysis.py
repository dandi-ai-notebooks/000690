# This script provides a more focused exploration of spike data from the Dandiset

import matplotlib.pyplot as plt
import numpy as np
import h5py
import remfile
import pynwb
import pandas as pd

# Save plots to file instead of displaying
plt.ioff()

# Load a file containing spike data 
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading main NWB file from {url}")

# Open as a remote file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print(f"Session ID: {nwb.session_id}")
print(f"Subject ID: {nwb.subject.subject_id}")

# Check for units (spike) data
print("\nExamining units (spike) information:")
if hasattr(nwb, 'units') and nwb.units is not None:
    # Get basic info
    units_df = nwb.units.to_dataframe()
    print(f"Total number of units: {len(units_df)}")
    
    # What properties do the units have?
    print("\nUnit properties:")
    for column in units_df.columns:
        print(f"- {column}")
    
    # Print firing rate statistics if available
    if 'firing_rate' in units_df.columns:
        print(f"\nFiring rate stats:")
        print(f"Mean: {units_df['firing_rate'].mean():.2f} Hz")
        print(f"Median: {units_df['firing_rate'].median():.2f} Hz")
        print(f"Min: {units_df['firing_rate'].min():.2f} Hz")
        print(f"Max: {units_df['firing_rate'].max():.2f} Hz")
    
    # Create firing rate histograms by quality if both columns exist
    if 'firing_rate' in units_df.columns and 'quality' in units_df.columns:
        # Convert to numeric if needed
        units_df['firing_rate'] = pd.to_numeric(units_df['firing_rate'], errors='coerce')
        
        # Plot histogram for each quality value
        quality_values = units_df['quality'].unique()
        if len(quality_values) > 0:
            plt.figure(figsize=(12, 8))
            for quality in quality_values:
                subset = units_df[units_df['quality'] == quality]
                if len(subset) > 0:
                    plt.hist(subset['firing_rate'], bins=30, alpha=0.5, label=f'Quality: {quality}')
            
            plt.xlabel('Firing Rate (Hz)')
            plt.ylabel('Number of Units')
            plt.title('Distribution of Firing Rates by Unit Quality')
            plt.legend()
            plt.savefig('explore/firing_rate_by_quality.png')
            plt.close()
            print("Created firing rate by quality histogram")
    
    # Try to get location information
    if 'location' in units_df.columns:
        print("\nUnits per brain region:")
        print(units_df['location'].value_counts())
    
        # Plot firing rate by brain region
        plt.figure(figsize=(12, 8))
        
        # Get the top 10 regions by unit count
        top_regions = units_df['location'].value_counts().nlargest(10).index.tolist()
        region_subset = units_df[units_df['location'].isin(top_regions)]
        
        if 'firing_rate' in units_df.columns:
            # Check if we have data to plot
            if len(region_subset) > 0:
                # Calculate mean and std of firing rates for each region
                stats = []
                for region in top_regions:
                    region_data = region_subset[region_subset['location'] == region]['firing_rate']
                    if len(region_data) > 0:
                        stats.append({
                            'location': region,
                            'mean': region_data.mean(),
                            'std': region_data.std()
                        })
                
                # Convert to DataFrame and sort
                region_stats = pd.DataFrame(stats)
                if len(region_stats) > 0:
                    region_stats = region_stats.sort_values('mean', ascending=False)
                    
                    # Plot bar chart with error bars
                    plt.bar(range(len(region_stats)), region_stats['mean'], yerr=region_stats['std'], 
                           tick_label=region_stats['location'], capsize=5)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel('Mean Firing Rate (Hz)')
                    plt.title('Average Firing Rate by Brain Region')
                    plt.tight_layout()
                    plt.savefig('explore/firing_rate_by_region.png')
                    plt.close()
                    print("Created firing rate by region plot")
    
    # Get a sample of spike times
    print("\nExamining spike timing patterns:")
    
    # Get units with the most spikes for a more informative analysis
    unit_spike_counts = []
    for i in range(len(units_df)):
        try:
            spike_times = nwb.units['spike_times'][i]
            unit_spike_counts.append((i, len(spike_times)))
        except Exception as e:
            print(f"Error getting spike times for unit {i}: {e}")
    
    # Sort by spike count (descending)
    unit_spike_counts.sort(key=lambda x: x[1], reverse=True)
    
    # Print info about top 5 units by spike count
    print("\nTop 5 units by spike count:")
    for i, (unit_id, spike_count) in enumerate(unit_spike_counts[:5]):
        try:
            spike_times = nwb.units['spike_times'][unit_id]
            print(f"Unit {unit_id}: {spike_count} spikes")
            print(f"  - First 5 spike times: {spike_times[:5]}")
            print(f"  - Last 5 spike times: {spike_times[-5:]}")
            
            # Calculate ISI (Inter-Spike Interval)
            if spike_count >= 2:
                isis = np.diff(spike_times)
                print(f"  - Mean ISI: {np.mean(isis):.6f} seconds")
                print(f"  - Median ISI: {np.median(isis):.6f} seconds")
                print(f"  - Min ISI: {np.min(isis):.6f} seconds")
                print(f"  - Max ISI: {np.max(isis):.6f} seconds")
                
                # Plot ISI histogram for this unit
                plt.figure(figsize=(10, 6))
                plt.hist(isis, bins=50)
                plt.xlabel('Inter-Spike Interval (seconds)')
                plt.ylabel('Count')
                plt.title(f'ISI Distribution for Unit {unit_id}')
                plt.savefig(f'explore/isi_histogram_unit_{unit_id}.png')
                plt.close()
        except Exception as e:
            print(f"Error analyzing unit {unit_id}: {e}")

    # Print the names of available stimulus presentations (intervals)
    print("\nStimulus presentation intervals available:")
    if hasattr(nwb, 'intervals') and len(nwb.intervals) > 0:
        for interval_name in nwb.intervals:
            try:
                interval_df = nwb.intervals[interval_name].to_dataframe()
                print(f"- {interval_name}: {len(interval_df)} presentations")
            except Exception as e:
                print(f"  Error reading interval {interval_name}: {e}")
    else:
        print("No stimulus intervals found in this file")
else:
    print("No units data found in this file")

print("Analysis complete")