"""
This script explores the structure and metadata of visual stimuli in Dandiset 000690.
We'll look only at the metadata about stimulus types without loading the actual stimulus frames
to avoid timeout issues with large datasets.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import os

print("Exploring stimuli in Dandiset 000690")

# Load the image NWB file
url = "https://api.dandiarchive.org/api/assets/4e1695f9-2998-41d8-8c6d-286509be5fb1/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get list of stimulus templates (without loading the data)
print("\n=== STIMULUS TEMPLATES ===")
stim_names = list(nwb.stimulus_template.keys())
print(f"Number of stimulus templates: {len(stim_names)}")
print("Template names:")
for name in stim_names:
    print(f"  - {name}")

# Parse the stimulus names to understand the parameters
stim_parameters = {}
for name in stim_names:
    # Basic type
    if name.startswith('natmovie'):
        stim_type = 'Natural Movie'
    elif '_Wd' in name:
        # Extract the base stimulus type
        parts = name.split('_')
        stim_type = parts[0]
        
        # Extract parameters where available
        params = {}
        for part in parts:
            if part.startswith('Wd'):
                params['width'] = part[2:] + ' degrees'
            elif part.startswith('Vel'):
                params['velocity'] = part[3:]
            elif part.startswith('Bndry'):
                params['boundary'] = part[5:]
            elif part.startswith('Cntst'):
                params['contrast'] = part[5:]
        stim_parameters[name] = {
            'type': stim_type,
            'parameters': params
        }
    else:
        stim_parameters[name] = {
            'type': name
        }

# Get all interval types to understand stimulus presentation timing
print("\n=== STIMULUS PRESENTATIONS ===")
interval_names = list(nwb.intervals.keys())
presentation_intervals = [name for name in interval_names if name.endswith('_presentations')]
print(f"Number of presentation interval types: {len(presentation_intervals)}")

# Sample a few intervals to understand timing
print("\n=== SAMPLE STIMULUS TIMING ===")
selected_intervals = ['SAC_Wd15_Vel2_Bndry1_Cntst0_loop_presentations', 
                     'natmovie_20sec_EagleSwoop1_540x960Full_584x460Active_presentations']

for interval_name in selected_intervals:
    if interval_name not in nwb.intervals:
        print(f"Interval {interval_name} not found, skipping")
        continue
        
    print(f"\nTimings for: {interval_name}")
    interval = nwb.intervals[interval_name]
    df = interval.to_dataframe().head(5)
    
    # Print basic timing info
    print(f"  First 5 presentation times:")
    for i, row in df.iterrows():
        duration = row['stop_time'] - row['start_time']
        print(f"    {i}: {row['start_time']:.2f}s - {row['stop_time']:.2f}s (duration: {duration:.4f}s)")
    
    # Get overall statistics on duration
    all_times = interval.to_dataframe()
    durations = all_times['stop_time'] - all_times['start_time']
    print(f"  Total presentations: {len(all_times)}")
    print(f"  Average duration: {durations.mean():.4f}s")
    print(f"  Total duration: {all_times['stop_time'].max() - all_times['start_time'].min():.2f}s")

# Categorize stimuli by type
print("\n=== STIMULUS CATEGORIES ===")
stim_types = {}
for name in stim_names:
    base_type = name.split('_')[0] if '_' in name else name
    stim_types.setdefault(base_type, []).append(name)

for base_type, variants in stim_types.items():
    print(f"\n{base_type}: {len(variants)} variants")
    for variant in variants:
        print(f"  - {variant}")

# Close the file
h5_file.close()
io.close()
print("\nExploration complete.")