"""
This script explores the metadata of visual stimuli used in the experiment,
without attempting to load the large image data that might cause timeouts.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
from collections import defaultdict

# Load the main NWB file (smaller than the image file)
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
print(f"Loading NWB file from URL: {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Explore stimulus presentation intervals
print("\nStimulus presentation intervals:")
stim_info = []

for interval_name, interval in nwb.intervals.items():
    if 'presentations' in interval_name:
        # Convert to dataframe for easier analysis
        try:
            # Extract only the first few rows to avoid loading too much data
            df = interval.to_dataframe().head(5)
            
            # Get counts and basic info
            num_presentations = len(interval.id[:])
            
            # Extract parameters from the name
            params = {}
            parts = interval_name.split('_')
            for part in parts:
                if 'Wd' in part:
                    params['width'] = part.replace('Wd', '')
                elif 'Vel' in part:
                    params['velocity'] = part.replace('Vel', '')
                elif 'Bndry' in part:
                    params['boundary'] = part.replace('Bndry', '')
                elif 'Cntst' in part:
                    params['contrast'] = part.replace('Cntst', '')
            
            # Get unique stimulus names if available
            unique_stim_names = []
            if 'stimulus_name' in df.columns:
                unique_stim_names = df['stimulus_name'].unique().tolist()
            
            # Store the information
            stim_info.append({
                'interval_name': interval_name,
                'num_presentations': num_presentations,
                'parameters': params,
                'unique_stim_names': unique_stim_names,
                'available_columns': df.columns.tolist()
            })
            
            print(f"  {interval_name}: {num_presentations} presentations")
            print(f"    Parameters: {params}")
            if unique_stim_names:
                print(f"    Sample stimulus names: {unique_stim_names}")
            print(f"    Available data columns: {df.columns.tolist()[:5]}...")
            
        except Exception as e:
            print(f"  Error processing {interval_name}: {e}")

# Analyze the types of stimuli
print("\nTypes of stimuli based on interval names:")
stimulus_categories = defaultdict(int)

for info in stim_info:
    name = info['interval_name']
    # Categorize based on prefix
    if 'SAC' in name:
        stimulus_categories['Standard Bar (SAC)'] += 1
    elif 'Disco' in name:
        stimulus_categories['Disco Bar'] += 1
    elif 'Ring' in name:
        stimulus_categories['Ring'] += 1
    elif 'Disk' in name:
        stimulus_categories['Disk'] += 1
    elif 'natmovie' in name:
        if 'Eagle' in name:
            stimulus_categories['Natural Movie (Eagle)'] += 1
        elif 'Snake' in name:
            stimulus_categories['Natural Movie (Snake)'] += 1
        elif 'Cricket' in name:
            stimulus_categories['Natural Movie (Cricket)'] += 1
        elif 'Squirrel' in name:
            stimulus_categories['Natural Movie (Squirrel)'] += 1
        else:
            stimulus_categories['Natural Movie (Other)'] += 1
    elif 'curl' in name:
        stimulus_categories['Curl'] += 1
    elif 'UD' in name:
        stimulus_categories['Up-Down'] += 1
    elif 'Green' in name:
        stimulus_categories['Green Bar'] += 1
    else:
        stimulus_categories['Other'] += 1

print("Stimulus categories:")
for category, count in stimulus_categories.items():
    print(f"  {category}: {count}")

# Look at the parameters used in different stimuli
print("\nParameter variations:")
width_vals = set()
velocity_vals = set()
boundary_vals = set()
contrast_vals = set()

for info in stim_info:
    params = info['parameters']
    if 'width' in params:
        width_vals.add(params['width'])
    if 'velocity' in params:
        velocity_vals.add(params['velocity'])
    if 'boundary' in params:
        boundary_vals.add(params['boundary'])
    if 'contrast' in params:
        contrast_vals.add(params['contrast'])

print(f"  Width values: {sorted(width_vals)}")
print(f"  Velocity values: {sorted(velocity_vals)}")
print(f"  Boundary types: {sorted(boundary_vals)}")
print(f"  Contrast values: {sorted(contrast_vals)}")

# Get information about the recording sessions
print("\nRecording Session Information:")
print(f"  Session ID: {nwb.session_id}")
print(f"  Institution: {nwb.institution}")
print(f"  Subject ID: {nwb.subject.subject_id}")
print(f"  Subject Age: {nwb.subject.age}")

print("\nExploration of stimulus metadata complete!")
io.close()