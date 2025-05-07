# This script explores the stimulus information in the Dandiset
# to understand the types of stimuli and their timing

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import islice

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/9b14e3b4-5d3e-4121-ae5e-ced7bc92af4e/download/"
print(f"Loading NWB file from {url}")
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic metadata about the session
print(f"Session ID: {nwb.session_id}")
print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Species: {nwb.subject.species}")
print(f"Age: {nwb.subject.age} ({nwb.subject.age_in_days} days)")
print(f"Sex: {nwb.subject.sex}")
print(f"Strain: {nwb.subject.strain}")

# List all available stimulus presentations
stimuli = [name for name in nwb.intervals.keys() if "presentations" in name]
print(f"Number of stimulus types: {len(stimuli)}")
print("Stimulus names:\n- " + "\n- ".join(stimuli))

# Get statistics for each stimulus type
stimulus_stats = []
for stim_name in stimuli:
    interval = nwb.intervals[stim_name]
    presentations = interval.to_dataframe()
    duration = presentations['stop_time'].iloc[0] - presentations['start_time'].iloc[0]
    stimulus_stats.append({
        'name': stim_name,
        'count': len(presentations),
        'total_duration': presentations['stop_time'].max() - presentations['start_time'].min(),
        'avg_duration': duration,
        'start_time_min': presentations['start_time'].min(),
        'stop_time_max': presentations['stop_time'].max()
    })

# Convert to DataFrame and save
stimulus_df = pd.DataFrame(stimulus_stats)
print("\nStimulus Statistics:")
print(stimulus_df.sort_values('start_time_min')[['name', 'count', 'avg_duration', 'total_duration']])

# Let's look at detailed info for one stimulus as an example
example_stim = "Stim01_SAC_Wd15_Vel2_White_loop_presentations"
example_df = nwb.intervals[example_stim].to_dataframe().head(3)
print(f"\nExample data for {example_stim}:")
relevant_columns = ['start_time', 'stop_time', 'stimulus_name', 'frame', 'orientation', 'size', 'units']
print(example_df[relevant_columns])

# Plot the timing of all stimuli
plt.figure(figsize=(12, 8))
plt.barh(range(len(stimulus_stats)), 
         [s['total_duration'] for s in stimulus_stats],
         left=[s['start_time_min'] for s in stimulus_stats],
         height=0.8)
plt.yticks(range(len(stimulus_stats)), [s['name'].split('_presentations')[0] for s in stimulus_stats])
plt.xlabel('Time (seconds)')
plt.title('Stimulus Presentation Timeline')
plt.tight_layout()
plt.savefig('explore/stimulus_timeline.png', dpi=300)

# Get unique stimulus names 
stim_names = set()
for stim_name in stimuli:
    interval = nwb.intervals[stim_name]
    presentations = interval.to_dataframe()
    stim_names.update(presentations['stimulus_name'].unique())

print("\nUnique stimulus names:")
for name in sorted(stim_names):
    print(f"- {name}")