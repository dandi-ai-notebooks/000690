"""
This script creates a summary of Dandiset 000690 based on information we've gathered.
It synthesizes what we know about the dataset structure and content.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# Create a directory for our visualizations if it doesn't exist
os.makedirs("explore", exist_ok=True)

# Create a visual summary of the file organization
file_structure = {
    'Subject files': [
        'Main NWB file (sub-XXXX_ses-XXXX.nwb)',
        'Image NWB file (sub-XXXX_ses-XXXX_image.nwb)',
        'Probe NWB files (sub-XXXX_ses-XXXX_probe-X_ecephys.nwb)'
    ],
    'Main file content': [
        'Eye tracking data',
        'Running wheel data',
        'Stimulus presentation timing',
        'Units data (neurons)',
        'Electrodes data',
        'Intervals data'
    ],
    'Image file content': [
        'Stimulus templates',
        'Visual stimuli frames',
        'Eye tracking data',
        'Running data'
    ],
    'Probe file content': [
        'LFP data',
        'Electrode locations',
        'Units data (for specific probe)'
    ]
}

# Create a visual summary of the stimulus types
stimulus_types = {
    'SAC (Standard)': 'White bar on black background, 15° width, 2 sec screen crossing',
    'SAC_Wd45': 'White bar, 45° width (wider bar)',
    'SAC_Vel8': 'White bar, faster velocity (8 units)',
    'SAC_Cntst1': 'Black bar on white background (contrast reversed)',
    'Disco2SAC': 'Colored striped bar (changing colors)',
    'DOT': 'Dot stimulus',
    'Disk': 'Disk-shaped stimulus',
    'Ring': 'Ring-shaped stimulus',
    'Natural Movies': 'Eagle swooping, squirrel/mice videos'
}

print("NWB File Structure and Content Summary")
print("======================================")

print("\nFile Organization:")
for category, files in file_structure.items():
    print(f"\n{category}:")
    for file in files:
        print(f"- {file}")

print("\nStimulus Types:")
for stim_type, description in stimulus_types.items():
    print(f"- {stim_type}: {description}")

print("\nExperimental Design:")
print("- Multiple mice were shown various visual stimuli while neural activity was recorded")
print("- Visual stimuli included simple bars, shapes, and natural movies")
print("- Neuropixels probes recorded activity across brain regions including visual areas and hippocampus")
print("- Eye tracking and running wheel data were collected simultaneously")
print("- Multiple parametric variations of stimuli were used to investigate encoding properties")

print("\nKey Features of the Dataset:")
print("- Comprehensive recordings across the visual pathway from thalamus to hippocampus")
print("- Simultaneous behavioral measurements (eye tracking, running)")
print("- Parametric variations of stimuli (width, velocity, contrast, boundary effects)")
print("- Natural movie stimuli with ethological relevance (predator videos)")
print("- High-density neural recordings with multiple Neuropixels probes per mouse")

# Create a visualization of the experimental setup
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.set_axis_off()

# Draw mouse
mouse_x, mouse_y = 3, 4
circle = plt.Circle((mouse_x, mouse_y), 0.8, color='gray')
ax.add_patch(circle)
ear1 = plt.Circle((mouse_x -.5, mouse_y + .7), 0.3, color='gray')
ear2 = plt.Circle((mouse_x + .5, mouse_y + .7), 0.3, color='gray')
ax.add_patch(ear1)
ax.add_patch(ear2)

# Draw screen
screen_x, screen_y = 1, 4
rect = plt.Rectangle((screen_x - 1.5, screen_y - 1), 1.5, 2, color='black')
ax.add_patch(rect)
plt.plot([screen_x - 1, screen_x - 1 + 0.3], [screen_y + 0.5, screen_y + 0.5], color='white', linewidth=3)

# Draw probes
probe_colors = ['red', 'blue', 'green', 'orange']
for i, (dx, dy) in enumerate([(-0.3, 0.5), (0, 0.8), (0.3, 0.4), (0.6, 0.7)]):
    plt.plot([mouse_x + dx, mouse_x + dx], [mouse_y - 0.5, mouse_y + dy], 
             color=probe_colors[i % len(probe_colors)], linewidth=2)

# Draw eye tracker
plt.plot([mouse_x + 1.2, mouse_x + 0.4], [mouse_y + 0.2, mouse_y + 0.2], 'r-', linewidth=1)
circle = plt.Circle((mouse_x + 1.4, mouse_y + 0.2), 0.2, fill=False, color='red')
ax.add_patch(circle)

# Draw labels
plt.text(mouse_x, mouse_y - 1.2, 'Mouse', ha='center')
plt.text(screen_x - 0.75, screen_y - 1.5, 'Visual Stimuli', ha='center')
plt.text(mouse_x + 0.3, mouse_y + 1.4, 'Neuropixels Probes', ha='center')
plt.text(mouse_x + 1.4, mouse_y - 0.3, 'Eye Tracking', ha='center')

# Draw running wheel
wheel_x, wheel_y = mouse_x - 2, mouse_y - 0.8
circle = plt.Circle((wheel_x, wheel_y), 0.6, fill=False, color='blue')
ax.add_patch(circle)
for i in range(8):
    angle = i * np.pi / 4
    plt.plot([wheel_x, wheel_x + 0.6 * np.cos(angle)], 
             [wheel_y, wheel_y + 0.6 * np.sin(angle)], 
             'b-', linewidth=1)
plt.text(wheel_x, wheel_y - 1, 'Running Wheel', ha='center')

# Draw brain regions
regions = ['V1', 'V2', 'HigherVisual', 'Hippocampus']
positions = [(mouse_x - 0.3, mouse_y), (mouse_x + 0.3, mouse_y + 0.2), 
             (mouse_x - 0.1, mouse_y + 0.4), (mouse_x + 0.6, mouse_y - 0.2)]
for region, (x, y) in zip(regions, positions):
    plt.text(x, y, region, ha='center', fontsize=8,
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))

plt.title('Dandiset 000690: Vision2Hippocampus Experimental Setup')
plt.savefig('explore/experiment_diagram.png', dpi=300, bbox_inches='tight')

print("\nSummary diagram saved as 'explore/experiment_diagram.png'")