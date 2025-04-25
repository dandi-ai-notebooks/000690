# explore/plot_running_speed.py
# Goal: Visualize a subset of the running speed data from the NWB file.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use seaborn styling
sns.set_theme()

print("Loading NWB file remotely...")
# Define the URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"

# Use remfile to access the remote file
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file, mode='r')

# Use pynwb to read the NWB file structure
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()
print("NWB file loaded.")

# Access running speed data and timestamps
running_speed_ts = nwb.processing['running']['running_speed']

# Load a subset of the data (e.g., first 10000 points) to avoid long load times
num_points_to_load = 10000
if len(running_speed_ts.data) >= num_points_to_load:
    print(f"Loading first {num_points_to_load} points of running speed data...")
    speed_data = running_speed_ts.data[:num_points_to_load]
    timestamps = running_speed_ts.timestamps[:num_points_to_load]
else:
    print("Loading all running speed data (less than 10000 points)...")
    speed_data = running_speed_ts.data[:]
    timestamps = running_speed_ts.timestamps[:]

print(f"Successfully loaded {len(speed_data)} data points.")

# Create the plot
print("Generating plot...")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(timestamps, speed_data)
ax.set_title(f'Running Speed (First {len(speed_data)} points)')
ax.set_xlabel('Time (s)')
ax.set_ylabel(f'Speed ({running_speed_ts.unit})')
plt.tight_layout()

# Save the plot
output_path = "explore/running_speed.png"
print(f"Saving plot to {output_path}")
plt.savefig(output_path)
plt.close(fig) # Close the figure to free memory

print("Script finished.")