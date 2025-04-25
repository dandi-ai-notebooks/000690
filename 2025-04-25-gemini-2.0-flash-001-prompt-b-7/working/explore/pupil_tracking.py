# explore/pupil_tracking.py
# This script explores the pupil tracking data in the NWB file and plots pupil area over time.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the pupil tracking data
eye_tracking = nwb.acquisition["EyeTracking"]
pupil_tracking = eye_tracking.pupil_tracking

# Load a subset of the data and timestamps
start = 0
end = 1000  # Load the first 1000 samples
pupil_area = pupil_tracking.area[:end]
timestamps = pupil_tracking.timestamps[:end]

# Plot the pupil area over time
plt.figure(figsize=(10, 5))
plt.plot(timestamps, pupil_area)
plt.xlabel("Time (s)")
plt.ylabel("Pupil Area (meters)")
plt.title("Pupil Area Over Time")
plt.savefig("explore/pupil_area.png")
plt.close()