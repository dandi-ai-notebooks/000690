import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Script to explore the EyeTracking data in the NWB file

url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get the eye tracking data
eye_tracking = nwb.acquisition["EyeTracking"]
pupil_tracking_data = eye_tracking.pupil_tracking.data
pupil_tracking_timestamps = eye_tracking.pupil_tracking.timestamps

# Plot the eye tracking position over time
num_timepoints = min(1000, len(pupil_tracking_timestamps))  # Limit the number of timepoints

plt.figure(figsize=(10, 5))
plt.plot(pupil_tracking_timestamps[:num_timepoints], pupil_tracking_data[:num_timepoints, 0], label="X Position")
plt.plot(pupil_tracking_timestamps[:num_timepoints], pupil_tracking_data[:num_timepoints, 1], label="Y Position")
plt.xlabel("Time (s)")
plt.ylabel("Position (pixels)")
plt.title("Eye Tracking Position Over Time")
plt.legend()
plt.savefig("explore/eye_tracking_position.png")

plt.close()

print("Eye tracking data exploration script completed. Plot saved to explore/eye_tracking_position.png")