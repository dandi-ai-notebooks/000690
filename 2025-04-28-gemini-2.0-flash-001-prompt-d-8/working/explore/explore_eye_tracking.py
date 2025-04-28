import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Script to explore eye tracking data from the NWB file

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get eye tracking data
eye_tracking = nwb.acquisition['EyeTracking']
pupil_tracking = eye_tracking.pupil_tracking
eye_tracking_data = pupil_tracking.data
eye_tracking_timestamps = pupil_tracking.timestamps[:]

# Plot the eye tracking data (first 1000 samples)
num_samples = min(1000, len(eye_tracking_data))
plt.figure(figsize=(10, 5))
plt.plot(eye_tracking_timestamps[:num_samples], eye_tracking_data[:num_samples, 0], label='X')
plt.plot(eye_tracking_timestamps[:num_samples], eye_tracking_data[:num_samples, 1], label='Y')
plt.xlabel('Time (s)')
plt.ylabel('Position (pixels)')
plt.title('Eye Tracking Data')
plt.legend()
plt.savefig('explore/eye_tracking_plot.png')
plt.close()

print("Eye tracking data exploration complete. Plot saved to explore/eye_tracking_plot.png")