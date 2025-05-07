import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode='r')
nwb = io.read()

# Get the EyeTracking data
eye_tracking = nwb.acquisition['EyeTracking']
pupil_tracking = eye_tracking.pupil_tracking
area = np.array(pupil_tracking.area[:])

# Print the first 10 pupil areas
print("First 10 pupil areas:", area[:10])

# Plot pupil area vs. time (index)
plt.figure(figsize=(10, 6))
plt.plot(area)
plt.xlabel("Index")
plt.ylabel("Pupil Area")
plt.title("Pupil Area vs. Index")
plt.savefig("explore/pupil_area_vs_index.png")
plt.close()