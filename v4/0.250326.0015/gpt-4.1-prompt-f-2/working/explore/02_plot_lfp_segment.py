# This script loads and plots a short segment of the LFP data from several channels of the NWB file.
# It visualizes the first 5 seconds for channels 0-4 and saves the figure as a PNG.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

probe_0_lfp_data = nwb.acquisition["probe_0_lfp_data"]
timestamps = probe_0_lfp_data.timestamps  # shape (10117092,)
data = probe_0_lfp_data.data              # shape (10117092, 95)

# We'll plot the first 5 seconds for channels 0-4
sample_mask = (timestamps[:] <= timestamps[0] + 5)
sample_indices = np.where(sample_mask)[0]
if len(sample_indices) == 0:
    raise Exception("No samples found for the first 5 seconds.")

channels_to_plot = range(5)
plt.figure(figsize=(10, 6))
for i, ch in enumerate(channels_to_plot):
    plt.plot(timestamps[sample_indices], data[sample_indices, ch] * 1e3 + i*2, label=f'Ch {ch}')
plt.xlabel("Time (s)")
plt.ylabel("LFP (mV) (traces offset vertically)")
plt.title("LFP: First 5 seconds, channels 0-4")
plt.legend()
plt.tight_layout()
plt.savefig("explore/lfp_segment.png")
plt.close()