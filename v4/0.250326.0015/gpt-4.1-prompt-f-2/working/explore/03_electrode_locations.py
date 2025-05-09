# This script visualizes the distribution of electrode locations (brain regions) for the selected NWB file.
# It loads the electrodes table and makes a bar plot of counts per location.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

elec_df = nwb.electrodes.to_dataframe()

counts = elec_df['location'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(8,5))
counts.plot(kind="bar")
plt.title("Electrode counts by brain region (location)")
plt.xlabel("Brain region")
plt.ylabel("Number of electrodes")
plt.tight_layout()
plt.savefig("explore/electrode_locations.png")
plt.close()