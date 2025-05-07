import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

# Load NWB file
url = 'https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/'
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get LFP data and electrodes
lfp_data = nwb.acquisition['probe_0_lfp_data']
electrodes = nwb.electrodes.to_dataframe()

# Plot first 1000 samples from first 5 channels
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(lfp_data.timestamps[:1000], lfp_data.data[:1000, i] + i*100, label=f'Channel {i}')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (uV)')
plt.title('LFP Traces (First 5 Channels)')
plt.legend()
plt.savefig('explore/lfp_traces.png')
plt.close()

# Plot electrode positions
plt.figure(figsize=(8, 8))
plt.scatter(electrodes['x'], electrodes['y'], c=electrodes['probe_vertical_position'])
plt.colorbar(label='Probe Vertical Position (um)')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.title('Electrode Positions')
plt.savefig('explore/electrode_positions.png')
plt.close()
