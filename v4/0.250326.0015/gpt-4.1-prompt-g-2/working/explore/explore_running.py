# Explore running wheel data in the chosen NWB file.
# This script will plot the first ~10 seconds of raw running wheel rotation and running speed.

import matplotlib.pyplot as plt
import remfile
import h5py
import pynwb
import numpy as np

# Config
NWB_URL = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
OUTPATH_PREFIX = "explore/running"

def load_nwb():
    remote_file = remfile.File(NWB_URL)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    return nwb

def plot_timeseries(series, label, fname, ylabel):
    # Plot first 2000 points (~10s for wheel, ~17kHz sampling)
    n = min(2000, series.data.shape[0])
    data = series.data[:n]
    t = np.arange(n)
    plt.figure(figsize=(7, 3))
    plt.plot(t, data, label=label)
    plt.xlabel('Sample')
    plt.ylabel(ylabel)
    plt.title(f'{label} First ~10s')
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def main():
    nwb = load_nwb()
    # Raw rotation (acquisition)
    wheel_rot = nwb.acquisition["raw_running_wheel_rotation"]
    plot_timeseries(wheel_rot, "Running Wheel Rotation", OUTPATH_PREFIX + "_rotation.png", "Radian")

    # Running speed (processing)
    running = nwb.processing["running"]
    speed = running.data_interfaces["running_speed"]
    plot_timeseries(speed, "Running Speed", OUTPATH_PREFIX + "_speed.png", "cm/s")

if __name__ == "__main__":
    main()