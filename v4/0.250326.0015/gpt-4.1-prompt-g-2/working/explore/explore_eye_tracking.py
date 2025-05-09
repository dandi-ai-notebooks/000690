# Explore eye tracking, corneal reflection, and pupil tracking time series in the chosen NWB file.
# This script will plot the first ~10 seconds of the x/y positions for eye_tracking, pupil_tracking, and corneal_reflection_tracking, as well as the likely blink trace, if available.

import matplotlib.pyplot as plt
import remfile
import h5py
import pynwb
import numpy as np

# Config
NWB_URL = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
OUTPATH_PREFIX = "explore/eye_tracking"

def load_nwb():
    remote_file = remfile.File(NWB_URL)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    return nwb

def plot_spatial_series(series, label, fname):
    # Plot first 10 seconds of x/y positions
    data = series.data
    times = series.timestamps if hasattr(series, "timestamps") and series.timestamps is not None else None
    # Try to load first 2000 points (should correspond to ~10s at ~200Hz)
    n = min(2000, data.shape[0])
    d = data[:n, :]
    t = np.arange(n) if times is None else np.array(times[:n])
    plt.figure(figsize=(7, 4))
    plt.plot(t, d[:, 0], label="x")
    plt.plot(t, d[:, 1], label="y")
    plt.xlabel('Time (s)')
    plt.ylabel('Position (meters)')
    plt.title(f"{label} First ~10s Position Traces")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_likely_blink(series, fname):
    data = series.data
    n = min(2000, data.shape[0])
    d = np.array(data[:n])
    t = np.arange(n)
    plt.figure(figsize=(7, 2))
    plt.plot(t, d, label="blink")
    plt.xlabel('Time (s)')
    plt.ylabel('Likely Blink (bool)')
    plt.title("Likely Blink Events First ~10s")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def main():
    nwb = load_nwb()
    et = nwb.acquisition["EyeTracking"]

    # Eye tracking (ellipse center of eye)
    et_eye = et.spatial_series["eye_tracking"]
    plot_spatial_series(et_eye, "Eye Tracking", OUTPATH_PREFIX + "_eye.png")

    # Pupil tracking (ellipse center of pupil)
    et_pupil = et.spatial_series["pupil_tracking"]
    plot_spatial_series(et_pupil, "Pupil Tracking", OUTPATH_PREFIX + "_pupil.png")

    # Corneal reflection
    et_corneal = et.spatial_series["corneal_reflection_tracking"]
    plot_spatial_series(et_corneal, "Corneal Reflection Tracking", OUTPATH_PREFIX + "_corneal.png")

    # Blink events if available
    if hasattr(et, "likely_blink"):
        plot_likely_blink(et.likely_blink, OUTPATH_PREFIX + "_blink.png")

if __name__ == "__main__":
    main()