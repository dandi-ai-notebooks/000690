"""
This script explores the running wheel and eye tracking data from the dataset
to understand behavioral measurements during the experiment.
"""

import pynwb
import h5py
import remfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
print("Connecting to DANDI archive...")
client = DandiAPIClient()
dandiset = client.get_dandiset("000690")

# Get a specific NWB file
print("Finding a representative subject/session...")
subject = "695763"  # The subject we looked at earlier
assets = list(dandiset.get_assets())
main_nwb = [asset for asset in assets 
            if f"sub-{subject}/" in asset.path and 
            not "_probe-" in asset.path and 
            not "_image.nwb" in asset.path and
            asset.path.endswith('.nwb')][0]

print(f"Selected main NWB file: {main_nwb.path}")

# Get the URL for the asset
asset_url = f"https://api.dandiarchive.org/api/assets/{main_nwb.identifier}/download/"

# Load the file
print(f"Loading main NWB file (this might take a moment)...")
remote_file = remfile.File(asset_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# --- Running Wheel Data ---
print("\n--- Running Wheel Data ---")
if hasattr(nwb.processing, 'running') and 'running_speed' in nwb.processing['running'].data_interfaces:
    running_speed = nwb.processing['running']['running_speed']
    
    # Get a sample of the running speed data
    sample_size = 10000  # Sample the first 10,000 points
    speeds = running_speed.data[:sample_size]
    timestamps = running_speed.timestamps[:sample_size]
    
    # Plot running speed
    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, speeds)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speed (cm/s)')
    plt.title(f'Running Speed for Mouse {subject}')
    plt.savefig('explore/running_speed.png')
    
    # Print some summary statistics
    print(f"Running speed data shape: {running_speed.data.shape}")
    print(f"Running speed sampling rate: {sample_size / (timestamps[-1] - timestamps[0]):.2f} Hz")
    print(f"Mean running speed (sample): {np.mean(speeds):.2f} cm/s")
    print(f"Max running speed (sample): {np.max(speeds):.2f} cm/s")
    print(f"Running data time span: {running_speed.timestamps[0]:.2f} - {running_speed.timestamps[-1]:.2f} seconds")
else:
    print("Running wheel data not found in this file")

# --- Eye Tracking Data ---
print("\n--- Eye Tracking Data ---")
if 'EyeTracking' in nwb.acquisition:
    eye_tracking = nwb.acquisition['EyeTracking']
    
    # Check available data
    print("Eye tracking data types:")
    for name in dir(eye_tracking):
        if not name.startswith('_'):
            print(f"- {name}")
    
    if hasattr(eye_tracking, 'pupil_tracking'):
        # Get pupil size data
        sample_size = 10000  # Sample the first 10,000 points
        pupil_tracking = eye_tracking.pupil_tracking
        pupil_area = pupil_tracking.area[:sample_size]
        timestamps = pupil_tracking.timestamps[:sample_size]
        
        # Plot pupil area
        plt.figure(figsize=(12, 4))
        plt.plot(timestamps, pupil_area)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Pupil Area')
        plt.title(f'Pupil Area for Mouse {subject}')
        plt.savefig('explore/pupil_area.png')
        
        # Print some summary statistics
        print(f"Pupil tracking data shape: {pupil_tracking.area.shape}")
        print(f"Pupil area sampling rate: {sample_size / (timestamps[-1] - timestamps[0]):.2f} Hz")
        print(f"Mean pupil area (sample): {np.mean(pupil_area):.2f}")
        print(f"Pupil data time span: {timestamps[0]:.2f} - {timestamps[-1]:.2f} seconds")
        
    if hasattr(eye_tracking, 'likely_blink'):
        # Get blink data
        blink_data = eye_tracking.likely_blink.data[:5000]  # Sample
        blink_timestamps = eye_tracking.likely_blink.timestamps[:5000]
        
        # Count blinks
        blink_count = np.sum(blink_data)
        print(f"Number of likely blinks in sample: {blink_count}")
        print(f"Blink rate: {blink_count / (blink_timestamps[-1] - blink_timestamps[0]):.2f} blinks/second")
else:
    print("Eye tracking data not found in this file")

# Close the file
h5_file.close()
io.close()
print("\nExploration complete!")