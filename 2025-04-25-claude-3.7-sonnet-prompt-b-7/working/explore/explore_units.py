"""
This script explores the spiking activity of neurons in the dataset.
We'll look at basic properties of the units (neurons) in the main NWB file
to get an understanding of the neural data without loading large arrays.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
print("Connecting to DANDI archive...")
client = DandiAPIClient()
dandiset = client.get_dandiset("000690")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")

# Get assets (we only need the list, not the actual data)
print("\nGetting assets list...")
assets = list(dandiset.get_assets())
print(f"Found {len(assets)} assets in the dataset")

# Filter to just the main NWB files (not the probe or image files)
main_nwb_files = [asset for asset in assets 
                  if not asset.path.endswith('_image.nwb') 
                  and not asset.path.endswith('_ecephys.nwb')
                  and asset.path.endswith('.nwb')]

print(f"Found {len(main_nwb_files)} main NWB files")

# Print the first few to verify
print("Sample main NWB files:")
for asset in main_nwb_files[:5]:
    print(f"- {asset.path}")

print("\nExploration complete!")