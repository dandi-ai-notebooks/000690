"""
This script analyzes a specific NWB probe file from the Dandiset 000690
using the tools_cli.py utility instead of loading the entire file.
"""

import json
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Let's use a broader approach - check multiple pages and subjects
print("Searching for a probe file...")

# Try different pages until we find a probe file
probe_files = []
page = 1
max_pages = 5  # Limit to avoid too many requests

while page <= max_pages and not probe_files:
    print(f"Checking page {page}...")
    result = subprocess.run(
        ["python", "tools_cli.py", "dandiset-assets", "000690", "--page", str(page)],
        capture_output=True, text=True
    )
    
    try:
        assets_data = json.loads(result.stdout)["results"]["results"]
        # Look for any probe files
        current_probe_files = [asset for asset in assets_data 
                              if "_probe-" in asset["path"]
                              and "_ecephys.nwb" in asset["path"]]
        
        if current_probe_files:
            probe_files.extend(current_probe_files)
            print(f"Found {len(current_probe_files)} probe files on page {page}")
            break
            
        page += 1
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing response: {e}")
        break

# Print available probe files
print(f"Found {len(probe_files)} probe files:")
for i, asset in enumerate(probe_files[:5]):  # Show at most 5
    print(f"{i+1}. {asset['path']}")
    
if len(probe_files) > 5:
    print(f"... and {len(probe_files) - 5} more")

# Choose the first probe file
if not probe_files:
    print("No probe files found. Using a hardcoded asset ID for testing.")
    # Hardcoded asset ID for a probe file we found earlier
    asset_id = "2ada1f14-7cdd-41d9-a3b8-36b0664f52e2"  # This is the probe-0 file we examined earlier
    file_pattern = "sub-695763/sub-695763_ses-1317661297_probe-0_ecephys.nwb"
else:
    selected_asset = probe_files[0]
    file_pattern = selected_asset["path"]
    asset_id = selected_asset["asset_id"]
print(f"Found asset ID: {asset_id}")

# Get information about the NWB file
asset_url = f"https://api.dandiarchive.org/api/assets/{asset_id}/download/"
print(f"Getting info for NWB file: {file_pattern}")
result = subprocess.run(
    ["python", "tools_cli.py", "nwb-file-info", "000690", asset_url],
    capture_output=True, text=True
)

# Parse the output to get unit properties
print("Analyzing unit data...")
output_lines = result.stdout.split("\n")

# Extract relevant information from the output
unit_info = []
in_units_section = False
current_unit = {}

for line in output_lines:
    # Look for evidence of units data
    if "nwb.units" in line:
        in_units_section = True
    
    if in_units_section and line.strip() and "nwb.units." in line:
        parts = line.split("#")
        if len(parts) > 1 and "(" in parts[1]:
            property_name = line.split(".")[2].split(" ")[0]
            property_type = parts[1].strip()[1:].split(")")[0]
            
            # Save unit property info
            if property_name not in ["id", "to_dataframe", "spike_times", "spike_amplitudes", "waveform_mean"]:
                print(f"Found unit property: {property_name} ({property_type})")
                unit_info.append({
                    "property": property_name,
                    "type": property_type
                })

print(f"\nIdentified {len(unit_info)} unit properties")

# Save the unit property information
with open("explore/unit_properties.txt", "w") as f:
    f.write(f"Unit properties for {file_pattern}:\n")
    f.write("=" * 50 + "\n\n")
    for prop in unit_info:
        f.write(f"{prop['property']} ({prop['type']})\n")

print("\nAnalysis complete!")