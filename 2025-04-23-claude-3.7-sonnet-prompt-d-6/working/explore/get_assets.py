"""
This script retrieves and organizes information about the assets in Dandiset 000690.
It categorizes files by subject and type to help understand the dataset organization.
"""

import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os

# Run the command to get the assets
result = subprocess.run(['python', '../tools_cli.py', 'dandiset-assets', '000690'], 
                        capture_output=True, text=True, check=True)

# Parse the JSON output
data = json.loads(result.stdout)

# Create a list to store asset information
asset_info = []

for asset in data['results']['results']:
    asset_id = asset['asset_id']
    path = asset['path']
    size_mb = asset['size'] / (1024 * 1024)  # Convert to MB
    
    # Extract subject, session and file type information
    parts = path.split('/')
    subject = parts[0]
    filename = parts[1]
    
    # Determine file type based on filename
    if '_image.nwb' in filename:
        file_type = 'image'
    elif '_probe-' in filename and '_ecephys.nwb' in filename:
        probe_num = filename.split('_probe-')[1].split('_')[0]
        file_type = f'probe-{probe_num}'
    elif '.nwb' in filename and not ('_image.nwb' in filename) and not ('_ecephys.nwb' in filename):
        file_type = 'main'
    else:
        file_type = 'other'
    
    asset_info.append({
        'asset_id': asset_id,
        'path': path,
        'subject': subject,
        'filename': filename,
        'file_type': file_type,
        'size_mb': size_mb
    })

# Convert to DataFrame for easier analysis
df = pd.DataFrame(asset_info)

# Print summary information
print(f"Total number of assets: {len(df)}")
print("\nSubjects:")
print(df['subject'].value_counts())

print("\nFile types:")
print(df['file_type'].value_counts())

# Save file counts per subject 
subject_file_counts = df.groupby(['subject', 'file_type']).size().unstack().fillna(0)
print("\nFile counts per subject:")
print(subject_file_counts)

# Create a bar chart showing the distribution of file sizes by file type
plt.figure(figsize=(10, 6))
df.boxplot(column='size_mb', by='file_type')
plt.title('File Sizes by Type')
plt.ylabel('Size (MB)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('file_size_distribution.png')

# Save the dataframe to a file for potential use in other scripts
df.to_csv('asset_info.csv', index=False)

print("\nDataframe saved to asset_info.csv")
print("Plot saved to file_size_distribution.png")