# This script explores the basic properties of Dandiset 000690,
# including how to load data from it and basic metadata

import matplotlib.pyplot as plt
import numpy as np
from dandi.dandiapi import DandiAPIClient
import pandas as pd

# Save plots instead of showing them
plt.ioff()

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("000690")

# Print basic information about the Dandiset
metadata = dandiset.get_raw_metadata()
print(f"Dandiset name: {metadata['name']}")
print(f"Dandiset URL: {metadata['url']}")
print(f"Dandiset description: {metadata['description'][:500]}...")
print(f"Dandiset citation: {metadata['citation']}")

# Get list of assets in the Dandiset
assets = list(dandiset.get_assets())
print(f"\nFound {len(assets)} assets in the dataset")

# Summarize assets by file type
file_types = {}
for asset in assets:
    ext = asset.path.split('.')[-1]
    if ext not in file_types:
        file_types[ext] = 0
    file_types[ext] += 1
    
print("\nAsset types:")
for ext, count in file_types.items():
    print(f"- {ext}: {count}")

# Print a few example paths to understand the organization
print("\nExample asset paths:")
for asset in assets[:10]:
    print(f"- {asset.path} (Size: {asset.size/1e9:.2f} GB)")
    
# Create a histogram of file sizes
sizes = [asset.size/1e9 for asset in assets]  # Convert to GB
plt.figure(figsize=(10, 6))
plt.hist(sizes, bins=20)
plt.xlabel('File Size (GB)')
plt.ylabel('Number of Files')
plt.title('Distribution of File Sizes in Dandiset 000690')
plt.savefig('explore/file_size_distribution.png')
plt.close()

# Analyze the subject and session structure
subjects = {}
for asset in assets:
    path_parts = asset.path.split('/')
    if len(path_parts) > 1:
        subject = path_parts[0]
        if subject not in subjects:
            subjects[subject] = {'sessions': set(), 'file_types': {}}
        
        # Extract session from filename
        if len(path_parts) > 1:
            filename = path_parts[-1]
            parts = filename.split('_')
            if len(parts) > 1:
                session = None
                for part in parts:
                    if part.startswith('ses-'):
                        session = part
                        break
                if session:
                    subjects[subject]['sessions'].add(session)
            
            # Track file types per subject
            if '.' in filename:
                ext = filename.split('.')[-1]
                if ext not in subjects[subject]['file_types']:
                    subjects[subject]['file_types'][ext] = 0
                subjects[subject]['file_types'][ext] += 1

# Print subject and session information
print("\nSubject and Session Summary:")
for subject, info in subjects.items():
    print(f"Subject {subject}:")
    print(f"  - Number of sessions: {len(info['sessions'])}")
    print(f"  - Sessions: {', '.join(sorted(info['sessions']))}")
    print(f"  - File types: {info['file_types']}")

# Save subject count summary as a chart
subject_file_counts = {subject: sum(info['file_types'].values()) for subject, info in subjects.items()}
plt.figure(figsize=(12, 6))
plt.bar(subject_file_counts.keys(), subject_file_counts.values())
plt.xlabel('Subject')
plt.ylabel('Number of Files')
plt.title('Files per Subject in Dandiset 000690')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('explore/files_per_subject.png')
plt.close()