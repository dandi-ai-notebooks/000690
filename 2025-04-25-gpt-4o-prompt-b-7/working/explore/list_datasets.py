# Script to list available datasets in the NWB file
# This will help identify correct paths for dataset exploration

import h5py
import remfile

# Define the URL for the NWB file
url = "https://api.dandiarchive.org/api/assets/fbcd4fe5-7107-41b2-b154-b67f783f23dc/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)

# Function to recursively list datasets in the file
def list_datasets(name, node):
    if isinstance(node, h5py.Dataset):
        print(node.name)

# Apply the function to list datasets
h5_file.visititems(list_datasets)

# Close the file
h5_file.close()