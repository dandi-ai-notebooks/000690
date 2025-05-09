# This script explores the high-level structure and metadata of the selected NWB file from Dandiset 000690.
# It prints: root attributes, subject/session info, and the structure and a sample from the electrodes table.

import pynwb
import h5py
import remfile
import pandas as pd

url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("NWB session description:", nwb.session_description)
print("NWB identifier:", nwb.identifier)
print("NWB session start time:", nwb.session_start_time)
print("NWB institution:", getattr(nwb, "institution", None))
print("NWB subject:")
if hasattr(nwb, "subject") and nwb.subject is not None:
    subj = nwb.subject
    print("  subject_id:", getattr(subj, "subject_id", None))
    print("  species:", getattr(subj, "species", None))
    print("  age:", getattr(subj, "age", None))
    print("  sex:", getattr(subj, "sex", None))
    print("  genotype:", getattr(subj, "genotype", None))
    print("  strain:", getattr(subj, "strain", None))
else:
    print("  No subject information found.")

# Electrode table exploration
if hasattr(nwb, "electrodes"):
    print("\nElectrodes table shape:", nwb.electrodes.to_dataframe().shape)
    print("Columns:", nwb.electrodes.to_dataframe().columns.tolist())
    print("First 5 rows of electrodes table:")
    print(nwb.electrodes.to_dataframe().head())
else:
    print("\nNo electrodes table found.")

# List acquisition keys
print("\nAcquisition keys:", list(nwb.acquisition.keys()))