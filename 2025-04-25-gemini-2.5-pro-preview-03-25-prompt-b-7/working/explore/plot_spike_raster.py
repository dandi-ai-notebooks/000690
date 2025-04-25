# Explore Units data: Plot spike times for a few units if available.

import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme()

# Load NWB file
url = "https://api.dandiarchive.org/api/assets/ba8760f9-91fe-4c1c-97e6-590bed6a783b/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, load_namespaces=True)
nwb = io.read()

plot_saved = False
if hasattr(nwb, 'units') and nwb.units is not None:
    print("Found units data.")
    units_df = nwb.units.to_dataframe()

    if not units_df.empty:
        # Select a few unit IDs (use index if 'unit_id' column doesn't exist or is just range)
        num_units_to_plot = min(5, len(units_df))
        unit_indices = range(num_units_to_plot)
        unit_ids = units_df.index[unit_indices].tolist() # Use DataFrame index as unit IDs

        # Select time range (e.g., first 60 seconds)
        time_start = 0
        time_end = 60

        spike_times_list = []
        valid_unit_ids = []
        for unit_idx in unit_indices:
            # Spike times are stored relative to the start of the segment in the 'spike_times' column
            all_spikes = nwb.units['spike_times'][unit_idx]
            # Filter spikes within the desired time range
            spikes_in_range = all_spikes[(all_spikes >= time_start) & (all_spikes <= time_end)]
            if len(spikes_in_range) > 0:
                spike_times_list.append(spikes_in_range)
                valid_unit_ids.append(unit_ids[unit_idx]) # Keep track of units with spikes in range

        if spike_times_list:
            # Create raster plot
            plt.figure(figsize=(12, 6))
            plt.eventplot(spike_times_list, color='black', linelengths=0.75)
            plt.yticks(ticks=np.arange(len(valid_unit_ids)), labels=valid_unit_ids) # Use actual unit IDs for labels
            plt.xlabel("Time (s)")
            plt.ylabel("Unit ID")
            plt.title(f"Spike Raster Plot ({time_end - time_start} sec)")
            plt.xlim(time_start, time_end)
            plt.tight_layout()

            # Save plot
            plt.savefig("explore/spike_raster.png")
            print("Saved plot to explore/spike_raster.png")
            plot_saved = True
        else:
            print("No spikes found in the selected time range for the chosen units.")
    else:
        print("Units table is empty.")
else:
    print("No units data found in this NWB file.")

# Close resources
io.close()
remote_file.close()

if not plot_saved:
    # Create an empty file if no plot was generated, to avoid errors in the next step
    with open("explore/spike_raster.png", 'w') as f:
        pass
    print("Created empty explore/spike_raster.png because no plot was generated.")