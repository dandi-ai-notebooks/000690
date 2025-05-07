# This script creates a summary of stimulus information based on our previous exploration
# Results will be saved to text files and plots

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Based on our previous exploration, define stimulus information
stimuli = [
    "Stim01_SAC_Wd15_Vel2_White_loop_presentations",
    "Stim02_SAC_Wd45_Vel2_White_loop_presentations",
    "Stim03_SAC_Wd15_Vel2_White_oneway_1_presentations",
    "Stim04_SAC_Wd15_Vel2_Black_loop_presentations",
    "Stim05_SAC_Wd15_Vel2_White_oneway_2_presentations",
    "Stim06_SAC_Wd15_Vel2_White_scramble_presentations",
    "Stim07_DOT_Wd15_Vel2_White_loop_presentations",
    "Stim08_SAC_Wd15_Vel6_White_loop_presentations",
    "Stim09_UD_Wd15_Vel2_White_loop_presentations",
    "Stim10_ROT_Wd15_Vel2_White_loop_presentations",
    "Stim11_Ring_Wd15_Vel2_White_loop_presentations",
    "Stim12_Disk_Wd15_Vel2_White_loop_presentations",
    "Stim13_SAC_Wd15_Vel2_Disco_loop_presentations", 
    "Stim14_natmovie_10secFast_EagleSwoop_presentations",
    "Stim15_natmovie_20sec_EagleSwoop_presentations",
    "Stim16A_natmovie_20sec_Flipped_A_EagleSwoop_presentations",
    "Stim16B_natmovie_20sec_Flipped_B_EagleSwoop_presentations",
    "Stim17A_natmovie_20sec_Occluded1to1_A_EagleSwoop_presentations",
    "Stim17B_natmovie_20sec_Occluded1to1_B_EagleSwoop_presentations"
]

# Decode stimulus descriptions
stimulus_info = []
for stim in stimuli:
    # Skip stimulus prefix, presentations suffix and split by underscore
    parts = stim.replace("Stim", "").split("_presentations")[0].split("_")
    
    # Interpret the parts
    info = {"full_name": stim}
    
    # Extract stimulus type (SAC, DOT, UD, etc)
    if parts[0].isdigit():
        # Handle case like "01", extract just the type
        if len(parts) > 1:
            info["stim_type"] = parts[1] 
    else:
        info["stim_type"] = parts[0]
    
    # Extract width if present (Wd15, Wd45)
    width_parts = [p for p in parts if p.startswith("Wd")]
    if width_parts:
        info["width"] = width_parts[0].replace("Wd", "")
    
    # Extract velocity if present (Vel2, Vel6)
    vel_parts = [p for p in parts if p.startswith("Vel")]
    if vel_parts:
        info["velocity"] = vel_parts[0].replace("Vel", "")
    
    # Extract color if present (White, Black, Disco)
    color_keywords = ["White", "Black", "Disco"]
    color_parts = [p for p in parts if p in color_keywords]
    if color_parts:
        info["color"] = color_parts[0]
    
    # Check if it's a movie
    if "movie" in stim.lower() or "eagle" in stim.lower():
        info["is_movie"] = True
        info["stim_type"] = "NaturalMovie"
    else:
        info["is_movie"] = False
    
    # Check movement pattern
    if "loop" in stim.lower():
        info["pattern"] = "loop"
    elif "oneway" in stim.lower():
        info["pattern"] = "one-way"
    elif "scramble" in stim.lower():
        info["pattern"] = "scrambled"
    elif "flipped" in stim.lower():
        info["pattern"] = "flipped"
    elif "occluded" in stim.lower():
        info["pattern"] = "occluded"
    
    stimulus_info.append(info)

# Convert to DataFrame and save
stimulus_df = pd.DataFrame(stimulus_info)

# Organize stimuli by type
stim_types = stimulus_df.groupby("stim_type").size().reset_index(name="count")
stim_types = stim_types.sort_values(by="count", ascending=False)

# Save stimulus info
with open('explore/stimulus_summary.txt', 'w') as f:
    f.write("Stimulus Types:\n")
    for _, row in stim_types.iterrows():
        f.write(f"- {row['stim_type']}: {row['count']} variants\n")
    
    f.write("\nStimulus Details:\n")
    for info in stimulus_info:
        desc = [info["full_name"]]
        if "stim_type" in info:
            desc.append(f"Type: {info['stim_type']}")
        if "width" in info:
            desc.append(f"Width: {info['width']}°")
        if "velocity" in info:
            desc.append(f"Velocity: {info['velocity']}")
        if "color" in info:
            desc.append(f"Color: {info['color']}")
        if "pattern" in info:
            desc.append(f"Pattern: {info['pattern']}")
        if "is_movie" in info and info["is_movie"]:
            desc.append("Natural Movie")
        
        f.write("- " + ", ".join(desc) + "\n")

# Create bar chart of stimulus types
plt.figure(figsize=(10, 6))
plt.bar(stim_types["stim_type"], stim_types["count"])
plt.xlabel('Stimulus Type')
plt.ylabel('Count')
plt.title('Number of Variants per Stimulus Type')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('explore/stimulus_types.png', dpi=300)

print(f"Created summary in explore/stimulus_summary.txt")
print(f"Created plot in explore/stimulus_types.png")

# Create more detailed analysis of the SAC stimulus variants
sac_stims = stimulus_df[stimulus_df["stim_type"] == "SAC"]
print(f"\nSAC stimulus variants: {len(sac_stims)}")
for _, row in sac_stims.iterrows():
    details = []
    for col in ["width", "velocity", "color", "pattern"]:
        if col in row and not pd.isna(row[col]):
            details.append(f"{col}={row[col]}")
    print(f"- {row['full_name']}: {', '.join(details)}")