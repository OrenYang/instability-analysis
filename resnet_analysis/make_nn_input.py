import pandas as pd
import os
import numpy as np

# -----------------------------
# Load raw CSV
# -----------------------------
df = pd.read_csv("resnet_analysis/results.csv")

# -----------------------------
# Build NN dataframe
# -----------------------------
nn_df = pd.DataFrame()

# ---- Shot number (extract from image name) ----
# Example: 2725_MCP1_3.jpg → 2725
nn_df["shot"] = df["Image"].apply(
    lambda x: int(os.path.splitext(x)[0].split("_")[0])
)

# ---- Image name (strip extension) ----
nn_df["image_name"] = df["Image"].apply(
    lambda x: os.path.splitext(x)[0]
)

# -----------------------------
# Raw values
# -----------------------------
radius = df["Pinch Radius (mm)"]
mrti   = df["MRTI Instability (mm)"]
height = df["Pinch Height (mm)"]
angle  = df["Avg Flaring Angle (deg)"]

# -----------------------------
# Safety: remove invalid heights
# -----------------------------
valid = height.notna() & (height > 0)

radius = radius[valid]
mrti   = mrti[valid]
height = height[valid]
angle  = angle[valid]
nn_df  = nn_df.loc[valid].reset_index(drop=True)

# -----------------------------
# PHYSICAL NORMALIZATION
# -----------------------------
nn_df["radius_norm"] = radius / height
nn_df["mrti_amplitude_norm"] = mrti / height

# Angle should NOT be height-normalized
nn_df["avg_angle"] = angle

# Keep pinch height (needed later!)
nn_df["pinch_height"] = height.values

# -----------------------------
# Drop invalid rows
# -----------------------------
nn_df = nn_df.dropna(
    subset=[
        "radius_norm",
        "mrti_amplitude_norm",
        "avg_angle",
        "pinch_height"
    ]
).reset_index(drop=True)

# -----------------------------
# Save
# -----------------------------
nn_df.to_csv("nn_training_labels_normalized_6.csv", index=False)

print("Saved:", len(nn_df), "physically-normalized samples")
