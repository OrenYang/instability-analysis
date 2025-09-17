import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths to your files
new_radii_csv = "/Users/orenyang/Desktop/instability_analysis/Kr_on_D2/sch/output2/radius_results.csv"
timing_radii_csv = "/Users/orenyang/Desktop/instability_analysis/Kr_on_D2/sch/output2/results.csv"

# Load both
df_new = pd.read_csv(new_radii_csv)
df_old = pd.read_csv(timing_radii_csv)

# Strip spaces from column names
df_new.columns = df_new.columns.str.strip()
df_old.columns = df_old.columns.str.strip()

# Normalize filenames
def normalize_name(name):
    return os.path.splitext(name)[0].lower()

df_new["key"] = df_new["Shot"].apply(normalize_name)
df_old["key"] = df_old["Image"].apply(normalize_name)

# Merge on key
df = pd.merge(df_old, df_new, on="key", how="inner")

# Extract shot number from filename (assumes it's the first part before '_')
df["Shot"] = df["key"].apply(lambda x: int(x.split('_')[0]))

# Plot comparison vs Shot
plt.figure(figsize=(8,6))
plt.scatter(df["Shot"], df["Pinch Radius (mm)"], label="Provided Radii", marker="o")
plt.scatter(df["Shot"], df["Radius_mm"], label="NPZ-derived Radii", marker="x")

plt.xlabel("Shot Number")
plt.ylabel("Radius (mm)")
plt.title("Pinch Radius vs Shot Number")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
