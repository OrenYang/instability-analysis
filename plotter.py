import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("output/output.csv")

x = 'Timing (ns)'
y = 'Average Angle (deg)'

# Convert columns to numeric, coercing errors
df[x] = pd.to_numeric(df[x], errors='coerce')
df[y] = pd.to_numeric(df[y], errors='coerce')

# Drop rows with missing data
df = df.dropna(subset=[x, y])

# Plotting
plt.scatter(df[x], df[y], marker='o')
plt.xlabel("Time (ns)")
plt.ylabel("flaring angle (deg)")
plt.tight_layout()
plt.show()
