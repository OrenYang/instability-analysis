import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# --- Open folder dialogs ---
root = tk.Tk()
root.withdraw()

fit_folder = filedialog.askdirectory(title="Select folder with *_erf_model_fit.npz files")
if not fit_folder:
    print("No fit folder selected.")
    exit()

csv_folder = filedialog.askdirectory(title="Select folder with CSV files")
if not csv_folder:
    print("No CSV folder selected.")
    exit()

# --- Collect files ---
fit_files = sorted([f for f in os.listdir(fit_folder) if f.endswith(".npz") and "erf_model_fit" in f])
csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith(".csv")])

if not fit_files:
    print("No *_erf_model_fit.npz files found.")
    exit()

# --- Extract psi values ---
def get_psi_from_name(filename):
    base = os.path.splitext(filename)[0]
    parts = base.split("_")[0].split("-")
    if len(parts) >= 2:
        try:
            return float(parts[0])
        except ValueError:
            return np.nan
    return np.nan

psi_values = np.array([get_psi_from_name(f) for f in fit_files])
norm = Normalize(vmin=np.nanmin(psi_values), vmax=np.nanmax(psi_values))
cmap = plt.cm.plasma
sm = ScalarMappable(norm=norm, cmap=cmap)

# --- Plot Radius vs Time ---
fig1, ax1 = plt.subplots(figsize=(8, 6))

for f, psi in zip(fit_files, psi_values):
    psi_label = str(int(psi)) if np.isfinite(psi) else "?"
    base = os.path.splitext(f)[0]
    parts = base.split("_")[0].split("-")
    mus = parts[1] if len(parts) >= 2 else "?"

    color = cmap(norm(psi)) if np.isfinite(psi) else "gray"

    # Plot CSV points if available
    csv_name = f"{parts[0]}-{parts[1]}.csv" if len(parts) >= 2 else None
    csv_path = os.path.join(csv_folder, csv_name) if csv_name else None
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if df.shape[1] >= 2:
            t_csv, r_csv = df.iloc[:, 0], df.iloc[:, 1]
            ax1.scatter(t_csv, r_csv, color=color, alpha=0.4, s=20, edgecolors="none")

    # Plot fit line
    data = np.load(os.path.join(fit_folder, f))
    t, r = data.get("t"), data.get("r")
    label = f"{psi_label} psi, {mus} µs"
    if t is not None and r is not None:
        ax1.plot(t, r, color=color, label=label, linewidth=2)

ax1.set_xlabel("Time (µs)")
ax1.set_ylabel("Radius (mm)")
ax1.set_title("Radius vs Time (raw data + erf_model fits)")
fig1.colorbar(sm, ax=ax1, label="Pressure (psi)")
ax1.legend()
fig1.tight_layout()

# --- Plot Velocity vs Time ---
cmap = plt.cm.viridis
sm = ScalarMappable(norm=norm, cmap=cmap)
fig2, ax2 = plt.subplots(figsize=(8, 6))

for f, psi in zip(fit_files, psi_values):
    psi_label = str(int(psi)) if np.isfinite(psi) else "?"
    base = os.path.splitext(f)[0]
    parts = base.split("_")[0].split("-")
    mus = parts[1] if len(parts) >= 2 else "?"
    color = cmap(norm(psi)) if np.isfinite(psi) else "gray"

    # Plot velocity line
    data = np.load(os.path.join(fit_folder, f))
    t, v = data.get("t"), data.get("v")
    label = f"{psi_label} psi, {mus} µs"
    if t is not None and v is not None:
        ax2.plot(t, v, color=color, label=label, linewidth=2)

ax2.set_xlabel("Time (µs)")
ax2.set_ylabel("Velocity (km/s)")
ax2.set_title("Velocity vs Time (erf_model fits)")
fig2.colorbar(sm, ax=ax2, label="Pressure (psi)")
ax2.legend()
fig2.tight_layout()

plt.show()
