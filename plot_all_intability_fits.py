"""
Batch plotting of instability amplitude fits from NPZ files.
Only the fitted curves are shown in the legend.
Experimental points with error bars are plotted but not labeled.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import matplotlib
matplotlib.use("TkAgg")

# --- Truncate colormap to avoid very light/white colors ---
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list("trunc", colors)

# --- Open folder dialog for NPZ ---
root = tk.Tk()
root.withdraw()
folder = filedialog.askdirectory(title="Select folder with *_exp_fit_limited.npz files")
if not folder:
    print("No folder selected.")
    exit()

# --- Gather NPZ files ---
files = sorted([f for f in os.listdir(folder) if f.endswith(".npz")])
if not files:
    print("No NPZ files found.")
    exit()

# --- Extract psi values for color mapping ---
psi_values = []
for f in files:
    base = os.path.splitext(f)[0]
    parts = base.split("-")
    try:
        psi_values.append(float(parts[0]))
    except:
        psi_values.append(np.nan)
psi_values = np.array(psi_values)

# --- Setup colormap ---
cmap = truncate_colormap(cm.Greys, 0.4, 1.0)
norm = Normalize(vmin=np.nanmin(psi_values), vmax=np.nanmax(psi_values))
sm = ScalarMappable(norm=norm, cmap=cmap)

# --- Create figure ---
fig, ax = plt.subplots(figsize=(8, 6))

# --- Loop through NPZs ---
for f, psi_val in zip(files, psi_values):
    path = os.path.join(folder, f)
    data = np.load(path)

    x = data["x"]
    y = data["y"]
    y_err_exp = data.get("y_err_exp", None)

    x_fit = data["x_fit"]
    y_fit = data["y_fit"]
    y_fit_std = data.get("y_std", None)

    # --- Determine label from filename ---
    base = os.path.splitext(f)[0]
    base = base.replace("_exp_fit_limited", "")
    parts = base.split("_")[0].split("-")

    try:
        if len(parts) == 2:
            psi, timing = parts
            label = f"{psi} psi, -{timing} µs"
        elif len(parts) == 4:
            psi1, timing1, psi2, timing2 = parts
            label = (
                f"Liner: {psi1} psi, -{timing1} µs | "
                f"Target: {psi2} psi, -{timing2} µs"
            )
        else:
            label = base
    except:
        label = base

    color = cmap(norm(psi_val)) if np.isfinite(psi_val) else "gray"

    # --- Plot experimental points with error bars (no label) ---
    if y_err_exp is not None:
        ax.errorbar(x, y, yerr=y_err_exp, fmt='o', color=color, markersize=4, capsize=3, alpha=0.4)
    else:
        ax.scatter(x, y, s=20, color=color, alpha=0.4)

    # --- Plot fitted curve (with label) ---
    ax.plot(x_fit, y_fit, color=color, alpha=1, label=label)

    # --- Plot 1σ fit uncertainty ---
    if y_fit_std is not None:
        ax.fill_between(x_fit, y_fit - y_fit_std, y_fit + y_fit_std, color=color, alpha=0.3)

# --- Finalize plot ---
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Instability Amplitude [mm]")
ax.set_ylim(0,3)
ax.legend(fontsize=10)
fig.colorbar(sm, ax=ax, label="Plenum Pressure [psi]")
fig.tight_layout()

parent_dir = os.path.dirname(folder)
fig.savefig(os.path.join(parent_dir, "instabilityAmplitude_plots.png"), dpi=300)

plt.show()
