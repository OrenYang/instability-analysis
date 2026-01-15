"""
Batch plotting of instability amplitude fits from NPZ files.

Workflow:
1. Opens a file dialog to select a folder containing NPZ files
   with names ending in *_exp_fit_limited.npz.
2. Loads each NPZ file, which should contain:
   - x: experimental time points [ns]
   - y: measured instability amplitudes [mm]
   - y_err_exp (optional): experimental error bars
   - x_fit: time points for fitted curve
   - y_fit: fitted instability amplitude
   - y_std (optional): standard deviation of the fit
3. Determines the label for each fitted curve from the filename.
   - Supports single or double psi/timing entries for liner and target.
4. Plots:
   - Experimental points (with error bars if available) in a light, semi-transparent color (no legend label)
   - Fitted curves (with labels) in full color
   - Optional 1σ uncertainty bands around the fit
5. Color maps curves by plenum pressure (psi) using a matplotlib colormap.
6. Adds axis labels, legend (fitted curves only), and a colorbar for psi.

Inputs:
- Folder of NPZ files named as *_exp_fit_limited.npz, where filenames
  encode pressure and timing information.

Outputs:
- Displays a figure with all experimental points, fits, and uncertainties.
- Saves figure as 'instabilityAmplitude_plots.png' in the parent directory
  of the selected folder.
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
from collections import Counter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

matplotlib.rcParams.update({
    "font.size": 12,
    })
legend_font=8

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
#cmap = cm.plasma
norm = Normalize(vmin=np.nanmin(psi_values), vmax=np.nanmax(psi_values))
sm = ScalarMappable(norm=norm, cmap=cmap)

# --- Map unique psi values to markers ---
psi_counts = Counter([p for p in psi_values if np.isfinite(p)])
markers = [None,'x','o', '^', 'v', '*', 'd', 'p', '|', 'P']  # marker cycle
markers_scatter = ['o', '^', 'v', 'D', '>', 'p', '*', '<']
psi_marker_index = {psi: 0 for psi in psi_counts if psi_counts[psi] > 1}

# --- Create figure ---
fig, ax = plt.subplots(figsize=(6, 4.5))
fig.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.95)

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
            label = f"{psi} PSI, -{timing} µs"
        elif len(parts) == 4:
            psi1, timing1, psi2, timing2 = parts
            label = (
                f"Liner: {psi1} PSI, -{timing1} µs | "
                f"Target: {psi2} PSI, -{timing2} µs"
            )
        else:
            label = base
    except:
        label = base

    color = cmap(norm(psi_val)) if np.isfinite(psi_val) else "gray"
    if np.isfinite(psi_val) and psi_counts[psi_val] > 1:
        marker = markers[psi_marker_index[psi_val] % len(markers)]
        marker_scatter = markers_scatter[psi_marker_index[psi_val] % len(markers_scatter)]
        psi_marker_index[psi_val] += 1
    else:
        marker_scatter = 'o'
        marker = None  # single line → no marker

    # --- Plot experimental points with error bars (no label) ---
    if y_err_exp is not None:
        ax.errorbar(x, y, yerr=y_err_exp, fmt='o', color=color, marker=marker_scatter, markersize=4, capsize=3, alpha=0.4)
    else:
        ax.scatter(x, y, s=20, color=color, marker=marker_scatter, alpha=0.4)

    # --- Plot fitted curve (with label) ---
    step = len(x_fit) // 3
    ax.plot(x_fit, y_fit, color=color, marker=marker, markevery=slice(step, -step, step), alpha=1, label=label)

    # --- Plot 1σ fit uncertainty ---
    if y_fit_std is not None:
        ax.fill_between(x_fit, y_fit - y_fit_std, y_fit + y_fit_std, color=color, alpha=0.3)

# --- Finalize plot ---
ax.set_xlabel("Time [ns]")
ax.set_ylabel("Instability Amplitude [mm]")
ax.set_ylim(0,3)
ax.legend(fontsize=legend_font)


cbar = fig.colorbar(sm, ax=ax, label="Plenum Pressure [PSI]")
cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#fig.tight_layout()

parent_dir = os.path.dirname(folder)
fig.savefig(os.path.join(parent_dir, "instabilityAmplitude_plots.svg"), format="svg")

plt.show()
