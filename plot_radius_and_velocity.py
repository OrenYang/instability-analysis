'''
This script loads multiple *_erf_model_fit.npz files and plots their
 fitted radius and velocity curves with error bars / 1σ uncertainty bands.

 ---------------------- USER INPUTS ----------------------
 NOTE: Figures are saved one level ABOVE the selected NPZ folder.
 1. Select a folder containing *_erf_model_fit.npz files.
    These NPZs must contain at least: t, r, v, and optionally r_std, v_std.
 2. Optionally select a folder containing raw CSVs with experimental data.
    Raw CSV filenames must match the NPZ prefix (before _erf_model_fit).
 ---------------------------------------------------------
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

matplotlib.rcParams.update({
    "font.size": 12,
    })
legend_font=8

r_margin_top = 1.1
r_margin_bot = 0

v_margin_top = 0
v_margin_bot = 1

# --- Truncate colormap to avoid very light/white colors ---
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list("trunc", colors)

# --- Open folder dialog for npz ---
root = tk.Tk()
root.withdraw()
folder = filedialog.askdirectory(title="Select folder with *_erf_model_fit.npz files")
if not folder:
    print("No folder selected.")
    exit()

# --- Optional: open folder with raw CSVs ---
csv_folder = filedialog.askdirectory(title="(Optional) Select folder with raw CSVs (Cancel to skip)")
if not csv_folder:
    csv_folder = None

# --- Gather npz files ---
files = sorted([
    f for f in os.listdir(folder)
    if f.endswith(".npz") and "erf_model_fit" in f
])

if not files:
    print("No *_erf_model_fit.npz files found.")
    exit()

# --- Extract psi/timing for color mapping ---
psi_values = []
for f in files:
    base = os.path.splitext(f)[0]
    parts = base.split("_")[0].split("-")
    try:
        psi_values.append(float(parts[0]))
    except:
        psi_values.append(np.nan)
psi_values = np.array(psi_values)

# --- Extract target timing for color mapping ---
target_timing_values = []
for f in files:
    base = os.path.splitext(f)[0]
    parts = base.split("_")[0].split("-")
    try:
        if len(parts) == 4:
            # Liner: psi1-timing1, Target: psi2-timing2
            target_timing = float(parts[3])
        elif len(parts) == 2:
            # Only one timing value, use that
            target_timing = float(parts[1])
        else:
            target_timing = np.nan
    except:
        target_timing = np.nan
    target_timing_values.append(target_timing)

target_timing_values = np.array(target_timing_values)

#psi_values = target_timing_values


cmap_r = truncate_colormap(cm.Greys, 0.4, 1.0)
cmap_v = truncate_colormap(cm.Greys, 0.4, 1.0)
#cmap_r = cm.viridis
#cmap_v = cm.viridis
# --- Color normalization ---
norm = Normalize(vmin=np.nanmin(psi_values), vmax=np.nanmax(psi_values))
sm_r = ScalarMappable(norm=norm, cmap=cmap_r)
sm_v = ScalarMappable(norm=norm, cmap=cmap_v)

# Map unique psi values to markers
psi_counts = Counter([p for p in psi_values if np.isfinite(p)])
markers = [None,'x','o', '^', 'v', '*', 'd', 'p', '|', 'P']  # marker cycle
markers_scatter = ['o', '^', 'v', 'D', '>', 'p', '*', '<']
psi_marker_index = {psi: 0 for psi in psi_counts if psi_counts[psi] > 1}

# ------------------ PLOT RADIUS ------------------
fig1, ax1 = plt.subplots(figsize=(6, 4.5))
fig1.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.95)

for f, psi_val in zip(files, psi_values):
    path = os.path.join(folder, f)
    data = np.load(path)

    t = data.get("t")
    r = data.get("r")
    r_std = data.get("r_std")  # optional

    base = os.path.splitext(f)[0]
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

    color = cmap_r(norm(psi_val)) if np.isfinite(psi_val) else "gray"
    if np.isfinite(psi_val) and psi_counts[psi_val] > 1:
        marker = markers[psi_marker_index[psi_val] % len(markers)]
        marker_scatter = markers_scatter[psi_marker_index[psi_val] % len(markers_scatter)]
        psi_marker_index[psi_val] += 1
    else:
        marker_scatter = 'o'
        marker = None  # single line → no marker

    # --- Plot raw CSV ---
    if csv_folder:
        csv_name = base.replace("_erf_model_fit", "") + ".csv"
        csv_path = os.path.join(csv_folder, csv_name)
        if os.path.exists(csv_path):
            try:
                raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                t_raw, r_raw = raw[:, 0], raw[:, 1]
                # Experimental error bars if CSV has >=3 columns
                if raw.shape[1] >= 3:
                    r_err = raw[:, 2]
                    ax1.errorbar(t_raw, r_raw, yerr=r_err, fmt='o', color=color, marker=marker_scatter, markersize=4, alpha=0.4, linewidth=1, capsize=2)
                else:
                    ax1.scatter(t_raw, r_raw, color=color, marker=marker_scatter,  s=8, alpha=0.4)
            except:
                pass

    # --- Plot fit ---
    if t is not None and r is not None:
        step = len(t)//3
        ax1.plot(t, r, color=color, label=label, marker=marker, markersize=6, markevery=slice(step, -step, step))

        # --- Add 1σ shading (no label) ---
        if r_std is not None:
            ax1.fill_between(t, r - r_std, r + r_std, color=color, alpha=0.25, linewidth=0)

ymin, ymax = ax1.get_ylim()
ax1.set_ylim(ymin * r_margin_bot, ymax * r_margin_top)

ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("Radius [mm]")
cbar1 = fig1.colorbar(sm_r, ax=ax1, label="Plenum Pressure [PSI]")
cbar1.ax.yaxis.set_major_locator(MultipleLocator(0.1))
cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.legend(fontsize=legend_font)
#fig1.tight_layout()

# ------------------ PLOT VELOCITY ------------------
fig2, ax2 = plt.subplots(figsize=(6, 4.5))
fig2.subplots_adjust(left=0.18, right=0.95, bottom=0.15, top=0.95)

psi_marker_index = {psi: 0 for psi in psi_counts if psi_counts[psi] > 1}

for f, psi_val in zip(files, psi_values):
    path = os.path.join(folder, f)
    data = np.load(path)

    t = data.get("t")
    v = data.get("v")
    v_std = data.get("v_std")  # optional

    base = os.path.splitext(f)[0]
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

    color = cmap_v(norm(psi_val)) if np.isfinite(psi_val) else "gray"

    if t is not None and v is not None:
        if np.isfinite(psi_val) and psi_counts[psi_val] > 1:
            marker = markers[psi_marker_index[psi_val] % len(markers)]
            psi_marker_index[psi_val] += 1
        else:
            marker = None  # single line → no marker
        step = len(t)//3
        ax2.plot(t, v, color=color, label=label, marker=marker, markersize=6, markevery=slice(step,-step,step))

        # --- Add 1σ velocity shading (no legend) ---
        if v_std is not None:
            ax2.fill_between(t, v - v_std, v + v_std, color=color, alpha=0.20, linewidth=0)

## Cap Y-axis
# --- Determine global capped velocity range ---
global_vmin = np.inf
global_vmax = -np.inf

for f in files:
    data = np.load(os.path.join(folder, f))
    v = data.get("v")
    v_std = data.get("v_std")

    if v is None:
        continue

    real_min = np.min(v)
    real_max = np.max(v)

    if v_std is not None:
        std_min = np.min(v - v_std)
        std_max = np.max(v + v_std)
    else:
        std_min, std_max = real_min, real_max

    # --- cap uncertainty influence to ±50 km/s ---
    capped_min = max(std_min-20, real_min - 50)
    capped_max = min(std_max+20, real_max + 50)

    global_vmin = min(global_vmin, capped_min)
    global_vmax = max(global_vmax, capped_max)

# --- apply final y-limits ---
ax2.set_ylim(global_vmin*v_margin_bot, global_vmax*v_margin_top)


ax2.set_xlabel("Time [ns]")
ax2.set_ylabel("Velocity [km/s]")
cbar2 = fig2.colorbar(sm_v, ax=ax2, label="Plenum Pressure [PSI]")
cbar2.ax.yaxis.set_major_locator(MultipleLocator(0.1))
cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.legend(fontsize=legend_font)
#fig2.tight_layout()

# --- Save figures (saved into the selected NPZ folder) ---
parent_dir = os.path.dirname(folder)
fig1.savefig(os.path.join(parent_dir, "radius_plots.svg"), format="svg")
fig2.savefig(os.path.join(parent_dir, "velocity_plots.svg"), format="svg")

plt.show()
