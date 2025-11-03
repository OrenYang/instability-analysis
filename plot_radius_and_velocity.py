import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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

# --- Collect npz files ---
files = sorted([
    f for f in os.listdir(folder)
    if f.endswith(".npz") and "erf_model_fit" in f
])

if not files:
    print("No *_erf_model_fit.npz files found in the folder.")
    exit()

# --- Extract psi and timing values for color mapping ---
psi_values, timing_values = [], []
for f in files:
    base = os.path.splitext(f)[0]
    parts = base.split("_")[0].split("-")
    if len(parts) >= 2:
        try:
            psi_values.append(float(parts[0]))
            timing_values.append(float(parts[1]))
        except ValueError:
            psi_values.append(np.nan)
            timing_values.append(np.nan)
    else:
        psi_values.append(np.nan)
        timing_values.append(np.nan)

psi_values = np.array(psi_values)
timing_values = np.array(timing_values)
color_values = psi_values

# --- Color mapping ---
norm = Normalize(vmin=np.nanmin(color_values), vmax=np.nanmax(color_values))
cmap_r = plt.cm.plasma
cmap_v = plt.cm.viridis
sm_r = ScalarMappable(norm=norm, cmap=cmap_r)
sm_v = ScalarMappable(norm=norm, cmap=cmap_v)

# --- Plot radius vs time ---
fig1, ax1 = plt.subplots(figsize=(8, 6))
for f, val in zip(files, color_values):
    path = os.path.join(folder, f)
    data = np.load(path)
    t = data.get("t")
    r = data.get("r")

    # --- Determine label ---
    base = os.path.splitext(f)[0]
    parts = base.split("_")[0].split("-")
    label = base  # fallback
    try:
        if len(parts) == 2:
            psi, timing = parts
            label = f"{psi} psi, -{timing} µs"
        elif len(parts) == 4:
            psi1, timing1, psi2, timing2 = parts
            label = f"Liner: {psi1} psi, -{timing1} µs | Target: {psi2} psi, -{timing2} µs"
    except Exception:
        pass

    color = cmap_r(norm(val)) if np.isfinite(val) else "gray"

    # --- Plot raw CSV data if available ---
    if csv_folder:
        csv_name = base.replace("_erf_model_fit", "") + ".csv"
        csv_path = os.path.join(csv_folder, csv_name)
        if os.path.exists(csv_path):
            try:
                raw = np.loadtxt(csv_path, delimiter=",", skiprows=1)
                t_raw, r_raw = raw[:, 0], raw[:, 1]
                ax1.scatter(t_raw, r_raw, color=color, s=8, alpha=0.4)
            except Exception:
                pass

    if t is not None and r is not None:
        ax1.plot(t, r, label=label, color=color)

ax1.set_xlabel("Time [ns]")
ax1.set_ylabel("Radius [mm]")
cbar1 = fig1.colorbar(sm_r, ax=ax1, label="Plenum Pressure [psi]")
ax1.legend()
fig1.tight_layout()


# --- Plot velocity vs time ---
fig2, ax2 = plt.subplots(figsize=(8, 6))
for f, val in zip(files, color_values):
    path = os.path.join(folder, f)
    data = np.load(path)
    t = data.get("t")
    v = data.get("v")

    # --- Determine label ---
    base = os.path.splitext(f)[0]
    parts = base.split("_")[0].split("-")
    label = base  # fallback
    try:
        if len(parts) == 2:
            psi, timing = parts
            label = f"{psi} psi, -{timing} µs"
        elif len(parts) == 4:
            psi1, timing1, psi2, timing2 = parts
            label = f"Liner: {psi1} psi, -{timing1} µs | Target: {psi2} psi, -{timing2} µs"
    except Exception:
        pass

    color = cmap_r(norm(val)) if np.isfinite(val) else "gray"

    color = cmap_v(norm(val)) if np.isfinite(val) else "gray"

    # --- Plot fitted curve ---
    if t is not None and v is not None:
        ax2.plot(t, v, label=label, color=color)

ax2.set_xlabel("Time [ns]")
ax2.set_ylabel("Velocity [km/s]")
cbar2 = fig2.colorbar(sm_v, ax=ax2, label="Plenum Pressure [psi]")
ax2.legend()
fig2.tight_layout()

plt.show()
