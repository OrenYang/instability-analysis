import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from scipy.optimize import curve_fit

# -------------------- USER OPTIONS --------------------
save = True
plot = True
make_title = False

matplotlib.rcParams.update({
    "font.size": 16,
    })
# ------------------------------------------------------

# --- Tkinter file dialogs ---
root = tk.Tk()
root.withdraw()

# Select amplitude CSV
f_csv = filedialog.askopenfilename(title="Select Instability Amplitude CSV")
if not f_csv:
    raise ValueError("No CSV selected")

# Select optional radius fit .npz file
f_fit = filedialog.askopenfilename(
    title="Select Radius Fit .npz File (optional)",
    filetypes=[("NPZ files", "*.npz")]
)
use_radius_fit = bool(f_fit)

# --- Load data ---
df = pd.read_csv(f_csv)
x_all = df["Instability Timing"].to_numpy()
y_all = df["Instability Mrti_Amplitude"].to_numpy()
mask_all = np.isfinite(x_all) & np.isfinite(y_all)
x_all, y_all = x_all[mask_all], y_all[mask_all]

# Check if CSV has standard deviation column
if "instability__mrti_err" in df.columns:
    y_err_all = df["instability__mrti_err"].to_numpy()
    y_err_all = y_err_all[mask_all]
else:
    y_err_all = None

# --- Determine cutoff time from radius fit (if provided) ---
if use_radius_fit:
    radius_fit = np.load(f_fit)
    t_radius = radius_fit["t"]
    r_radius = radius_fit["r"]
    t_min = t_radius[np.argmin(r_radius)]
    print(f"✅ Fitting up to t = {t_min:.2f} ns (min radius)")
    mask_fit = x_all <= t_min
else:
    t_min = np.max(x_all)
    print(f"✅ No radius fit provided; using all data for fit")
    mask_fit = np.ones_like(x_all, dtype=bool)

x_fit_data = x_all[mask_fit]
y_fit_data = y_all[mask_fit]
if y_err_all is not None:
    y_fit_err = y_err_all[mask_fit]
else:
    y_fit_err = None

# --- Define exponential model ---
def exp_model(t, A0, gamma):
    return A0 * np.exp(gamma * t)

# --- Fit data using absolute sigma ---
popt, pcov = curve_fit(
    exp_model,
    x_fit_data,
    y_fit_data,
    p0=[0.1, 1/200],
    sigma=y_fit_err,
    absolute_sigma=True
)
A0_fit, gamma_fit = popt
A0_err, gamma_err = np.sqrt(np.diag(pcov))

# --- Generate fitted curve ---
x_fit = np.linspace(min(x_fit_data), max(x_fit_data), 200)
y_fit = exp_model(x_fit, *popt)

# --- Compute fit uncertainty (1σ propagation) ---
J = np.zeros((len(x_fit), len(popt)))
eps = 1e-8
for i in range(len(popt)):
    p_eps = popt.copy()
    p_eps[i] += eps
    J[:, i] = (exp_model(x_fit, *p_eps) - y_fit) / eps
y_var = np.sum(J @ pcov * J, axis=1)
y_std = np.sqrt(y_var)

# --- Plot ---
plt.figure(figsize=(7, 5))

# Plot data with error bars if available
if y_err_all is not None:
    plt.errorbar(x_all, y_all, yerr=y_err_all, fmt='o', color="green", label="Experimental data", capsize=3,)
else:
    plt.scatter(x_all, y_all, color="green", label="Experimental data", s=30)

plt.plot(x_fit, y_fit, color="orange", label=f"Exponential fit")#: A₀={A0_fit:.2f}±{A0_err:.2f}, γ={gamma_fit:.2e}±{gamma_err:.2e}
plt.fill_between(x_fit, y_fit - y_std, y_fit + y_std, color="orange", alpha=0.3)  # No legend entry

if use_radius_fit:
    plt.axvline(t_min, color="gray", linestyle="--", alpha=0.5, label="Min radius time")

plt.ylim(0, 3)
plt.xlabel("Time [ns]")
plt.ylabel("Instability Amplitude [mm]")

# --- Title from filename ---
base_name = os.path.splitext(os.path.basename(f_csv))[0]
parts = base_name.split('-')
title = base_name
try:
    if len(parts) == 2:
        psi, timing = parts
        title = f"{psi} PSI, -{timing} µs"
    elif len(parts) == 4:
        psi1, timing1, psi2, timing2 = parts
        title = f"Liner: {psi1} PSI, {timing1} ns | Target: {psi2} PSI, {timing2} ns"
except Exception:
    pass

if make_title:
    plt.title(title)

plt.legend()
plt.tight_layout()

# --- Save results ---
if save:
    base_dir = os.path.dirname(f_csv)
    plot_dir = os.path.join(base_dir, "plots")
    fit_dir = os.path.join(base_dir, "fits")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(fit_dir, exist_ok=True)

    plot_path = os.path.join(plot_dir, f"{base_name}_exp_fit.svg")
    plt.savefig(plot_path, format="svg")

    plot_path = os.path.join(plot_dir, f"{base_name}_exp_fit.png")
    fit_path = os.path.join(fit_dir, f"{base_name}_exp_fit.npz")

    plt.savefig(plot_path, dpi=300)
    print(f"✅ Plot saved to: {plot_path}")

    np.savez(
        fit_path,
        x=x_all,
        y=y_all,
        y_err_exp=y_err_all,
        x_fit_data=x_fit_data,
        y_fit_data=y_fit_data,
        x_fit=x_fit,
        y_fit=y_fit,
        y_std=y_std,
        A0=A0_fit,
        gamma=gamma_fit,
        A0_err=A0_err,
        gamma_err=gamma_err,
        cov=pcov,
        t_limit=t_min,
    )
    print(f"✅ Fit data saved to: {fit_path}")

if plot:
    plt.show()
else:
    plt.close()
