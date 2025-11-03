import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from scipy.optimize import curve_fit

root = tk.Tk()
root.withdraw()

f = filedialog.askopenfilename(title="Select CSV")
if not f:
    raise ValueError("No CSV selected")

df = pd.read_csv(f)

# Extract relevant columns and drop NaNs
x = df["Instability Timing"].to_numpy()
y = df["Instability Mrti_Amplitude"].to_numpy()
mask = np.isfinite(x) & np.isfinite(y)
x, y = x[mask], y[mask]

# --- Define exponential model ---
def exp_model(t, A0, tau):
    return A0 * np.exp(t / tau)

# --- Fit data ---
popt, pcov = curve_fit(exp_model, x, y, p0=[0.1, 200])
A0_fit, tau_fit = popt

print(A0_fit, tau_fit)

# --- Generate fitted curve ---
x_fit = np.linspace(min(x), max(x), 200)
y_fit = exp_model(x_fit, *popt)

# --- Plot ---
plt.figure(figsize=(7,5))
plt.scatter(x, y, label="Data", color="blue")
plt.plot(x_fit, y_fit, label=f"Fit: A₀={A0_fit:.2f}, τ={tau_fit:.2f}", color="red")
plt.ylim(0,3)
plt.xlabel("Time [ns]")
plt.ylabel("Instability Amplitude")
plt.legend()
plt.tight_layout()
plt.show()
