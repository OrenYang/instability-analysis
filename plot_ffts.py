import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from scipy.interpolate import interp1d
import matplotlib.colors as mcolors


if __name__ == '__main__':

    root = tk.Tk()
    root.withdraw()

    dir = filedialog.askdirectory(title='Select FFT Folder')
    if not dir:
        raise ValueError("No folder selected")

    # Load timing CSV
    csv_path = os.path.join(dir, "fft_times.csv")
    df = pd.read_csv(csv_path)  # must have "file" and "time" columns

    # Collect FFT data
    wavelengths_global = []
    psd_matrix = []
    times = []

    for _, row in df.iterrows():
        file_name = row["file"]
        time = row["time"]

        fft_path = os.path.join(dir, file_name)
        if not os.path.exists(fft_path):
            print(f"Skipping missing file: {fft_path}")
            continue

        # Load FFT npz file
        data = np.load(fft_path)

        wl = data["fft_wavelengths_detrended"]
        psd = data["fft_psd_detrended"]

        # store global range
        wavelengths_global.append((wl.min(), wl.max()))

        # keep raw until we define global grid
        psd_matrix.append((wl, psd))
        times.append(time)

    # define global grid (min of mins to max of maxes)
    wl_min = min(mn for mn, mx in wavelengths_global)
    wl_max = max(mx for mn, mx in wavelengths_global)
    common_wavelengths = np.linspace(wl_min, wl_max, 1000)  # choose resolution

    # interpolate all PSDs onto common wavelength grid
    psd_interp_matrix = []
    for wl, psd in psd_matrix:
        f = interp1d(wl, psd, bounds_error=False, fill_value=np.nan)
        psd_interp_matrix.append(f(common_wavelengths))

    psd_interp_matrix = np.array(psd_interp_matrix).T  # shape (wavelengths, times)
    times = np.array(times)

    # sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    psd_interp_matrix = psd_interp_matrix[:, sort_idx]

    # -----------------------------
    # Interpolate along the time axis
    # -----------------------------
    time_uniform = np.linspace(times.min(), times.max(), 500)  # uniform time grid
    psd_time_interp = np.empty((len(common_wavelengths), len(time_uniform)))

    for i in range(len(common_wavelengths)):
        f = interp1d(times, psd_interp_matrix[i, :], bounds_error=False, fill_value=np.nan)
        psd_time_interp[i, :] = f(time_uniform)

    # -----------------------------
    # Plot the fully interpolated PSD
    # -----------------------------
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(time_uniform, common_wavelengths)

    contour = plt.pcolormesh(
        X, Y, psd_time_interp,
        shading='auto',
        cmap='viridis',
        #norm=mcolors.LogNorm(
        #    vmin=np.nanmin(psd_time_interp[psd_time_interp>0]),
        #    vmax=np.nanmax(psd_time_interp)
        #)
    )

    plt.colorbar(contour, label="PSD (detrended, log scale)")
    plt.xlabel("Time (ns)")
    plt.ylabel("Wavelength (nm)")
    plt.title("FFT Contour Plot (Detrended, Interpolated in Wavelength and Time)")

    # Add vertical red dashed lines at original time points
    for t in times:
        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.8)

    plt.show()
