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
from scipy.signal import butter, filtfilt

use_log = True
use_lowPass_filter = False
use_waveNumber = False
min_percent = 0.0001 # min_val = min_percent*max_val, minimum of color bar

def lowpass_filter(y, cutoff_freq, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, y)

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
    axis_global = []
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

        # Choose wavelength or wavenumber
        if use_waveNumber:
            axis_vals = 1.0 / wl
        else:
            axis_vals = wl

        if use_lowPass_filter:
            fs = 1.0 / (axis_vals[1] - axis_vals[0])   # sampling rate
            cutoff_freq = 0.1 * fs
            psd = lowpass_filter(psd, cutoff_freq, fs)

        # store global range
        axis_global.append((axis_vals.min(), axis_vals.max()))
        psd_matrix.append((axis_vals, psd))
        times.append(time)

    # define global grid
    axis_min = min(mn for mn, mx in axis_global)
    axis_max = max(mx for mn, mx in axis_global)
    common_axis = np.linspace(axis_min, axis_max, 1000)

    # interpolate all PSDs onto common grid
    psd_interp_matrix = []
    for axis_vals, psd in psd_matrix:
        f = interp1d(axis_vals, psd, bounds_error=False, fill_value=np.nan)
        psd_interp_matrix.append(f(common_axis))

    psd_interp_matrix = np.array(psd_interp_matrix).T  # (axis, times)
    times = np.array(times)

    # sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    psd_interp_matrix = psd_interp_matrix[:, sort_idx]

    # Interpolate along the time axis
    time_uniform = np.linspace(times.min(), times.max(), 500)
    psd_time_interp = np.empty((len(common_axis), len(time_uniform)))

    for i in range(len(common_axis)):
        f = interp1d(times, psd_interp_matrix[i, :], bounds_error=False, fill_value=np.nan)
        psd_time_interp[i, :] = f(time_uniform)

    # Plot
    plt.figure(figsize=(8, 6))
    X, Y = np.meshgrid(time_uniform, common_axis)

    if use_log:
        contour = plt.pcolormesh(
            X, Y, psd_time_interp,
            shading='auto',
            cmap='jet',
            norm=mcolors.LogNorm(
                vmin=min_percent*np.nanmax(psd_time_interp),#np.nanmin(psd_time_interp[psd_time_interp > 0]),
                vmax=np.nanmax(psd_time_interp)
            )
        )
    else:
        contour = plt.pcolormesh(
            X, Y, psd_time_interp,
            shading='auto',
            cmap='jet'
        )

    plt.colorbar(contour, label="PSD")
    plt.xlabel("Time (ns)")
    ylabel = "Wavenumber (1/mm)" if use_waveNumber else "Wavelength (mm)"
    plt.ylabel(ylabel)
    plt.title("FFT Contour Plot")

    # Add vertical red dashed lines at original time points
    for t in times:
        plt.axvline(x=t, color='red', linestyle='--', linewidth=0.8)

    plt.show()
