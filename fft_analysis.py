import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from collections import defaultdict
from PIL import Image
import os
from scipy.signal import detrend

def single_valued_profile(xs, ys, agg=np.median):
    y_to_x = defaultdict(list)
    for x, y in zip(xs, ys):
        y_to_x[y].append(x)
    ys_sorted = sorted(y_to_x.keys())
    xs_agg = [agg(y_to_x[y]) for y in ys_sorted]
    return np.array(ys_sorted), np.array(xs_agg)

def fft_analysis(npz_file,img_file, save_folder="ffts"):
    image = Image.open(img_file)
    img_height_px = image.size[1]
    pxmm = img_height_px / 13.5

    ########### Load boundary data #####################
    data = np.load(npz_file)

    left_y_single, left_x_single = single_valued_profile(data['left_x'], data['left_y'], agg=np.min)
    right_y_single, right_x_single = single_valued_profile(data['right_x'], data['right_y'], agg=np.max)
    z = np.intersect1d(left_y_single, right_y_single)
    left_dict = dict(zip(left_y_single, left_x_single))
    right_dict = dict(zip(right_y_single, right_x_single))
    half_width_px = np.array([(right_dict[i] - left_dict[i]) / 2 for i in z])

    z_mm = z / pxmm
    half_width_mm = half_width_px / pxmm
    half_width_mm = half_width_mm - np.mean(half_width_mm)
    ##################################################


    ######## HALF-WIDTH FFT #######################
    N = len(half_width_mm)
    N_pad = 2 * N

    dz = np.mean(np.diff(z_mm))
    L = N*dz
    fft_vals = rfft(half_width_mm, n=N_pad)
    fft_freqs = rfftfreq(N_pad, d=dz)

    psd = (2.0 * dz / N) * np.abs(fft_vals)**2
    psd[0] /= 2

    fft_wavelengths = 1/fft_freqs[1:]
    fft_psd = psd[1:]
    #################################################


    ####### HALF-WIDTH DETRENDED FFT ###############
    half_width_mm_detrended = detrend(half_width_mm, type='linear')

    fft_vals_detrended = rfft(half_width_mm_detrended, n=N_pad)

    fft_freqs_detrended = fft_freqs
    psd_detrended = (2.0 * dz / N) * np.abs(fft_vals_detrended)**2
    psd_detrended[0] /= 2

    fft_wavelengths_detrended = 1/fft_freqs[1:]
    fft_psd_detrended = psd_detrended[1:]
    #################################################


    ##### LEFT EDGE FFT and DETRENDED LEFT EDGE FFT ######
    z_left_mm = left_y_single / pxmm
    dz_left = np.mean(np.diff(z_left_mm))
    N_left = len(left_x_single)
    N_left_pad = 2 * N_left
    L_left = N_left * dz_left
    fft_freqs_left = rfftfreq(N_left_pad, d=dz_left)

    left_x_mm = (left_x_single - np.mean(left_x_single))/pxmm
    fft_vals_left = rfft(left_x_mm, n=N_left_pad)

    psd_left = (2 * dz_left / N_left) * np.abs(fft_vals_left)**2
    fft_psd_left = psd_left[1:]
    fft_wavelengths_left = 1 / fft_freqs_left[1:]

    left_mm_detrended = detrend(left_x_mm, type='linear')
    fft_vals_left_detr = rfft(left_mm_detrended, n=N_left_pad)
    psd_left_detr = (2 * dz_left / N_left) * np.abs(fft_vals_left_detr)**2
    fft_psd_left_detr = psd_left_detr[1:]
    #################################################

    ##### RIGHT EDGE FFT and DETRENDED RIGHT EDGE FFT ######
    z_right_mm = right_y_single / pxmm
    dz_right = np.mean(np.diff(z_right_mm))
    N_right = len(right_x_single)
    N_right_pad = 2 * N_right
    L_right = N_right * dz_right
    fft_freqs_right = rfftfreq(N_right_pad, d=dz_right)

    right_x_mm = (right_x_single - np.mean(right_x_single))/pxmm
    fft_vals_right = rfft(right_x_mm, n=N_right_pad)

    psd_right = (2 * dz_right / N_right) * np.abs(fft_vals_right)**2
    fft_psd_right = psd_right[1:]
    fft_wavelengths_right = 1 / fft_freqs_right[1:]

    right_mm_detrended = detrend(right_x_mm, type='linear')
    fft_vals_right_detr = rfft(right_mm_detrended, n=N_right_pad)
    psd_right_detr = (2 * dz_right / N_right) * np.abs(fft_vals_right_detr)**2
    fft_psd_right_detr = psd_right_detr[1:]
    #######################################################

    ########### Save FFT results to NPZ #####################
    os.makedirs(save_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_file))[0]
    npz_path = os.path.join(save_folder, f"{base_name}_fft.npz")

    np.savez(npz_path,
             fft_wavelengths=fft_wavelengths,
             fft_psd=fft_psd,
             fft_wavelengths_detrended=fft_wavelengths_detrended,
             fft_psd_detrended=fft_psd_detrended,
             fft_wavelengths_left=fft_wavelengths_left,
             fft_psd_left=fft_psd_left,
             fft_psd_left_detr=fft_psd_left_detr,
             fft_wavelengths_right=fft_wavelengths_right,
             fft_psd_right=fft_psd_right,
             fft_psd_right_detr=fft_psd_right_detr)

    print(f"Saved FFT results to {npz_path}")
    ##########################################################

    '''plt.plot(fft_wavelengths, fft_psd)
    plt.plot(fft_wavelengths_left, fft_psd_left_detr)
    plt.plot(fft_wavelengths_right, fft_psd_right_detr)
    plt.show()

    plt.plot(half_width_mm, z_mm)
    plt.show()'''
    return

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # --- Select folder with NPZ files ---
    npz_folder = filedialog.askdirectory(title="Select folder with boundary_points NPZ files")
    # --- Select folder with images ---
    img_folder = filedialog.askdirectory(title="Select folder with images")

    save_root = os.path.abspath(os.path.join(img_folder, ".."))
    save_folder = os.path.join(save_root, "ffts")
    os.makedirs(save_folder, exist_ok=True)

    # --- Find all NPZ files ending with '_boundary_points.npz' ---
    npz_files = [f for f in os.listdir(npz_folder) if f.endswith("_boundary_points.npz")]

    for npz_file in npz_files:
        base_name = npz_file.replace("_boundary_points.npz", "")
        # Find matching image
        possible_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        img_file = None
        for ext in possible_exts:
            candidate = os.path.join(img_folder, base_name + ext)
            if os.path.exists(candidate):
                img_file = candidate
                break
        if img_file is None:
            print(f"No image found for {npz_file}, skipping...")
            continue

        npz_path = os.path.join(npz_folder, npz_file)
        fft_analysis(npz_path, img_file, save_folder=save_folder)
