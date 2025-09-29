import os
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import csv

# Fixed real-world image height in mm
height_mm = 13.5  # <-- change this to your setup

def process_pair(npz_path, image_path, height_mm):
    # Load NPZ data
    data = np.load(npz_path)

    if not all(k in data for k in ["left_x", "left_y", "right_x", "right_y"]):
        raise KeyError(f"Missing keys in {npz_path}")

    left_x, left_y = data["left_x"], data["left_y"]
    right_x, right_y = data["right_x"], data["right_y"]

    # Compute mean x at each y for left
    unique_left_y = np.unique(left_y)
    left_means = [np.mean(left_x[left_y == y]) for y in unique_left_y]
    mean_left_x = np.mean(left_means)

    # Compute mean x at each y for right
    unique_right_y = np.unique(right_y)
    right_means = [np.mean(right_x[right_y == y]) for y in unique_right_y]
    mean_right_x = np.mean(right_means)

    # Radius in px
    avg_radius_px = (mean_right_x - mean_left_x) / 2

    # Linear fit: x = a*y + b
    left_fit = np.polyfit(unique_left_y, left_means, deg=1)
    right_fit = np.polyfit(unique_right_y, right_means, deg=1)

    # Predicted x values from the fit
    left_fit_x = np.polyval(left_fit, unique_left_y)
    right_fit_x = np.polyval(right_fit, unique_right_y)

    # Residuals
    left_residuals = left_means - left_fit_x
    right_residuals = right_means - right_fit_x

    # Instability amplitude = std of residuals
    left_instability = np.std(left_residuals)
    right_instability = np.std(right_residuals)
    instability_amplitude = (left_instability + right_instability) / 2
    
    # Image height in px
    with Image.open(image_path) as img:
        height_px = img.height

    # Conversion factor
    px_to_mm = height_mm / height_px

    # Convert radius to mm
    avg_radius_mm = avg_radius_px * px_to_mm

    instability_amplitude = instability_amplitude * px_to_mm

    return avg_radius_px, avg_radius_mm, height_px, px_to_mm, mean_left_x, mean_right_x, left_instability, right_instability, instability_amplitude


if __name__ == "__main__":
    # Tkinter root (hidden)
    root = tk.Tk()
    root.withdraw()

    # Ask for NPZ folder
    npz_folder = filedialog.askdirectory(title="Select NPZ folder")
    if not npz_folder:
        raise ValueError("No NPZ folder selected")

    # Ask for image folder
    image_folder = filedialog.askdirectory(title="Select image folder")
    if not image_folder:
        raise ValueError("No image folder selected")

    results = []
    valid_exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    for file in os.listdir(npz_folder):
        if file.endswith("_boundary_points.npz"):
            npz_path = os.path.join(npz_folder, file)

            # Strip "_boundary_points" so base name matches image files
            base_name = file.replace("_boundary_points.npz", "")

            # Find matching image with any valid extension
            image_path = None
            for ext in valid_exts:
                candidate = os.path.join(image_folder, base_name + ext)
                if os.path.exists(candidate):
                    image_path = candidate
                    break

            if not image_path:
                print(f"⚠️ No matching image for {file}, skipping")
                continue

            try:
                avg_px, avg_mm, height_px, scale, mean_left, mean_right, left_instabiliy, right_instability, instability_amplitude = process_pair(
                    npz_path, image_path, height_mm
                )
                results.append((base_name, avg_px, avg_mm, instability_amplitude))
                print(f"{base_name}: {avg_px:.2f} px = {avg_mm:.2f} mm")
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

    # Save results to CSV one folder up from NPZ folder
    parent_folder = os.path.dirname(npz_folder)
    csv_path = os.path.join(parent_folder, "radius_results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Shot", "Radius_px", "Radius_mm","instability amplitude"])
        writer.writerows(results)

    print(f"\n✅ Results saved to {csv_path}")
