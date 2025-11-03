import numpy as np
import matplotlib.pyplot as plt

def generate_zpinch_image(
    radius_mm,
    wavelengths_mm,
    amplitudes_mm,
    img_height_px,          # ✅ new required input
    img_height_mm,          # ✅ new required input
    output_file="zpinch.png"
):
    """
    Generate a Z-pinch image with perturbed boundaries.

    Parameters:
        radius_mm (float): Average radius of the Z-pinch in mm
        wavelengths_mm (list of float): List of perturbation wavelengths (mm)
        amplitudes_mm (list of float): List of corresponding amplitudes (mm)
        img_height_px (int): Total image height in pixels (vertical axis)
        img_height_mm (float): Physical height of the image in mm
        output_file (str): Filename to save the PNG image
    """

    # mm → px scaling (vertical)
    mm_to_px = img_height_px / img_height_mm

    # Vertical axis in mm
    y_mm = np.linspace(0, img_height_mm, img_height_px)

    # Build perturbation in mm
    perturb_mm = np.zeros_like(y_mm)
    for amp_mm, wave_mm in zip(amplitudes_mm, wavelengths_mm):
        perturb_mm += amp_mm * np.sin(2 * np.pi * y_mm / wave_mm)

    # Convert values from mm to pixels
    radius_px = radius_mm * mm_to_px
    perturb_px = perturb_mm * mm_to_px

    # Auto-determine horizontal image size
    max_radius_px = radius_px + np.max(np.abs(perturb_px))
    img_width_px = max(img_height_px,int(2 * max_radius_px + img_height_px/5) ) # add padding

    # Create blank image
    image = np.zeros((img_height_px, img_width_px), dtype=np.uint8)

    # Find center of image horizontally
    center_x = img_width_px // 2

    # Fill inside boundary
    for i in range(img_height_px):
        left = int(center_x - radius_px - perturb_px[i])
        right = int(center_x + radius_px + perturb_px[i])
        left = max(0, left)
        right = min(img_width_px, right)
        image[i, left:right] = 255

    # Save image
    plt.imsave(output_file, image, cmap='gray', vmin=0, vmax=255)
    print(f"Saved image as {output_file} with shape {image.shape}")


# Example usage
if __name__ == "__main__":
    wavelengths = [9] #mm
    amplitudes = [1]
    wavelength_str = "_".join(f"{w:.2f}" for w in wavelengths)

    generate_zpinch_image(
        radius_mm=1.5,
        wavelengths_mm=wavelengths,
        amplitudes_mm=amplitudes,
        img_height_px=1500,        # height in pixels
        img_height_mm=13.5,       # height in mm
        output_file=f"zpinch_{wavelength_str}.png"
    )
