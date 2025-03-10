import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.signal import find_peaks

def main():

    image_path = 'zoomed_test.png'
    margin = 50

    img = Image.open(image_path)
    wid, hgt = img.size

    left_points = []
    right_points = []
    for i in range(hgt):
        _, _, left, right = find_edge(image_path, i, margin)
        left_points.append(left)
        right_points.append(right)
    find_boarders(image_path, left_points, right_points, np.arange(hgt), margin)

    return

#This function finds all the edge points at a given y value in the image.
#'img' is the image path, 'y' is the y value, 'margin' is the amount of
#pixels on the left and right edges that are ignored in the analysis
def find_edge(img, y=0, margin=50, plot=False):
    im = Image.open(img).convert('L')
    im_array = np.array(im)

    profile = im_array[y]
    x = np.linspace(0, 1, len(profile))

    # Fit and smooth the profile
    coefs = np.polyfit(x, profile, 15)
    smooth_profile = np.polyval(coefs, x)

    # Compute gradient
    gradient = np.gradient(smooth_profile)
    gradient = gradient[margin:-margin]  # Ignore margins

    # Find peaks in the absolute gradient (i.e., strong intensity changes)
    peaks, properties = find_peaks(np.abs(gradient), prominence=0.01)  # Adjust prominence as needed
    peaks += margin  # Shift index back due to margin cropping

    # Identify the main peak (brightest region in the original smoothed profile)
    peak_idx = np.argmax(smooth_profile)

    left_points = [p for p in peaks if p < peak_idx]
    right_points = [p for p in peaks if p > peak_idx]

    # Plot if needed
    if plot:
        fig, ax = plt.subplots(3)
        ax[1].plot(x, smooth_profile, 'r--', label="Smoothed Profile")
        ax[1].plot(x, profile, label="Original Profile")
        ax[1].legend()

        for p in left_points:
            ax[1].axvline(x[p], color='green', linestyle='--')
        for p in right_points:
            ax[1].axvline(x[p], color='red', linestyle='--')

        ax[0].imshow(im)
        ax[0].axhline(y, color='red', linestyle='--', linewidth=1)

        ax[2].plot(x[margin:-margin], gradient, label="Gradient")
        ax[2].legend()

        for p in left_points:
            pixel = x[p] * im.width
            ax[0].plot(pixel, y, 'yo', markersize=5)
        for p in right_points:
            pixel = x[p] * im.width
            ax[0].plot(pixel, y, 'yo', markersize=5)

        for a in [ax[0], ax[1]]:
            a.set_xticks([])
            a.set_yticks([])

        plt.tight_layout()
        plt.show()

    return profile, smooth_profile, left_points, right_points

def find_boarders(img, left_points, right_points, y=0, margin=50):
    im = Image.open(img).convert('L')

    im_array = np.array(im)

    profile = im_array[y]
    x = np.linspace(0,1,len(profile))

    left_boundary = []
    right_boundary = []
    left_y = []
    right_y = []

    for i in range(len(y)):
        for j in left_points[i]:
            left_y.append(y[i])
            left_boundary.append(j)
        for j in right_points[i]:
            right_y.append(y[i])
            right_boundary.append(j)

    plt.imshow(im)
    plt.plot(left_boundary, left_y, 'go', markersize=2)
    plt.plot(right_boundary, right_y, 'ro', markersize=2)

    plt.show()
    return

if __name__ == '__main__':
    main()
