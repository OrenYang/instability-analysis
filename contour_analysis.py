import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = "zoomed_test.png"
margin_top = 20
margin_bot = 10
threshold_fraction = 0.2
pinch_height = 13.5 #mm

def find_edges(image_path, margin_top=0, margin_bot=0, threshold_fraction=0.5, pinch_height=0, save=False, output_folder = ''):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian Blur (optional, helps with noise)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Compute min and peak intensity in the image
    min_intensity = np.min(blurred)
    peak_intensity = np.max(blurred)

    # Set threshold as a fraction between min and max intensity
    threshold_value = int(min_intensity + threshold_fraction * (peak_intensity - min_intensity))

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    im = Image.open(image_path).convert('L')
    im_array = np.array(im)
    wid, hgt = im.size

    # Compute peak intensity index along each row, constrained to center ±100 pixels
    center_x = wid // 2  # Compute center of the image
    search_range = 35  # Limit peak detection to within ±100 pixels of center

    peak_idx = []
    for i in range(hgt):
        profile = im_array[i]

        # Limit search range to [center_x - 100, center_x + 100], ensuring bounds
        left_bound = max(center_x - search_range, 0)
        right_bound = min(center_x + search_range, wid)

        # Find the local peak within this range
        local_max_index = np.argmax(profile[left_bound:right_bound]) + left_bound

        peak_idx.append(local_max_index)

    # Initialize arrays for x and y coordinates
    left_x = []
    left_y = []
    right_x = []
    right_y = []

    # Define cutoff range (ignore top 10 and bottom 10 rows)
    y_min, y_max = margin_top, hgt - margin_bot

    # Extract contour points, ignoring the top and bottom 10 rows
    for contour in contours:
        for point in contour:
            x, y = point[0]  # Extract x and y coordinates
            if y_min <= y <= y_max:  # Only process contours within valid range
                if y < len(peak_idx):  # Ensure y index is within bounds
                    if x < peak_idx[y]:
                        left_x.append(x)
                        left_y.append(y)
                    elif x > peak_idx[y]:
                        right_x.append(x)
                        right_y.append(y)

    # Draw contours on the original image
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)

    # Convert BGR to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)

    # Fit left edge to look at zippering
    left_coef = np.polyfit(left_y,left_x,1)
    left_poly1d_fn = np.poly1d(left_coef)
    left_avg = left_poly1d_fn(left_y)

    # Fit right edge to look at zippering
    right_coef = np.polyfit(right_y,right_x,1)
    right_poly1d_fn = np.poly1d(right_coef)
    right_avg = right_poly1d_fn(right_y)

    # Find mean pinch radius
    left_mean = np.mean(left_x)
    right_mean = np.mean(right_x)

    # Convert pixels to mm, given image is cropped to pinch height
    pxmm = hgt/pinch_height

    # find average radius in mm
    pinch_radius = (right_mean-left_mean)/2/pxmm
    print('Pinch Radius: {} mm'.format(pinch_radius))

    # Find instability amplitude
    left_std = np.std(left_x)
    right_std = np.std(right_x)
    instability = (left_std+right_std)/2/pxmm
    print('Instability Amplitude: {} mm'.format(instability))

    # Display using Matplotlib
    plt.figure()
    plt.imshow(image_rgb)
    plt.plot(left_x,left_y, 'bo', markersize=1)
    plt.plot(right_x,right_y, 'ro', markersize=1)
    plt.plot(left_avg, left_y, color='purple', linestyle='dashed')
    plt.plot(right_avg, right_y, color='purple', linestyle='dashed')
    plt.vlines([left_mean,right_mean],0, hgt, color='yellow', linestyle='dashed')
    plt.axis("off")  # Hide axis
    plt.title(image_path.split('.')[0].split('/')[1].replace("_", " "))

    if save:
        plt.savefig('{}{}_fit'.format(output_folder, image_path.split('.')[0].split('/')[1]))

    plt.show()

    return round(pinch_radius,2), round(instability,2), round(left_mean,2), round(right_mean,2), left_avg, right_avg
