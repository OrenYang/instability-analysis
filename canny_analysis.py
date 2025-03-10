import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_canny_edges(image_path):
    # Step 1: Load the image
    image = cv2.imread(image_path)

    # Step 2: Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply GaussianBlur to reduce noise and improve edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Step 4: Perform edge detection using Canny
    edges = cv2.Canny(blurred_image, 10, 30)

    # Step 5: Display the detected edges
    plt.figure(figsize=(8, 8))
    plt.imshow(edges, cmap='gray')  # Display in grayscale
    plt.title("Canny Edge Detection")
    plt.axis('off')
    plt.show()

# Example usage
image_path = 'test.png'  # Replace with your image file path
#plot_canny_edges(image_path)

def detect_boundaries(image_path):
    # Step 1: Load Image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 3: Apply Bilateral Filtering to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 4: Apply sharpening to enhance edge definition
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(filtered, -1, sharpening_kernel)

    # Step 5: Adaptive Canny Edge Detection
    median_intensity = np.median(sharpened)
    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))
    edges = cv2.Canny(sharpened, 20, 62)

    # Step 6: Detect left and right boundaries
    height, width = edges.shape
    left_boundary = []
    right_boundary = []

    for y in range(height):
        row = np.where(edges[y, :] > 0)[0]  # Get edge pixel locations in the row

        if len(row) > 0:
            left_boundary.append((row[0], y))   # Leftmost point in this row
            right_boundary.append((row[-1], y)) # Rightmost point in this row

    # Step 7: Draw boundaries on original image
    image_with_boundaries = image.copy()
    for x, y in left_boundary:
        cv2.circle(image_with_boundaries, (x, y), 1, (0, 255, 0), -1)  # Green for left boundary
    for x, y in right_boundary:
        cv2.circle(image_with_boundaries, (x, y), 1, (0, 0, 255), -1)  # Red for right boundary

    # Step 8: Display Results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(edges, cmap='gray')
    ax[0].set_title("Canny Edge Detection")
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(image_with_boundaries, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Detected Boundaries")
    ax[1].axis('off')

    plt.show()

    return left_boundary, right_boundary

# Example usage
image_path = 'test.png'  # Replace with your image path
#left_boundary, right_boundary = detect_boundaries(image_path)


def enhanced_canny(image_path):
    # Load image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Step 2: Apply Bilateral Filtering to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 3: Apply sharpening to enhance edge definition
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(filtered, -1, sharpening_kernel)

    # Step 4: Adaptive Canny Edge Detection
    median_intensity = np.median(sharpened)
    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))
    edges = cv2.Canny(sharpened, 1, 60)

    # Display results
    plt.figure(figsize=(8, 8))
    plt.imshow(edges, cmap='gray')
    plt.title("Enhanced Canny Edge Detection")
    plt.axis('off')
    plt.show()

# Example usage
image_path = 'test.png'  # Replace with your image file path
#enhanced_canny(image_path)



def detect_left_right_boundaries(image_path):
    # Step 1: Load Image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 3: Reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 4: Sharpen the image
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(filtered, -1, sharpening_kernel)

    # Step 5: Apply Adaptive Canny Edge Detection
    median_intensity = np.median(sharpened)
    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))
    edges = cv2.Canny(sharpened, 10, 80)

    # Step 6: Extract left and right boundary points
    height, width = edges.shape
    left_boundary = []
    right_boundary = []

    for y in range(height):
        row = np.where(edges[y, :] > 0)[0]  # Get edge positions in row
        if len(row) > 0:
            left_boundary.append((row[0], y))  # First detected edge in row (left)
            right_boundary.append((row[-1], y))  # Last detected edge in row (right)

            # If there are multiple left/right candidates, we keep additional points
            if len(row) > 2:
                left_boundary.append((row[1], y))  # Second leftmost (if exists)
                right_boundary.append((row[-2], y))  # Second rightmost (if exists)

    # Step 7: Draw boundaries on the original image
    image_with_boundaries = image.copy()
    for x, y in left_boundary:
        cv2.circle(image_with_boundaries, (x, y), 1, (0, 255, 0), -1)  # Green for left boundary
    for x, y in right_boundary:
        cv2.circle(image_with_boundaries, (x, y), 1, (0, 0, 255), -1)  # Red for right boundary

    # Step 8: Display Results
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(edges, cmap='gray')
    ax[0].set_title("Canny Edge Detection")
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(image_with_boundaries, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Detected Left & Right Boundaries")
    ax[1].axis('off')

    plt.show()

    return left_boundary, right_boundary

# Example usage
image_path = 'test3.tif'  # Replace with your image path
left_boundary, right_boundary = detect_left_right_boundaries(image_path)


from scipy.interpolate import UnivariateSpline

def detect_and_trace_boundaries(image_path):
    # Step 1: Load Image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 3: Reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 4: Sharpen the image
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(filtered, -1, sharpening_kernel)

    # Step 5: Apply Adaptive Canny Edge Detection
    median_intensity = np.median(sharpened)
    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity))
    edges = cv2.Canny(sharpened, 20, 62)

    # Step 6: Extract left and right boundary points
    height, width = edges.shape
    left_boundary = []
    right_boundary = []

    for y in range(height):
        row = np.where(edges[y, :] > 0)[0]  # Get edge positions in row
        if len(row) > 0:
            left_boundary.append((row[0], y))  # First detected edge in row (left)
            right_boundary.append((row[-1], y))  # Last detected edge in row (right))

    # Convert to NumPy arrays for curve fitting
    left_boundary = np.array(left_boundary)
    right_boundary = np.array(right_boundary)

    # Step 7: Fit a smooth curve (Spline) to the boundaries
    if len(left_boundary) > 3 and len(right_boundary) > 3:
        left_spline = UnivariateSpline(left_boundary[:, 1], left_boundary[:, 0], s=50)
        right_spline = UnivariateSpline(right_boundary[:, 1], right_boundary[:, 0], s=50)

        y_values = np.arange(height)
        left_x_fitted = left_spline(y_values)
        right_x_fitted = right_spline(y_values)

    # Step 8: Draw boundaries on a new figure
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.plot(left_x_fitted, y_values, 'g-', linewidth=2, label="Left Boundary")  # Green Line
    ax.plot(right_x_fitted, y_values, 'r-', linewidth=2, label="Right Boundary")  # Red Line
    ax.set_title("Traced Left and Right Boundaries")
    ax.axis('off')
    ax.legend()

    plt.show()

    return left_x_fitted, right_x_fitted

# Example usage
image_path = 'test.png'  # Replace with your image path
#left_fitted, right_fitted = detect_and_trace_boundaries(image_path)
