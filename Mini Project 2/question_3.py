import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Lena image and apply Canny edge detection
lena_path = './lena.png'
lena_image = cv2.imread(lena_path, cv2.IMREAD_GRAYSCALE)
paper_path = './paper.bmp'
paper_image = cv2.imread(paper_path, cv2.IMREAD_GRAYSCALE)
shape_path = './shape.bmp'
shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)
# Applying Canny edge detection with chosen parameters
lena_edges = cv2.Canny(lena_image, 100, 200)
paper_edges = cv2.Canny(paper_image, 100, 200)
shape_edges = cv2.Canny(shape_image, 100, 200)
# Apply HoughLines on the paper and shape images
# Use standard parameters for line detection

# Using the pre-processed edge images from previous steps for consistency
paper_lines = cv2.HoughLines(paper_edges, 1, np.pi / 180, threshold=50)
shape_lines = cv2.HoughLines(shape_edges, 1, np.pi / 180, threshold=50)

# Draw lines on a copy of the original images
def draw_hough_lines(image, lines):
    image_with_lines = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return image_with_lines

# Draw lines on the original images
paper_with_lines = draw_hough_lines(paper_edges, paper_lines)
shape_with_lines = draw_hough_lines(shape_edges, shape_lines)

# Plotting results
plt.figure(figsize=(18, 6))

# Lena Edge Detection
plt.subplot(1, 3, 1)
plt.imshow(lena_edges, cmap='gray')
plt.title("Lena Image - Canny Edge Detection")

# Paper with Hough Lines
plt.subplot(1, 3, 2)
plt.imshow(paper_with_lines)
plt.title("Paper Image - Hough Lines")

# Shape with Hough Lines
plt.subplot(1, 3, 3)
plt.imshow(shape_with_lines)
plt.title("Shape Image - Hough Lines")

plt.show()