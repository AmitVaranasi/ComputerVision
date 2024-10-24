import cv2
import numpy as np
import matplotlib.pyplot as plt


def convolution(im, kernel):
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    im_height, im_width = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size - 1) / 2)
    im_padded = np.zeros((im_height + pad_size * 2, im_width + pad_size * 2))
    im_padded[pad_size:-pad_size, pad_size:-pad_size] = im

    im_out = np.zeros_like(im)
    for x in range(im_width):
        for y in range(im_height):
            im_patch = im_padded[y:y + kernel_size, x:x + kernel_size]
            new_value = np.sum(kernel * im_patch)
            im_out[y, x] = new_value
    return im_out


def get_gaussian_kernel(kernel_size, sigma):
    kernel_x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    for i in range(kernel_size):
        kernel_x[i] = np.exp(-(kernel_x[i] / sigma) ** 2 / 2)
    kernel = np.outer(kernel_x.T, kernel_x.T)

    kernel *= 1.0 / kernel.sum()
    return kernel


def compute_gradient(im):
    sobel_filter_x = np.array([[1, 0, -1], [5, 0, -5], [1, 0, -1]])
    sobel_filter_y = np.array([[1, 5, 1], [0, 0, 0], [-1, -5, -1]])
    gradient_x = convolution(im, sobel_filter_x)
    gradient_y = convolution(im, sobel_filter_y)

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    magnitude *= 255.0 / magnitude.max()
    direction = np.arctan2(gradient_y, gradient_x)
    direction *= 180 / np.pi
    return magnitude, direction


def nms(magnitude, direction):
    height, width = magnitude.shape
    res = np.zeros(magnitude.shape)
    direction[direction < 0] += 180  # (-180, 180) -> (0, 180)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_direction = direction[y, x]
            current_magnitude = magnitude[y, x]
            if (0 <= current_direction < 22.5) or (157.5 <= current_direction <= 180):
                p = magnitude[y, x - 1]
                r = magnitude[y, x + 1]

            elif 22.5 <= current_direction < 67.5:
                p = magnitude[y + 1, x + 1]
                r = magnitude[y - 1, x - 1]

            elif 67.5 <= current_direction < 112.5:
                p = magnitude[y - 1, x]
                r = magnitude[y + 1, x]

            else:
                p = magnitude[y - 1, x + 1]
                r = magnitude[y + 1, x - 1]

            if current_magnitude >= p and current_magnitude >= r:
                res[y, x] = current_magnitude

    return res

def apply_hysteresis_thresholding(nms_output, low_threshold, high_threshold):
    strong_edges_map = (nms_output >= high_threshold)
    weak_edges_map = ((nms_output >= low_threshold) & (nms_output < high_threshold))
    noise_map = (nms_output < low_threshold)

    return noise_map, weak_edges_map, strong_edges_map

def edge_linking(strong_edges_map, weak_edges_map):
    final_edges_map = strong_edges_map.copy()

    # Get image dimensions
    h, w = weak_edges_map.shape

    # Check all weak edge pixels
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Process only weak edges
            if weak_edges_map[y, x] == 1:
                # Check surrounding 8 pixels for a connection to a strong edge
                if np.any(strong_edges_map[y - 1:y + 2, x - 1:x + 2] == 1):
                    final_edges_map[y, x] = 1  # Link to strong edge
                else:
                    final_edges_map[y, x] = 0  # Remove as noise

    return final_edges_map


im = cv2.imread("./lena.png",0)
im = im.astype(float)

gaussian_kernel = get_gaussian_kernel(9, 2**(1/2))
im_smoothed = convolution(im, gaussian_kernel)

gradient_magnitude, gradient_direction = compute_gradient(im_smoothed)

edge_nms = nms(gradient_magnitude, gradient_direction)

threshold_sets = [(10, 15), (15, 25), (30, 50)]
results = []
linked_edges_results = []

for low, high in threshold_sets:
    noise_map, weak_edges_map, strong_edges_map = apply_hysteresis_thresholding(edge_nms, low, high)
    results.append((noise_map, weak_edges_map, strong_edges_map))
    linked_edges_map = edge_linking(strong_edges_map, weak_edges_map)
    linked_edges_results.append(linked_edges_map)

    # Visualizing the noise, weak edges, and strong edges for each threshold set
fig, axes = plt.subplots(len(threshold_sets), 3, figsize=(12, 12))
for i, (noise_map, weak_edges_map, strong_edges_map) in enumerate(results):
    axes[i, 0].imshow(noise_map, cmap='gray')
    axes[i, 0].set_title(f'Set {i+1} - Noise Map (Low: {threshold_sets[i][0]}, High: {threshold_sets[i][1]})')
        
    axes[i, 1].imshow(weak_edges_map, cmap='gray')
    axes[i, 1].set_title(f'Set {i+1} - Weak Edges')
        
    axes[i, 2].imshow(strong_edges_map, cmap='gray')
    axes[i, 2].set_title(f'Set {i+1} - Strong Edges')
        
    for ax in axes[i]:
        ax.axis('off')

fig, axes = plt.subplots(1, len(threshold_sets), figsize=(15, 5))
for i, linked_edges_map in enumerate(linked_edges_results):
    axes[i].imshow(linked_edges_map, cmap='gray')
    axes[i].set_title(f'Linked Edges (Low: {threshold_sets[i][0]}, High: {threshold_sets[i][1]})')
    axes[i].axis('off')

plt.tight_layout()
plt.show()




#--------------------------------------------------------------------------------------------------------------------------------------

# Code for the surpressing local maxima in hough transform

#---------------------------------------------------------------------------------------------------------------------------------------



def non_maximum_suppression(accumulator, threshold=50):
    suppressed_accumulator = np.zeros_like(accumulator)
    for i in range(1, accumulator.shape[0] - 1):
        for j in range(1, accumulator.shape[1] - 1):
            if accumulator[i, j] >= threshold and accumulator[i, j] > accumulator[i - 1, j] and \
               accumulator[i, j] > accumulator[i + 1, j] and accumulator[i, j] > accumulator[i, j - 1] and \
               accumulator[i, j] > accumulator[i, j + 1] and accumulator[i, j] > accumulator[i-1, j - 1] and \
               accumulator[i, j] > accumulator[i+1, j - 1] and accumulator[i, j] > accumulator[i-1, j + 1] and \
               accumulator[i, j] > accumulator[i+1, j + 1]:
                suppressed_accumulator[i, j] = accumulator[i, j]
    return suppressed_accumulator

def HoughTransform(edge_map):
    theta_values = np.deg2rad(np.arange(-90.0, 90.0))
    height, width = edge_map.shape
    diagonal_length = int(round(math.sqrt(width * width + height * height)))
    rho_values = np.linspace(-diagonal_length, diagonal_length, diagonal_length * 2 + 1)

    accumulator = np.zeros((len(rho_values), len(theta_values)), dtype=int)
    y_coordinates, x_coordinates = np.nonzero(edge_map)

    for edge_idx in range(len(x_coordinates)):
        x = x_coordinates[edge_idx]
        y = y_coordinates[edge_idx]
        for theta_idx in range(len(theta_values)):
            theta = theta_values[theta_idx]
            rho = int(round(x * np.cos(theta) + y * np.sin(theta)))
            accumulator[rho + diagonal_length, theta_idx] += 1

    return accumulator, theta_values, rho_values


im = cv2.imread('./paper.bmp')
im2 = cv2.imread('./shape.bmp')

im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

edge_map = cv2.Canny(im_gray, 70, 150)
edge_map_im2 = cv2.Canny(im2_gray, 70, 150)

accumulator, theta_values, rho_values = HoughTransform(edge_map)
paper_suppressed = non_maximum_suppression(accumulator)

accumulator2, theta_values2, rho_values2 = HoughTransform(edge_map_im2)
shape_suppressed = non_maximum_suppression(accumulator2)

lines = np.argwhere(paper_suppressed > 5)
lines2 = np.argwhere(shape_suppressed > 50)

height, width = im_gray.shape
for line in lines:
    rho = rho_values[line[0]]
    theta = theta_values[line[1]]
    slope = -np.cos(theta)/np.sin(theta)
    intercept = rho/np.sin(theta)
    x1, x2 = 0, width
    y1 = int(slope*x1 + intercept)
    y2 = int(slope*x2 + intercept)
    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 2)


height_im2, width_im2 = im2_gray.shape
for line in lines2:
    rho = rho_values2[line[0]]
    theta = theta_values2[line[1]]
    slope = -np.cos(theta)/np.sin(theta)
    intercept = rho/np.sin(theta)
    x1, x2 = 0, width_im2
    y1 = int(slope*x1 + intercept)
    y2 = int(slope*x2 + intercept)
    cv2.line(im2, (x1, y1), (x2, y2), (0, 0, 255), 2)


cv2.imshow("Output", im)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imshow("Output", im2)
cv2.waitKey(10000)
cv2.destroyAllWindows()



#--------------------------------------------------------------------------------------------------------------------------------------

# Canny edge detector and Hough transform implemented in OpenCV

#---------------------------------------------------------------------------------------------------------------------------------------



lena_path = './lena.png'
lena_image = cv2.imread(lena_path, cv2.IMREAD_GRAYSCALE)
paper_path = './paper.bmp'
paper_image = cv2.imread(paper_path, cv2.IMREAD_GRAYSCALE)
shape_path = './shape.bmp'
shape_image = cv2.imread(shape_path, cv2.IMREAD_GRAYSCALE)

lena_edges = cv2.Canny(lena_image, 100, 200)
paper_edges = cv2.Canny(paper_image, 100, 200)
shape_edges = cv2.Canny(shape_image, 100, 200)


paper_lines = cv2.HoughLines(paper_edges, 1, np.pi / 180, threshold=50)
shape_lines = cv2.HoughLines(shape_edges, 1, np.pi / 180, threshold=50)


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


paper_with_lines = draw_hough_lines(paper_edges, paper_lines)
shape_with_lines = draw_hough_lines(shape_edges, shape_lines)

plt.figure(figsize=(18, 6))


plt.subplot(1, 3, 1)
plt.imshow(lena_edges, cmap='gray')
plt.title("Lena Image - Canny Edge Detection")


plt.subplot(1, 3, 2)
plt.imshow(paper_with_lines)
plt.title("Paper Image - Hough Lines")


plt.subplot(1, 3, 3)
plt.imshow(shape_with_lines)
plt.title("Shape Image - Hough Lines")

plt.show()