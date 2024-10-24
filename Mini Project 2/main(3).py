import cv2
import numpy as np
import math



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
    cv2.line(im, (x1, y1), (x2, y2), (0, 0, 255), 1)


height_im2, width_im2 = im2_gray.shape
for line in lines2:
    rho = rho_values2[line[0]]
    theta = theta_values2[line[1]]
    slope = -np.cos(theta)/np.sin(theta)
    intercept = rho/np.sin(theta)
    x1, x2 = 0, width_im2
    y1 = int(slope*x1 + intercept)
    y2 = int(slope*x2 + intercept)
    cv2.line(im2, (x1, y1), (x2, y2), (0, 0, 255), 1)


cv2.imshow("Edges", edge_map)
cv2.imshow("Hough Transform", (paper_suppressed*255/paper_suppressed.max()).astype(np.uint8))
cv2.imshow("Output", im)
cv2.waitKey(10000)
cv2.destroyAllWindows()

cv2.imshow("Edges im2", edge_map_im2)
cv2.imshow("Hough Transform", (shape_suppressed*255/shape_suppressed.max()).astype(np.uint8))
cv2.imshow("Output", im2)
cv2.waitKey(10000)
cv2.destroyAllWindows()