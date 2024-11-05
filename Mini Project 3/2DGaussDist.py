import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# Define the multivariate Gaussian PDF function
def multivariate_gaussian_pdf(x, mean, cov, det_sigma, inv_sigma):
    diff = x - mean
    exponent = -0.5 * np.dot(np.dot(diff.T, inv_sigma), diff)
    norm_factor = 1.0 / (2 * np.pi * np.sqrt(det_sigma))
    return norm_factor * np.exp(exponent)

def calculate_hs_histogram(img):
    height, width, _ = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hs = img_hsv[:, :, :2]
    mean = np.mean(img_hs, axis=(0, 1))

    cov = np.zeros((2, 2))
    for i in range(height):
        for j in range(width):
            h = img_hsv[i, j, 0]
            s = img_hsv[i, j, 1]
            hs = np.array((h, s))
            diff = hs - mean
            diff_reshape = diff.reshape(-1, 1)
            cov += np.dot(diff_reshape, diff_reshape.T)
    cov = cov / (height * width - 1)
    return cov, mean

# Training
sum_cov = np.zeros((2, 2))
sum_mean = np.zeros((2,))
for i in range(1, 13):
    print(i)
    img_train = cv2.imread(f'/Users/amitmaheshwarvaranasi/Desktop/{i}.png')
    cov, mean = calculate_hs_histogram(img_train)
    sum_cov += cov
    sum_mean += mean
sum_cov = sum_cov / 12
sum_mean = sum_mean / 12
sum_mean = sum_mean.reshape(-1, 1)  # Reshape mean to (2, 1)

det_sigma = np.linalg.det(sum_cov)
inv_sigma = np.linalg.inv(sum_cov)

# Testing
test_image = cv2.imread('testing_image.bmp')
hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
hue, saturation = hsv_image[:, :, 0], hsv_image[:, :, 1]
pixels = np.column_stack((hue.flatten(), saturation.flatten()))

probabilities = np.array([
    multivariate_gaussian_pdf(pixel, sum_mean.flatten(), sum_cov, det_sigma, inv_sigma)
    for pixel in pixels
])
probability_map = probabilities.reshape(hue.shape)

# Thresholding and segmentation
threshold = 0.0001  # Adjust the threshold as needed
skin_mask = probability_map > threshold
skin_segmented_image = np.zeros_like(test_image)
skin_segmented_image[skin_mask] = test_image[skin_mask]


# Display the segmentation result
skin_segmented_image_rgb = cv2.cvtColor(skin_segmented_image, cv2.COLOR_BGR2RGB)
plt.imshow(skin_segmented_image_rgb)
plt.title("Segmented Skin Image")
plt.axis("off")
plt.show()

