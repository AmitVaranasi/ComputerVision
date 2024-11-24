import numpy as np
import cv2
import math

def calculate_hs_histogram(img, bin_size):
    height, width, _ = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    max_h = 179
    max_s = 255
    hs_hist = np.zeros((math.ceil(max_h+1/bin_size), math.ceil(max_s+1/bin_size)))
    for i in range(height):
        for j in range(width):
            h = img_hsv[i, j, 0]
            s = img_hsv[i, j, 1]
            hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] += 1
    # hs_hist /= hs_hist.sum()
    return hs_hist

def color_segmentation(img, hs_hist, bin_size, threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width, 1))
    for i in range(height):
        for j in range(width):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            if hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] > threshold:
                mask[i, j, 0] = 1
    return mask


# Training
bin_size = 20
max_h = 179
max_s = 255
hs_hist = np.zeros((math.ceil(max_h+1/bin_size), math.ceil(max_s+1/bin_size)))
for i in range(1,13):
    print(i)
    img_train = cv2.imread(f'/Users/amitmaheshwarvaranasi/Desktop/{i}.png')
    hs_hist+=calculate_hs_histogram(img_train, bin_size)
avg_hist = hs_hist/hs_hist.sum()
# Testing
img_test = cv2.imread("./testing_image.bmp")

threshold = 0.03
mask = color_segmentation(img_test, avg_hist, bin_size, threshold)

img_seg = img_test * mask

cv2.imshow("Input", img_test)
cv2.imshow("Mask", (mask*255).astype(np.uint8))
cv2.imshow("Segmentation", img_seg.astype(np.uint8))
cv2.waitKey()


#-------------------------------------------------------------------------------------

# 2d Gaussian Distribution code

#-------------------------------------------------------------------------------------


import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# Define the multivariate Gaussian PDF function
def multivariate_gaussian_pdf(x, mean, det_sigma, inv_sigma):
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
hs_values_collection = []
sum_cov = np.zeros((2, 2))
sum_mean = np.zeros((2,))
for i in range(1, 13):
    img_train = cv2.imread(f'/Users/amitmaheshwarvaranasi/Desktop/{i}.png')
    #cov, mean = calculate_hs_histogram(img_train)
    hsv_image = cv2.cvtColor(img_train, cv2.COLOR_BGR2HSV)
    train_hue, train_saturation = hsv_image[:, :, 0], hsv_image[:, :, 1]
    hs_values = np.column_stack((train_hue.flatten(),train_saturation.flatten()))
    hs_values_collection.append(hs_values)

hs_values_collection = np.vstack(hs_values_collection)
training_datamean = np.mean(hs_values_collection,axis=0)
trainingdata_covarience = np.zeros((2, 2))

for i in range(len(hs_values_collection)):
    diff = hs_values_collection[i] - training_datamean
    diff_reshape = diff.reshape(-1, 1)
    trainingdata_covarience += np.dot(diff_reshape, diff_reshape.T)

trainingdata_covarience/=len(hs_values_collection)-1

det_sigma = np.linalg.det(trainingdata_covarience)
inv_sigma = np.linalg.inv(trainingdata_covarience)

# Testing
test_image = cv2.imread('./testing_image.bmp')
hsv_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)
hue, saturation = hsv_image[:, :, 0], hsv_image[:, :, 1]
pixels = np.column_stack((hue.flatten(), saturation.flatten()))

probabilities = np.array([
    multivariate_gaussian_pdf(pixel, training_datamean, det_sigma, inv_sigma)
    for pixel in pixels
])
probability_map = probabilities.reshape(hue.shape)

# Thresholding and segmentation
threshold = 0.0001 # Adjust the threshold as needed
skin_mask = probability_map > threshold
skin_segmented_image = np.zeros_like(test_image)
skin_segmented_image[skin_mask] = test_image[skin_mask]

# Display the segmentation result
skin_segmented_image_rgb = cv2.cvtColor(skin_segmented_image, cv2.COLOR_BGR2RGB)
plt.imshow(skin_segmented_image_rgb)
plt.title("Segmented Skin Image")
plt.axis("off")
plt.show()

#-------------------------------------------------------------------------------------

# Harris corner detection

#-------------------------------------------------------------------------------------


import cv2
import numpy as np
import matplotlib.pyplot as plt

checkerboard_img = cv2.imread('./checkboard.png')
toy_img = cv2.imread('./toy.png')

def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

  
    dst = cv2.cornerHarris(src=gray, blockSize=block_size, ksize=ksize, k=k)

    dst = cv2.dilate(dst, None)

    image[dst > threshold * dst.max()] = [0, 0, 255] 
    return image
checkerboard_corners = harris_corner_detection(checkerboard_img.copy())
toy_corners = harris_corner_detection(toy_img.copy())

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(cv2.cvtColor(checkerboard_corners, cv2.COLOR_BGR2RGB))
ax[0].set_title('Checkerboard Corners')
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(toy_corners, cv2.COLOR_BGR2RGB))
ax[1].set_title('Toy Image Corners')
ax[1].axis('off')

plt.show()