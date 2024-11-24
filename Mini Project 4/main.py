import cv2
import numpy as np
import matplotlib.pyplot as plt


image = np.ones((500, 500, 3), dtype=np.uint8) * 255


a = np.array([100, 40])
cv2.circle(image, tuple(a), radius=10, color=(0, 0, 255), thickness=-1)


theta = np.radians(-60) 
rotation_matrix_origin = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

b = rotation_matrix_origin @ a  
b = b.astype(int)
cv2.circle(image, tuple(b), radius=10, color=(0, 255, 0), thickness=-1)


c = np.array([100, 100])
cv2.circle(image, tuple(c), radius=10, color=(0, 0, 0), thickness=-1)


translated_a = a - c 
d_relative = rotation_matrix_origin @ translated_a  
d = d_relative + c  
d = d.astype(int) 
cv2.circle(image, tuple(d), radius=10, color=(255, 0, 0), thickness=-1) 

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Geometric Transformations")
plt.axis("off")
plt.show()


#---------------------------------------------------------------------------
#wrap affine
#---------------------------------------------------------------------------

# # Load the image (lena.png)
# image_path = "lena.png"  # Replace with the correct path to the image
# lena_image = cv2.imread(image_path)

# # Step 1: Move the image to the right by 100 pixels and to the bottom by 200 pixels
# translation_matrix = np.array([[1, 0, 100], [0, 1, 200]], dtype=np.float32)
# translated_image = cv2.warpAffine(lena_image, translation_matrix, (lena_image.shape[1], lena_image.shape[0]))

#     # Step 2: Flip the image horizontally with respect to the image center
# flip_matrix = np.array([[-1, 0, lena_image.shape[1]], [0, 1, 0]], dtype=np.float32)
# flipped_image = cv2.warpAffine(lena_image, flip_matrix, (lena_image.shape[1], lena_image.shape[0]))

#     # Step 3: Rotate the image around the origin clockwise by 45 degrees
# theta = np.radians(-45)  # Negative for clockwise rotation
# rotation_matrix_origin = np.array([
#     [np.cos(theta), -np.sin(theta), 0],
#     [np.sin(theta),  np.cos(theta), 0]
# ], dtype=np.float32)
# rotated_image_origin = cv2.warpAffine(lena_image, rotation_matrix_origin, (lena_image.shape[1], lena_image.shape[0]))

#     # Step 4: Rotate the image around the image center clockwise by 45 degrees
# # Step 4: Rotate the image around the image center clockwise by 45 degrees
# center_x, center_y = lena_image.shape[1] // 2, lena_image.shape[0] // 2

# # Define 3x3 transformation matrices
# translation_to_center = np.array([
#     [1, 0, -center_x],
#     [0, 1, -center_y],
#     [0, 0, 1]
# ], dtype=np.float32)

# rotation_matrix_center = np.array([
#     [np.cos(theta), -np.sin(theta), 0],
#     [np.sin(theta),  np.cos(theta), 0],
#     [0, 0, 1]
# ], dtype=np.float32)

# translation_back = np.array([
#     [1, 0, center_x],
#     [0, 1, center_y],
#     [0, 0, 1]
# ], dtype=np.float32)

# # Combine transformations: Translate to center, rotate, then translate back
# combined_matrix = translation_back @ rotation_matrix_center @ translation_to_center

# # Extract the 2x3 portion for cv2.warpAffine()
# combined_matrix_2x3 = combined_matrix[:2, :]

# # Apply the combined transformation
# rotated_image_center = cv2.warpAffine(lena_image, combined_matrix_2x3, (lena_image.shape[1], lena_image.shape[0]))
#     # Display all images
# images = [translated_image, flipped_image, rotated_image_origin, rotated_image_center]
# titles = [
#     "Translated (100px right, 200px down)",
#     "Flipped Horizontally",
#     "Rotated 45° Clockwise (Origin)",
#     "Rotated 45° Clockwise (Center)"
# ]

# plt.figure(figsize=(12, 8))
# for i, (img, title) in enumerate(zip(images, titles)):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.title(title)
#     plt.axis("off")
# plt.tight_layout()
# plt.show()