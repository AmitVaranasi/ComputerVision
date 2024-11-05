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