import cv2
import numpy as np
import matplotlib.pyplot as plt

#---------------------READ ME------------------------------------------

'''
running the code

1.For the first question there is a common function convolutionOperation which will be called by the mean and median filetring 
2.Running each filtering function produces 3 or 4 images based on the filtering which is being applied
3.For running the sharpening filter code we need to un comment the sections of code which has been highlighted and comment the rest in that filter
4.For running the Filter 2d function uncomment the section which is indicated in the gaussian filter code and comment the rest
'''

#----------------------------------------------------------------------

im = cv2.imread("lena.png")
im = im.astype(float)

def convolutionOperation(im, kernel):
    kernel = np.flip(kernel)
    im_height, im_width, im_channels = im.shape
    kernel_size = kernel.shape[0]
    pad_size = int((kernel_size-1)/2)
    im_padded = np.zeros((im_height+pad_size*2, im_width+pad_size*2, im_channels))
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y+kernel_size, x:x+kernel_size, c]
                new_value = np.sum(kernel*im_patch)
                im_out[y, x, c]= new_value
    return im_out


#----------------- Mean Filtering ----------------------------------

#---------------Uncomment while running sharpening filter code------
meanKernal3x3 = np.ones((3,3))/9
meanKernal5x5 = np.ones((5,5))/25
meanKernal7x7 = np.ones((7,7))/49
meanFilteredImage3x3 = convolutionOperation(im,meanKernal3x3)
meanFilteredImage5x5 = convolutionOperation(im,meanKernal5x5)
meanFilteredImage7x7 = convolutionOperation(im,meanKernal7x7)
#-------------------------------------------------------------------

meanFilteredImage3x3 = meanFilteredImage3x3.astype(np.uint8)
meanFilteredImage5x5 = meanFilteredImage5x5.astype(np.uint8)
meanFilteredImage7x7 = meanFilteredImage7x7.astype(np.uint8)
cv2.imwrite('mean_output_image_3x3.png', meanFilteredImage3x3)
cv2.imwrite('mean_output_image_5x5.png', meanFilteredImage5x5)
cv2.imwrite('mean_output_image_7x7.png', meanFilteredImage7x7)
#---------------------------------------------------------------------

#----------------- Gaussian Filtering ----------------------------------

#---------------Uncomment while running sharpening filter/filter2D code----------
def generateGaussianFilter(size):
    size = int(size)
    kernel = np.zeros((size,size))
    center = size//2
    sigma = 1
    total_sum = 0
    for i in range(size):
        for j in range(size):
            x,y = i-center,j-center
            kernel[i,j] = (1/(2*np.pi*sigma**2))*(np.exp(-(x**2+y**2)/(2*sigma**2)))
            total_sum += kernel[i,j]
    gaussianKernal = kernel/total_sum
    return gaussianKernal

gaussianKernal3x3 = generateGaussianFilter(3)
gaussianKernal5x5 = generateGaussianFilter(5)
gaussianKernal7x7 = generateGaussianFilter(7)

gaussianFilteredImage3x3 = convolutionOperation(im,gaussianKernal3x3)
gaussianFilteredImage5x5 = convolutionOperation(im,gaussianKernal5x5)
gaussianFilteredImage7x7 = convolutionOperation(im,gaussianKernal7x7)
#-----------------------------------------------------------------------------
gaussianFilteredImage3x3 = gaussianFilteredImage3x3.astype(np.uint8)
gaussianFilteredImage5x5 = gaussianFilteredImage5x5.astype(np.uint8)
gaussianFilteredImage7x7 = gaussianFilteredImage7x7.astype(np.uint8)

cv2.imwrite('gaussian_output_image_3x3.png', gaussianFilteredImage3x3)
cv2.imwrite('gaussian_output_image_5x5.png', gaussianFilteredImage5x5)
cv2.imwrite('gaussian_output_image_7x7.png', gaussianFilteredImage7x7)
#--------------------------------------------------------------------------------

#----------------- Sharpening Filtering -----------------------------------------

sharpeningFilterImage3x3 = im-gaussianFilteredImage3x3
sharpeningFilterImage5x5 = im-gaussianFilteredImage5x5
sharpeningFilterImage7x7 = im-gaussianFilteredImage7x7
updatedSharpeningFilterImage3x3 = np.clip(im+2*sharpeningFilterImage3x3,0,255)
updatedSharpeningFilterImage5x5 = np.clip(im+2*sharpeningFilterImage5x5,0,255)
updatedSharpeningFilterImage7x7 = np.clip(im+2*sharpeningFilterImage7x7,0,255)
updatedSharpeningFilterImage3x3 = updatedSharpeningFilterImage3x3.astype(np.uint8)
updatedSharpeningFilterImage5x5 = updatedSharpeningFilterImage5x5.astype(np.uint8)
updatedSharpeningFilterImage7x7 = updatedSharpeningFilterImage7x7.astype(np.uint8)
cv2.imwrite('sharpening_gaussian_output_image_3x3.png', updatedSharpeningFilterImage3x3)
cv2.imwrite('sharpening_gaussian_output_image_5x5.png', updatedSharpeningFilterImage5x5)
cv2.imwrite('sharpening_gaussian_output_image_7x7.png', updatedSharpeningFilterImage7x7)

sharpeningFilterImage3x3 = im-meanFilteredImage3x3
sharpeningFilterImage5x5 = im-meanFilteredImage5x5
sharpeningFilterImage7x7 = im-meanFilteredImage7x7
updatedSharpeningFilterImage3x3 = np.clip(im+2*sharpeningFilterImage3x3,0,255)
updatedSharpeningFilterImage5x5 = np.clip(im+2*sharpeningFilterImage5x5,0,255)
updatedSharpeningFilterImage7x7 = np.clip(im+2*sharpeningFilterImage7x7,0,255)
updatedSharpeningFilterImage3x3 = updatedSharpeningFilterImage3x3.astype(np.uint8)
updatedSharpeningFilterImage5x5 = updatedSharpeningFilterImage5x5.astype(np.uint8)
updatedSharpeningFilterImage7x7 = updatedSharpeningFilterImage7x7.astype(np.uint8)
cv2.imwrite('sharpening_mean_output_image_3x3.png', updatedSharpeningFilterImage3x3)
cv2.imwrite('sharpening_mean_output_image_5x5.png', updatedSharpeningFilterImage5x5)
cv2.imwrite('sharpening_mean_output_image_7x7.png', updatedSharpeningFilterImage7x7)

#--------------------------------------------------------------------------------

#----------------- Midean Filtering ----------------------------------

median_im = cv2.imread("art.png")
median_im = median_im.astype(float)


def median_filtering(im, kernel_size):
    im_height, im_width, im_channels = im.shape
    pad_size = int((kernel_size-1)/2)
    im_padded = np.zeros((im_height+pad_size*2, im_width+pad_size*2, im_channels))+-1
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y+kernel_size, x:x+kernel_size, c]
                updated_im_patch = im_patch[im_patch != -1]
                new_value = np.median(updated_im_patch)
                im_out[y, x, c]= new_value
    return im_out

medianFilteredImage3x3 = median_filtering(median_im,3)
medianFilteredImage5x5 = median_filtering(median_im,5)
medianFilteredImage7x7 = median_filtering(median_im,7)
medianFilteredImage9x9 = median_filtering(median_im,9)
medianFilteredImage3x3 = medianFilteredImage3x3.astype(np.uint8)
medianFilteredImage5x5 = medianFilteredImage5x5.astype(np.uint8)
medianFilteredImage7x7 = medianFilteredImage7x7.astype(np.uint8)
medianFilteredImage9x9 = medianFilteredImage9x9.astype(np.uint8)
cv2.imwrite('median_output_image_3x3.png', medianFilteredImage3x3)
cv2.imwrite('median_output_image_5x5.png', medianFilteredImage5x5)
cv2.imwrite('median_output_image_7x7.png', medianFilteredImage7x7)
cv2.imwrite('median_output_image_9x9.png', medianFilteredImage9x9)

#---------------------------------------------------------------------

#----------------- Mean Filtering ----------------------------------

mean_im = cv2.imread("art.png")
mean_im = mean_im.astype(float)


def mean_filtering(im, kernel_size):
    im_height, im_width, im_channels = im.shape
    pad_size = int((kernel_size-1)/2)
    im_padded = np.zeros((im_height+pad_size*2, im_width+pad_size*2, im_channels))+-1
    im_padded[pad_size:-pad_size, pad_size:-pad_size, :] = im

    im_out = np.zeros_like(im)
    for c in range(im_channels):
        for x in range(im_width):
            for y in range(im_height):
                im_patch = im_padded[y:y+kernel_size, x:x+kernel_size, c]
                updated_im_patch = im_patch[im_patch != -1]
                new_value = np.mean(updated_im_patch)
                im_out[y, x, c]= new_value
    return im_out

meanFilteredImage3x3 = mean_filtering(mean_im,3)
meanFilteredImage5x5 = mean_filtering(mean_im,5)
meanFilteredImage7x7 = mean_filtering(mean_im,7)
meanFilteredImage9x9 = mean_filtering(mean_im,9)
meanFilteredImage3x3 = meanFilteredImage3x3.astype(np.uint8)
meanFilteredImage5x5 = meanFilteredImage5x5.astype(np.uint8)
meanFilteredImage7x7 = meanFilteredImage7x7.astype(np.uint8)
meanFilteredImage9x9 = meanFilteredImage9x9.astype(np.uint8)
cv2.imwrite('mean_output_art_image_3x3.png', meanFilteredImage3x3)
cv2.imwrite('mean_output_art_image_5x5.png', meanFilteredImage5x5)
cv2.imwrite('mean_output_art_image_7x7.png', meanFilteredImage7x7)
cv2.imwrite('mean_output_art_image_9x9.png', meanFilteredImage9x9)

#---------------------------------------------------------------------

#---------------------Filter2D Function ------------------------------

libFunctionOutput3x3 = cv2.filter2D(im,-1,gaussianKernal3x3)
libFunctionOutput5x5 = cv2.filter2D(im,-1,gaussianKernal5x5)
libFunctionOutput7x7 = cv2.filter2D(im,-1,gaussianKernal7x7)
libFunctionOutput3x3 = libFunctionOutput3x3.astype(np.uint8)
libFunctionOutput5x5 = libFunctionOutput5x5.astype(np.uint8)
libFunctionOutput7x7 = libFunctionOutput7x7.astype(np.uint8)
cv2.imwrite('filter2d_gaussian_3x3_output_image.png', libFunctionOutput3x3)
cv2.imwrite('filter2d_gaussian_5x5_output_image.png', libFunctionOutput5x5)
cv2.imwrite('filter2d_gaussian_7x7_output_image.png', libFunctionOutput7x7)

#---------------------------------------------------------------------

