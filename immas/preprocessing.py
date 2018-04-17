import numpy as np
import cv2


# Contrast Limited Adaptive Histogram Equalization (CLAHE)
# An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image.
# Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.
# Inputs:
# img - image
# clip - If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization
# grid - size of the block of the image where histogram equalization is going to be performed
# Output:
# new_img - image obtained after CLAHE use

#global parameters
CLAHE_GRID = 8;
CLAHE_CLIP = 20;

KERNEL = np.ones((10,10),np.uint8)


def resize(image):
	small = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
	return small

def open(image):
	return cv2.morphologyEx(image, cv2.MORPH_OPEN, KERNEL)

def close(image):
	return cv2.morphologyEx(image, cv2.MORPH_CLOSE, KERNEL)

def erode(image):
	return cv2.morphologyEx(image, cv2.MORPH_ERODE, KERNEL)

def dilate(image):
    return cv2.morphologyEx(image, cv2.MORPH_DILATE, KERNEL)

def clahe (image, CLAHE_CLIP , CLAHE_GRID = 8):

    '''
    Applies Limited Adaptive Histogram Equalization (CLAHE) to an image.
    Local details are enhanced even in regions that are darker or lighter than most of the image.
        
    Args:
        img (uint8): image file.
        clip (float): contrast limit (default = 10.0). The pixels above are clipped and distributed uniformly to other bins before applying histogram equalization.        
        grid (tuple): size of the block (default = (8,8)) of the image where histogram equalization is going to be performed.

    Returns:
         image obtained after CLAHE application. 
    '''
    clahe = cv2.createCLAHE( CLAHE_CLIP=10.0, tileGridSize=(CLAHE_GRID,CLAHE_GRID))
    return clahe.apply(image)
 
