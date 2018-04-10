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

def clahe (img, clip=10.0, grid=(8,8)):
    clahe = cv2.createCLAHE(clip,grid)
    new_img = clahe.apply(img)
    return new_img