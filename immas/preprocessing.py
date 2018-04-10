import numpy as np
import cv2

# Contrast Limited Adaptive Histogram Equalization (CLAHE)
# An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image.
# Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.

def clahe (img, clip=2.0, grid=(8,8)):
    clahe = cv2.createCLAHE(clip,grid)
    new_img = clahe.apply(img)
    return new_img