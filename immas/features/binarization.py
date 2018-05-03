import cv2

from ..segmentation import fullSegmentation
from ..preprocessing import fullPreprocessing

def get_candidates_mask(img):
    '''
    Function that applies preprocessing techniques and returns binary image for contour and
    feature extraction of mass candidates.

    Args:
        img (numpy.array): mammogram as plain image, from which we should obtain binary image.

    Returns:
        numpy.array: binary images that will be used for contour and feature extraction.    
    '''

    img_processed = fullPreprocessing(img)
    img_thresh = fullSegmentation(img_processed)

    return img_thresh
    