import cv2

from ..segmentation import multithresholding
from ..preprocessing import clahe

def get_candidates_mask(img):
    '''
    Function that applies preprocessing techniques and returns binary image for contour and
    feature extraction of mass candidates.

    Args:
        img (numpy.array): mammogram as plain image, from which we should obtain binary image.

    Returns:
        numpy.array: binary images that will be used for contour and feature extraction.    
    '''

    img_clahe = clahe(img)
    img_thresh = multithresholding(img_clahe)
    _, img_thresh = cv2.threshold(img_thresh, img_thresh.max() - 1, 1, cv2.THRESH_BINARY)

    return img_thresh
    