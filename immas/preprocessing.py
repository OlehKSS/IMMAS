import numpy as np
import cv2

def clahe (img, clip=10.0, grid=(8,8)):
    '''
    Applies Limited Adaptive Histogram Equalization (CLAHE) to an image.
    Local details are enhanced even in regions that are darker or lighter than most of the image.
        
    Args:
        img (uint8): image file.
        clip (float): contrast limit (default = 10.0). The pixels above are clipped and distributed uniformly to other bins before applying histogram equalization.        
        grid (tuple): size of the block (default = (8,8)) of the image where histogram equalization is going to be performed.

    Returns:
        (new_img): image obtained after CLAHE application. 
    '''

    clahe = cv2.createCLAHE(clip,grid)
    new_img = clahe.apply(img)
    return new_img