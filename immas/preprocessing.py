import numpy as np
import cv2


def resize(image, fx, fy):
    
    '''
    Resizes the image down to or up to the specified size. 
    Args:
        img (uint16): image file.
        fx (float): (default = 0.2)scale factor along the horizontal axis;        
        fy (float): (default = 0.2)scale factor along the vertical axis;

    Returns:
        image obtained after resizing . 
    '''
    return cv2.resize(image, (0,0), fx=0.2, fy=0.2)
	

def open(image, kernel_size = (10,10)):
    
    '''
    Performs a morphological transformations with OPENING operation.
    Args:
        img (uint16): image file.
        kernel_size (tuple): size of the kernel to be used (default = (10,10)) 

    Returns:
        image obtained after opening. 
    '''
    KERNEL = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, KERNEL)
    

def close(image,  kernel_size = (10,10)):

    '''
    Performs a morphological transformations with CLOSE operation.
    Args:
        img (uint16): image file.
        kernel_size (tuple): size of the kernel to be used (default = (10,10)) 

    Returns:
        image obtained after opening. 

    '''
    KERNEL = np.ones(kernel_size,np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, KERNEL)

def erode(image,  kernel_size = (10,10)):

    '''
    Performs a morphological transformations with EROSION operation.
    Args:
        img (uint16): image file.
        kernel_size (tuple): size of the kernel to be used (default = (10,10)) 

    Returns:
        image obtained after opening. 

    '''
    KERNEL = np.ones(kernel_size,np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, KERNEL)

def dilate(image,  kernel_size = (10,10)):

    '''
    Performs a morphological transformations with DILATION operation.
    Args:
        img (uint16): image file.
        kernel_size (tuple): size of the kernel to be used (default = (10,10)) 

    Returns:
        image obtained after opening. 

    '''
    KERNEL = np.ones(kernel_size,np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_DILATE, KERNEL)
    

def clahe (image, CLAHE_CLIP=10, CLAHE_GRID = 8):

    '''
    Applies Limited Adaptive Histogram Equalization (CLAHE) to an image.
    Local details are enhanced even in regions that are darker or lighter than most of the image.
        
    Args:
        img (uint16): image file.
        clip (float): contrast limit (default = 10.0). The pixels above are clipped and distributed uniformly to other bins before applying histogram equalization.        
        grid (tuple): size of the block (default = (8,8)) of the image where histogram equalization is going to be performed.

    Returns:
         image obtained after CLAHE application. 
    '''
    clahe = cv2.createCLAHE( CLAHE_CLIP=10.0, tileGridSize=(CLAHE_GRID,CLAHE_GRID))
    return clahe.apply(image)
 
