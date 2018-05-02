import numpy as np
import cv2
import math
import pywt
from scipy.ndimage.filters import median_filter


def resize(image, fx=0.25, fy=0.25):
    
    '''
    Resizes the image down to or up to the specified size. 
    Args:
        img (uint16): image file.
        fx (float): (default = 0.2)scale factor along the horizontal axis;        
        fy (float): (default = 0.2)scale factor along the vertical axis;

    Returns:
        image obtained after resizing . 
    '''
    return cv2.resize(image, (0,0), fx=fx, fy=fy)
	

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
    

def clahe (image, clip=10.0, grid=8):

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
    myclahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
    return myclahe.apply(image)
 
def waveletTransform (image,  kernel_size =5):
    '''
    Wavelet transform performs sigle level 2D wavelet transform followed by median filtering and reconstruction by inverse wavelet transform.

    Args:
        img (uint16): image file.
        kernel : size of kernel used for median filter(default = 5)

    Returns:
         image obtained after inverse transfrom of filtered image details (converted to int).

    The dwt2() function performs single level 2D Discrete Wavelet Transform.
    Parameters: 
    data – img
    wavelet – Wavelet to use in the transform. It defaultly uses "db4" also named as Daubechies 4
    '''
    coeffs2 = pywt.dwt2(image, 'db4')
    LL, (LH, HL, HH) = coeffs2
    LL = math.sqrt(2)*LL 
    LH = median_filter(LH, kernel_size)
    HL = median_filter(HL, kernel_size)
    HH = median_filter(HH, kernel_size)
    coeffs2 = LL, (LH, HL, HH) 
    result = pywt.idwt2(coeffs2, 'db4')
    imgmax = np.max(result)
    imgmin = np.min(result)
    newmax = 65535
    newmin = 0
    result = (((result - imgmin) * ((newmax - newmin)/(imgmax - imgmin))) + newmin).astype('uint16')
    return result


def morphoEnhancement(image, kernel_size = 20):
    
    '''
    Top hat approach for morphological enhancement
    
    Args:
        image (uint16):     image file
        kernel_size (float):size of kernel for morphological operations, by 
        default 20x20
        
    Returns:
        image enhanced 
        
    It is possible to add the bright areas (top hat) to the image and subtract
    the dark areas (bottom hat) from it. As a result, there will be an enhancement 
    in the contrast between bright and dark areas. To improve the contrast even 
    more clahe is applied
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    bothat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    enhanced = image + tophat - bothat
    
    final = clahe(enhanced, 2.0)
    return final

def fullPreprocessing (img):
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

    img = morphoEnhancement(img)
    img = waveletTransform(img)
    return img
