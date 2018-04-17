import cv2
import numpy as np
from matplotlib import pyplot as plt

PATH = './dataset/images/'
MASK = './dataset/masks/'
image_name='50993841_de5e8d61e501a71b_MG_L_CC_ANON.tif'
image_mask='50993841_de5e8d61e501a71b_MG_L_CC_ANON.mask.png'


def resize(image):
	small = cv2.resize(image, (0,0), fx=0.25, fy=0.25)
	return small

def clahe (image, CLAHE_CLIP=10.0 , CLAHE_GRID = 8):

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
    clahe = cv2.createCLAHE( CLAHE_CLIP, tileGridSize=(CLAHE_GRID,CLAHE_GRID))
    return clahe.apply(image)


if __name__== "__main__":
	
	#add assertion of file existance here
	kernel = np.ones((10,10),np.uint8)
	im = cv2.imread(PATH.__add__(image_name), 0)
	msk = cv2.imread(MASK.__add__(image_mask), 0)

	#resize input image
	im = resize(im)
	#apply image mask
	cv2.imshow('Original_image',im) 



	#apply contrast enhacement
	contrasted = clahe(im,20,8);
	cv2.imshow('Improved_image(CLAHE)',contrasted) 


	eroded = cv2.morphologyEx(im, cv2.MORPH_ERODE, kernel) 
	cv2.imshow('eroded',eroded)

	closed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
	cv2.imshow('closed',closed) 

	opened = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
	cv2.imshow('opened',opened)

	dilate = cv2.morphologyEx(im, cv2.MORPH_DILATE, kernel)
	cv2.imshow('opened',opened)


	cv2.waitKey(0) 




