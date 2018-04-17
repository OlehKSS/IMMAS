import cv2
import numpy as np
from matplotlib import pyplot as plt

PATH = '../dataset/images/'
MASK = '../dataset/masks/'
image_name='50993841_de5e8d61e501a71b_MG_L_CC_ANON.tif'
image_mask='50993841_de5e8d61e501a71b_MG_L_CC_ANON.mask.png'


def resize(image):
	small = cv2.resize(image, (0,0), fx=0.2, fy=0.2)
	return small

def CLAHE(image,grid):

	clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(grid,grid))
	image_out = clahe.apply(image)
	return image_out


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
	contrasted = CLAHE(im,8);
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




