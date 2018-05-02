import cv2
import numpy as np
from matplotlib import pyplot as plt
import pywt
import math
from scipy.ndimage.filters import median_filter

PATH = './dataset/images/'
MASK = './dataset/masks/'
image_name='22580098_6200187f3f1ccc18_MG_L_ML_ANON.tif'
image_mask='22580098_6200187f3f1ccc18_MG_L_ML_ANON.mask.png'


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
	contrasted = clahe(im,10,8);
	#cv2.imshow('Improved_image(CLAHE)',contrasted) 

	titles = ['Approximation', ' Horizontal detail',
	          'Vertical detail', 'Diagonal detail']
	coeffs2 = pywt.dwt2(contrasted, 'db4')
	LL, (LH, HL, HH) = coeffs2
	fig = plt.figure()
	for i, a in enumerate([LL, LH, HL, HH]):
	    plt.subplot(2, 2, i + 1)
	    plt.imshow(a, origin='image', interpolation="nearest", cmap=plt.cm.gray)
	    plt.title(titles[i], fontsize=12)
	    
	plt.suptitle("dwt2 coefficients", fontsize=14)
	plt.show()

	print ("After loop")
	LL = math.sqrt(2)*LL
	LH = median_filter(LH, 5)
	HL = median_filter(HL, 5)
	HH = median_filter(HH, 5)
	coeffs2 = LL, (LH, HL, HH) 
	plt.subplot(2,2,1)
	plt.imshow(contrasted, cmap="gray")
	plt.axis('off')
	plt.title('Before wavelet transform')
	
	reconstructed = pywt.idwt2(coeffs2, 'db4')
	plt.subplot(2,2,2)
	plt.imshow(reconstructed, interpolation="nearest", cmap=plt.cm.gray)
	plt.title('After wavelet transform')
	

	plt.subplot(2,2,3)
	plt.imshow(contrasted - reconstructed[:contrasted.shape[0],:contrasted.shape[1]], interpolation="nearest", cmap=plt.cm.gray)
	plt.title('Subtraction')
	plt.show()

	cv2.waitKey(0) 




