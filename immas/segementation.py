import cv2, numpy

def multithresholding(img):
    '''
        Performs multi-thresholding to aid segmentation.
        
        Args:
        img (uint8): GRAYSCALE image file.
        
        Returns:
        thresholded_img (uint8): thresholded image file.
        '''
    
    if (img.shape[-1] == 3):
        raise ValueError(
                         "Error in multithresholding(): image is a color image. Image must be grayscale")
        sys.exit()
    if (numpy.amax(img) > 256):
        raise ValueError(
                         "Error in multithresholding(): image is not 8 bit")
        sys.exit()

    # Compute histogram
    bins, bins_c = numpy.histogram(img, 256)

    # Total number of pixels
    N = numpy.shape(img)[0]*numpy.shape(img)[1]
    
    # Initialize variables for the function
    MT = 0.
    maxBetweenVar = 0.
    W0K = 0.
    M0K = 0.
    optimalThresh1 = 0.
    optimalThresh2 = 0.
    
    for k in range (0,256):
        MT += float(k) *  (bins[k]/N)

        for t1 in range (0,256):
            W0K += (bins[t1]/N)
            M0K += float(t1) * (bins[t1]/N)
            M0 = M0K/ W0K
        
            W1K = 0.
            M1K = 0.
        
            for t2 in range (t1 + 1,256):
                W1K += (bins[t2]/N)
                M1K += float(t2) * (bins[t2]/N)
                M1 = float(M1K/W1K)
            
                W2K = 1. - (W0K + W1K)
                M2K = MT - (M0K + M1K)
            
                if W2K <= 0:
                    break
        
                M2 = M2K/W2K
            
                currVarB = W0K * (M0 - MT)**2 + W1K * (M1 - MT)**2 + W2K * (M2 - MT)**2
            
                if maxBetweenVar < currVarB:
                    maxBetweenVar = currVarB
                    optimalThresh1 = t1
                    optimalThresh2 = t2

    binary1 = (img > optimalThresh1)/255
    binary2 = (img > optimalThresh2)/255
    thresholded_img = binary1 * int(255/3) + binary2 * int(2*255/3)
    return thresholded_img

def mean_shift_image(img,sp,sr):
    '''
        Performs mean shifting to aid segmentation.
        
        Args:
        img (uint8, gray): image file.
        sp (int): The spatial window radius.
        sr (int): The color window radius.
        
        Returns:
        shifted_img (uint8, gray): shifted image file.
        '''
    
    if (img.shape[-1] == 3):
        raise ValueError(
"Error in multithresholding(): image is a color image. Image must be grayscale")
        sys.exit()
    
    if (numpy.amax(img) > 256):
        img = (img / 256).astype('uint8')

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    shifted_img_color = cv2.pyrMeanShiftFiltering(img_color, sp,sr)
    shifted_img_gray = cv2.cvtColor(shifted_img_color, cv2.COLOR_BGR2GRAY)

    return shifted_img_gray
