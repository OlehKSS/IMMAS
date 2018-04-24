import cv2, numpy

def show_image(img, window_name):
    '''
        Shows images using cv2 imshow in smaller windows. Close by pressing any key
        
        Args:
        img (uint8): image file.
        window_name (string): name of window to be displayed
        
        Returns:
        0
        '''
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 332, 408)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    return 0

def multithresholding(img):
    '''
        Performs multi-thresholding to aid segmentation (2 thresholds).
        
        Args:
        img (uint8): GRAYSCALE image file.
        
        Returns:
        thresholded_img (uint8): thresholded image file.
        '''
    
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
        img (uint8,3-channel): COLOR image file.
        sp (int): The spatial window radius.
        sr (int): The color window radius.
        
        Returns:
        shifted_img (uint8,3-channel): shifted image file.
    '''
    shifted_img = cv2.pyrMeanShiftFiltering(img, sp,sr)
    
    return shifted_img
