import cv2, numpy, sys
from sklearn.metrics import jaccard_similarity_score
from immas import preprocessing

def multithresholding(img):
    '''
        Performs multi-thresholding to aid segmentation.
        
        Args:
        img (uint): GRAYSCALE image file.
        
        Returns:
        thresholded_img (uint8): thresholded image file.
        '''
    
    if (img.shape[-1] == 3):
        raise ValueError(
                         "Error in multithresholding(): image is a color image. Image must be grayscale")
        sys.exit()
    if (numpy.amax(img) > 256):
        img = (img/ 256).astype('uint8')

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

    binary1 = (img > optimalThresh1)
    binary2 = (img > optimalThresh2)
    thresholded_img = (binary1 * int(255/3) + binary2 * int(2*255/3)).astype('uint8')
    return thresholded_img

def mean_shift(img,sp,sr):
    '''
        Performs mean shifting to aid segmentation.
        
        Args:
        img (uint): image file.
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

def thresh_to_binary(img):
    binary_img = ((img > 200) * 255).astype('uint8')
    return binary_img

def jaccard_index(segmented_images,groundtruth_images):
    
    
    '''
        Performs jaccard index calculation on an image or set of images to assess mass mask placement.
        
        Args:
        segmented_images(uint8): GRAYSCALE predicted mask image 0R list of predicted mask images.
        groundtruth_images(uint8): GRAYSCALE groundtruth mask image 0R list of groundtruth mask images.
        
        Returns:
        av_jaccard_index (float): max jaccard index for one image or average max jaccard index for several images.
        '''
    
    if len(numpy.shape(segmented_images)) == 2:
        
        if numpy.shape(segmented_images) != numpy.shape(groundtruth_images):
            raise ValueError(
                             "Error in jaccard_index(): the number of groundtruth images is different than the number of segmented images")
            sys.exit()
        elif segmented_images.dtype != 'uint8':
            raise ValueError("Error in jaccard_index(): segmented_images are not uint8")
            sys.exit()
        elif groundtruth_images.dtype != 'uint8':
            raise ValueError("Error in jaccard_index(): groundtruth_images are not uint8")
            sys.exit()
        
        _, contours, _ = cv2.findContours(segmented_images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        save_jaccard = numpy.zeros(len(contours))
        if len(contours) == 0:
            av_max_jaccard = 0
        else:
            for i in range(0, len(contours)):
                mask = numpy.zeros(segmented_images.shape, dtype='uint8')
                cv2.drawContours(mask, [contours[i]], -1, 255, thickness=cv2.FILLED)
                save_jaccard[i] = jaccard_similarity_score(groundtruth_images, mask)
            av_max_jaccard = numpy.amax(save_jaccard)
    
    else:
        num_segmented_images = len(segmented_images)
        num_groundtruth_images = len(groundtruth_images)
        
        # Checks to ensure input are of the correct format
        if 0 == num_segmented_images:
            raise ValueError("Error in jaccard_index(): the set of segmented images is empty")
            sys.exit()
        elif num_groundtruth_images != num_segmented_images:
            raise ValueError(
                             "Error in jaccard_index(): the number of groundtruth images {} is different than the number of segmented images {}".format(num_groundtruth_images, num_segmented_images))
            sys.exit()
        elif segmented_images[0].dtype != 'uint8':
            raise ValueError("Error in jaccard_index(): segmented_images are not uint8")
            sys.exit()
        elif groundtruth_images[0].dtype != 'uint8':
            raise ValueError("Error in jaccard_index(): groundtruth_images are not uint8")
            sys.exit()

        max_jaccard = 0
        for j in range (0,num_segmented_images):
            if len(contours) == 0:
                max_jaccard = max_jaccard + 0
            else:
                _, contours, _ = cv2.findContours(segmented_images[j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                save_jaccard = numpy.zeros(len(contours))
                for i in range(0, len(contours)):
                    mask = numpy.zeros(segmented_images[j].shape, dtype='uint8')
                    cv2.drawContours(mask, [contours[i]], -1, 255, thickness=cv2.FILLED)
                    save_jaccard[i] = jaccard_similarity_score(groundtruth_images[j], mask)
                max_jaccard = max_jaccard + numpy.amax(save_jaccard)

        av_max_jaccard = max_jaccard/num_segmented_images
    
    return av_max_jaccard


def fullSegmentation(img):
    '''
    Applies specific segmentation sequence to an image

    Args:
        img (uint): image file.

    Returns:
        new_img (uint8): image obtained after segmentation
    '''
    img = mean_shift(img, 20, 20)
    img = multithresholding(img)
    img = thresh_to_binary(img)
    img = preprocessing.open(img, (20, 20))
 
    return img