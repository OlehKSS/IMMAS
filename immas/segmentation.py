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
                             "Error in jaccard_index(): the size of groundtruth image is different than the size of segmented image")
            sys.exit()
        elif segmented_images.dtype != 'uint8':
            raise ValueError("Error in jaccard_index(): segmented_images are not uint8")
            sys.exit()
        elif groundtruth_images.dtype != 'uint8':
            raise ValueError("Error in jaccard_index(): groundtruth_images are not uint8")
            sys.exit()
        
        _, segmented_contours, _ = cv2.findContours(segmented_images, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(segmented_contours) == 0:
            av_max_jaccard = 0
        else:
            _, groundtruth_contours, _ = cv2.findContours(groundtruth_images, cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_SIMPLE)
            save_jaccard = numpy.zeros(len(segmented_contours) * len(groundtruth_contours))
            counter = 0
            for i in range(0, len(segmented_contours)):
                for j in range(0, len(groundtruth_contours)):
                    groundtruth_mask = numpy.zeros(segmented_images.shape, dtype='uint8')
                    cv2.drawContours(groundtruth_mask, [groundtruth_contours[j]], -1, 255, thickness=cv2.FILLED)
                    segmented_mask = numpy.zeros(segmented_images.shape, dtype='uint8')
                    cv2.drawContours(segmented_mask, [segmented_contours[i]], -1, 255, thickness=cv2.FILLED)
                    summed_images = (groundtruth_mask / 2) + (segmented_mask / 2)
                    union_pixels = (summed_images.shape[0] * summed_images.shape[1]) - (summed_images == 0).sum()
                    intersection_pixels = (summed_images == 255).sum()
                    save_jaccard[counter] = intersection_pixels / union_pixels
                    counter = counter + 1
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
        num_images = num_segmented_images

        max_jaccard = 0
        for j in range(0, num_images):
            _, segmented_contours, _ = cv2.findContours(segmented_images[j], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(segmented_contours) == 0:
                continue
            else:
                _, groundtruth_contours, _ = cv2.findContours(groundtruth_images[j], cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_SIMPLE)
                save_jaccard = numpy.zeros(len(segmented_contours) * len(groundtruth_contours))
                counter = 0
                for i in range(0, len(segmented_contours)):
                    for k in range(0, len(groundtruth_contours)):
                        groundtruth_mask = numpy.zeros(segmented_images[j].shape, dtype='uint8')
                        cv2.drawContours(groundtruth_mask, [groundtruth_contours[k]], -1, 255, thickness=cv2.FILLED)
                        segmented_mask = numpy.zeros(segmented_images[j].shape, dtype='uint8')
                        cv2.drawContours(segmented_mask, [segmented_contours[i]], -1, 255, thickness=cv2.FILLED)
                        summed_images = (groundtruth_mask / 2) + (segmented_mask / 2)
                        union_pixels = (summed_images.shape[0] * summed_images.shape[1]) - (summed_images == 0).sum()
                        intersection_pixels = (summed_images == 255).sum()
                        save_jaccard[counter] = intersection_pixels / union_pixels
                        counter = counter + 1
                print(save_jaccard)
                max_jaccard = max_jaccard + numpy.amax(save_jaccard)
        av_max_jaccard = max_jaccard / num_images

    return av_max_jaccard


def fullSegmentation(img):
    '''
    Applies specific segmentation sequence to an image

    Args:
        img (uint): image file.

    Returns:
        new_img (uint8): image obtained after segmentation
    '''
    img = multithresholding(img)
    img = thresh_to_binary(img)
    img = preprocessing.open(img, (23, 23))
    img = preprocessing.close(img,(5,5))
    return img
