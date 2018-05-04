import numpy

def dice_similarity(segmented_images,groundtruth_images):
    '''
        Performs dice similarity score calculation.
        
        Args:
        segmented_image (uint): segmentation results we want to evaluate (1 image, treated as binary)
        groundtruth_image (uint): reference/manual/groundtruth segmentation image
        
        Returns:
        dice_index (float): DICE similarity score
        '''

    # Settings for one image
    segData = segmented_images + groundtruth_images
    TP_value = numpy.amax(segmented_images) + numpy.amax(groundtruth_images)
    TP = (segData == TP_value).sum()  # found a true positive: segmentation result and groundtruth match(both are positive)
    segData_FP = 2. * segmented_images + groundtruth_images
    segData_FN = segmented_images + 2. * groundtruth_images
    FP = (segData_FP == 2 * numpy.amax(segmented_images)).sum() # found a false positive: segmentation result and groundtruth mismatch
    FN = (segData_FN == 2 * numpy.amax(groundtruth_images)).sum() # found a false negative: segmentation result and groundtruth mismatch
    return 2*TP/(2*TP+FP+FN)  # according to the definition of DICE similarity score
