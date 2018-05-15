import numpy, cv2

from . import basic_functions
from .constants import DICE_INDEX_DEFAULT_THRESHOLD, CLASS_ID_POS, CLASS_ID_NEG

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


def find_match(img, visual_result = "no"):
    _, segmented_contours, _ = cv2.findContours(img.image_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, groundtruth_contours, _ = cv2.findContours(img.cropped_ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_mass_grd = len(groundtruth_contours)
    
    num_TP = 0
    num_FP = 0
    for i in range(0, len(segmented_contours)):
        segmented_mask = numpy.zeros(img.image_data.shape, dtype='uint8')
        cv2.drawContours(segmented_mask, [segmented_contours[i]], -1, 255, thickness=cv2.FILLED)
        DICE = numpy.zeros(len(groundtruth_contours))
        for j in range(0, len(groundtruth_contours)):
            groundtruth_mask = numpy.zeros(img.image_data.shape, dtype='uint8')
            cv2.drawContours(groundtruth_mask, [groundtruth_contours[j]], -1, 255, thickness=cv2.FILLED)
            DICE[j] = dice_similarity(segmented_mask,groundtruth_mask)
        if numpy.amax(DICE) >= 0.2:
            num_TP = num_TP + 1
        else:
            num_FP = num_FP + 1
    if visual_result == "yes":
        basic_functions.accuracy(img.image_data,img.cropped_ground_truth,"yes")

    return num_TP, num_FP, num_mass_grd


def get_rois(mask, mask_ground_truth=None, dice_threshold=DICE_INDEX_DEFAULT_THRESHOLD):
    '''
    Function for obtaining regions of interest from the mommogram.
    It will assign labels for the regions depending on the Dice index. The label for a true
    positive region is 1, for the false positive region is -1.

    Args:
        mask (numpy.array): a binarized mask of mammogram regions of interest.
        mask_ground_truth (np.array): a ground truth for the given mammogram image, optional.
        dice_threshold (int): threshold for Dice index, all regions that higher than this
        threshold will be considered as massses.

    Returns:
        ([dict], [dict]): two lists, one of true positive contours (masses) and another one of
        false positive regions. Each region is represented as an instance of class dict
        with fields:
        {
            "class_id: 1 for a mass, -1 for non-mass,
            "contour": opencv defined contour,
            "dice_index": maximum of the Dice index for the region
        } 
    '''

    # class ids for masses and non-masses
    id_tpr = CLASS_ID_POS
    id_fpr = CLASS_ID_NEG

    regions_tpr = []
    regions_fpr = []

    if mask_ground_truth is None:
        # if we don't have a ground truth for the given image we just return all ROIs as FPR
        _, segmented_contours, _ = cv2.findContours(mask, 
                                                    cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
        
        for seg_contour in segmented_contours:
            regions_fpr.append({
                "class_id": id_fpr,
                "contour": seg_contour,
                "dice_index": 0})
    else:
        _, segmented_contours, _ = cv2.findContours(mask, 
                                                    cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_SIMPLE)
        _, groundtruth_contours, _ = cv2.findContours(mask_ground_truth, 
                                                    cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_SIMPLE)

        #let's create a separate mask for each contour from ground truth image
        gt_masks = []

        for contour_gt in groundtruth_contours:
            groundtruth_mask = numpy.zeros(mask.shape, dtype='uint8')
            cv2.drawContours(groundtruth_mask, 
                            [contour_gt], 
                            -1, 
                            255, 
                            thickness=cv2.FILLED)

            gt_masks.append(groundtruth_mask)
        
        for seg_contour in segmented_contours:
            segmented_mask = numpy.zeros(mask.shape, dtype='uint8')
            cv2.drawContours(segmented_mask, [seg_contour], -1, 255, thickness=cv2.FILLED)

            dice_indices = numpy.zeros(len(gt_masks))

            for i, gt_mask in enumerate(gt_masks):
                dice_indices[i] = dice_similarity(segmented_mask, gt_mask)

            max_pos = dice_indices.argmax()   
            
            if dice_indices[max_pos] >= dice_threshold:
                regions_tpr.append({
                    "class_id": id_tpr,
                    "contour": seg_contour,
                    "dice_index": dice_indices[max_pos]})
            else:
                regions_fpr.append({
                    "class_id": id_fpr,
                    "contour": seg_contour,
                    "dice_index": dice_indices[max_pos]})            

    return regions_tpr, regions_fpr
