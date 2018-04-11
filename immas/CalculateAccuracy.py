"""
Write a function which finds the accuracy of a given output image given the image and the ground truth
Based on Alessandro Bria's C++ file compute-segmentation-accuracy.cpp

Accuracy (ACC) is defined as:
    ACC = (True positives + True negatives)/(number of samples)
    i.e., as the ratio between the number of correctly classified samples (in our case, pixels)
    and the total number of samples (pixels)

Input:  segemented_images - segmentation results we want to evaluate (1 or more images, treated as binary)
        groundtruth_images - reference/manual/groundtruth segmentation images
        visual_results - false color images displaying the comparison between automated segmentation results and groundtruth
        True positives = blue, True negatives = gray, False positives = yellow, False negatives = red, in format "yes" or "no"
        num_images - put 1 if one image, otherwise leave blank
"""
import sys
import numpy

def accuracy(segmented_images, groundtruth_images, mask_images, visual_results, num_images=2):

    # True positives (TP), True negatives (TN), and total number N of pixels are all we need
    TP = 0
    TN = 0
    N = 0

    #Settings for one image
    if num_images == 1:
        if visual_results == "no":
            segData = segmented_images + groundtruth_images
            TP = TP + (segData == numpy.amax(
                    segData)).sum()  # found a true positive: segmentation result and groundtruth match(both are positive)
            TN = TN + (
                            segData == 0).sum()  # found a true negative: segmentation result and groundtruth match(both are positive)
            N = N + (segData.shape[0] * segData.shape[1])
        else:
            sys.exit()

    #All other cases
    else:

        num_segmented_images = len(segmented_images)
        num_groundtruth_images = len(groundtruth_images)
        num_mask_images = len(mask_images)

        # Checks to ensure input are of the correct format
        if 0 == num_segmented_images:
            raise ValueError("Error in accuracy(): the set of segmented images is empty")
            sys.exit()
        elif num_groundtruth_images != num_segmented_images:
            raise ValueError(
                "Error in accuracy(): the number of groundtruth images {} is different than the number of segmented images {}".format(
                    num_groundtruth_images, num_segmented_images))
            sys.exit()
        elif num_mask_images != num_segmented_images:
            raise ValueError(
                "Error in accuracy(): the number of mask images {} is different than the number of segmented images{}".format(
                    num_mask_images, num_segmented_images))
            sys.exit()

        if visual_results == "no":
            for i in range(0, num_segmented_images):
                segData = segmented_images[i] + groundtruth_images[i]
                TP = TP + (segData == numpy.amax(segData)).sum() # found a true positive: segmentation result and groundtruth match(both are positive)
                TN = TN + (segData == 0).sum() # found a true negative: segmentation result and groundtruth match(both are positive)
                N = N + (segData.shape[0] * segData.shape[1])

        else:
            sys.exit()

    return (TP + TN) / N;  # according to the definition of Accuracy
