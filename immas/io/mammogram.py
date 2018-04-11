import numpy as np
import cv2

class MammogramImage:
    '''Class for storing information about mammogram image.
        Args:
            image_path (str): path to an image.
            ground_truth_path (str): path to the corectly segmented image.
            mask_path (str): path to the mask for this image.
            pmuscle_mask_path (str): path to the pectoral muscle mask.
    '''

    def __init__(self, image_path, mask_path, ground_truth_path=None, pmuscle_mask_path=None):      
        self.image_data = None
        self.image_ground_truth = None

        if image_path.strip() and mask_path.strip():
            self._read_image(image_path, mask_path, pmuscle_mask_path)
        else:
            raise ValueError(
                "Image path or mask path is not correct. Image path is {0}. Mask path is {1}".format(
                    image_path,
                    mask_path
                ))

        if ground_truth_path and ground_truth_path.strip():
            self._read_ground_truth(ground_truth_path)    

                

    def _read_image(self, image_path, mask_path, pmuscle_mask_path=None):
        '''Reads image and applies mask to it.'''
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        # if we have pectoral muscle mask we will remove the muscle
        if pmuscle_mask_path and pmuscle_mask_path.strip():
            pectoral_mask = cv2.imread(pmuscle_mask_path, cv2.IMREAD_GRAYSCALE)
            _, pectoral_mask = cv2.threshold(pectoral_mask, 127, 1, cv2.THRESH_BINARY)
            # combine two masks
            mask = mask * (1 - pectoral_mask)

        # crop image according to the given mask
        self.image_data = image * mask
        non_zero_pxls = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(non_zero_pxls)
        self.image_data = self.image_data[y:y+h, x:x+w]


    def _read_ground_truth(self, ground_truth_path):
        '''Reads and adds ground truth image to the object'''    
        self.image_ground_truth = cv2.imread(ground_truth_path)
        
