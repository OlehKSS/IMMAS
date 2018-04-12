import numpy as np
import cv2

class MammogramImage:
    '''
    Class for storing information about mammogram image.

    Args:
        image_path (str): path to an image.
        mask_path (str): path to the mask for this image.
        ground_truth_path (str): path to the corectly segmented image.
        pmuscle_mask_path (str): path to the pectoral muscle mask.
        load_data (bool): if True images data will be loaded during object instantiation, 
            otherwise you should trigger data reading manually.
    '''

    def __init__(self, image_path, mask_path, ground_truth_path=None, pmuscle_mask_path=None,
                 load_data=True):      
        self.image_data = None
        self.image_ground_truth = None
        self._image_mask = None

        self._image_path = image_path
        self._mask_path = mask_path
        self._ground_truth_path = ground_truth_path
        self._pmuscle_mask_path = pmuscle_mask_path

        if load_data:
            self.read_data()

    @property
    def uncropped_image(self):
        '''Returns uncropped image'''

        uncropped_image = self._image_mask.astype("uint16")
        
        non_zero_pxls = cv2.findNonZero(self._image_mask)
        x, y, w, h = cv2.boundingRect(non_zero_pxls)
        uncropped_image[y:y+h, x:x+w] = self.image_data

        return uncropped_image 
  

    def read_data(self):
        '''Reads data of the image'''

        if self._image_path.strip() and self._mask_path.strip():
            self._read_image(self._image_path, self._mask_path, self._pmuscle_mask_path)
        else:
            raise ValueError(
                "Image path or mask path is not correct. Image path is {0}. Mask path is {1}".format(
                    self._image_path,
                    self._mask_path
                ))

        if self._ground_truth_path and self._ground_truth_path.strip():
            self._read_ground_truth(self._ground_truth_path)
          

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

        # saves mask applied to the image, it is required for final result comparison
        self._image_mask = mask
        # crop image according to the given mask
        self.image_data = image * mask
        non_zero_pxls = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(non_zero_pxls)
        self.image_data = self.image_data[y:y+h, x:x+w]


    def _read_ground_truth(self, ground_truth_path):
        '''Reads and adds ground truth image to the object'''
  
        self.image_ground_truth = cv2.imread(ground_truth_path)
        
