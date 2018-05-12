import numpy as np
import cv2

from ..constants import DICE_INDEX_DEFAULT_THRESHOLD
from ..features import get_img_features

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
        file_name (str): name of the mammogram file without its extension.
    '''

    feature_names = None

    def __init__(self, image_path, mask_path, ground_truth_path=None, pmuscle_mask_path=None,
                 load_data=True, file_name=None):
        self.file_name = file_name

        self._image_data = None
        self._image_ground_truth = None
        self._image_mask = None
        # rectangle that bound brest
        self._bounding_rect = None
        # true positive (masses) and false positive ROIs
        self._regions_tpr = None
        self._regions_fpr = None

        self._image_path = image_path
        self._mask_path = mask_path
        self._ground_truth_path = ground_truth_path
        self._pmuscle_mask_path = pmuscle_mask_path

        if load_data:
            self.read_data()

    @property
    def image_data(self):
        '''Returns cropped image'''

        return self._image_data

    @image_data.setter
    def image_data(self, img):
        '''
        Sets the image data value. Will raise an error if the sizes of images
        do not match.

        Args:
            img (numpy.array): image to assign to.

        Returns: None.   
        '''

        if (self._image_data.shape == img.shape):
            self._image_data = img
        else:
            raise AttributeError(
                "Can not assign numpy array of shape {0} to array of shape {1}".format(
                    img.shape,
                    self._image_data.shape)
                )

    @property
    def has_masses(self):
        '''
        Returns True if masses are present on the groundthruth  image, 
        otherwise returns False.
        '''
        return True if self._ground_truth_path else False

    @property
    def image_ground_truth(self):
        '''Returns correct ground truth image'''
        if self._ground_truth_path:
            return self._image_ground_truth
        else:
            # generation of black image for images with no masses
            return np.zeros(self._image_mask.shape, dtype='uint8')

    @property
    def uncropped_image(self):
        '''Returns uncropped image.'''

        uncropped_image = self._image_mask.astype(self._image_data.dtype)

        x, y, w, h = self._bounding_rect.values()
        uncropped_image[y:y+h, x:x+w] = self._image_data

        return uncropped_image

    @property
    def cropped_ground_truth(self):
        '''Returns cropped ground truth image.'''
        x, y, w, h = self._bounding_rect.values()

        return self.image_ground_truth[y:y+h, x:x+w]

    @property
    def regions_tpr(self):
        '''
        Returns regions that were classified as masses as a dict object with fields.
        Each region is represented as an instance of class dict with fields:
        {
            "class_id: 1 for a mass, -1 for non-mass,
            "contour": opencv defined contour,
            "dice_index": maximum of the Dice index for the region,
            "features": numpy array of features
        }
        '''

        return self._regions_tpr

    @property
    def regions_fpr(self):
        '''
        Returns false positive regions as a dict object with fields.
        Each region is represented as an instance of class dict with fields:
        {
            "class_id: 1 for a mass, -1 for non-mass,
            "contour": opencv defined contour,
            "dice_index": maximum of the Dice index for the region,
            "features": numpy array of features
        }
        '''

        return self._regions_fpr

    def restore_background(self):
        '''
        Method for clearing background of the cropped image. Can be used to discard
        effects of the filters on the background.

        Args: None.

        Returns: None.
        '''

        x, y, w, h = self._bounding_rect.values()
        cropped_mask = self._image_mask[y:y+h, x:x+w]
        self._image_data = self._image_data * cropped_mask

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

    def get_img_features(self,
                         contour_max_number=10,
                         train=True,
                         dice_threshold=DICE_INDEX_DEFAULT_THRESHOLD):
        '''
        Finds mass candidates contours and their features.

        Args:
            contour_max_number (int): maximum number of contours (without groundtruth) 
            to take into account, default is 10. In case you do not want to limit the number 
            of contours provide None as the parameter value.
            train (bool): if True, dataframe of features will be returned with correct class
            ids assigned to each candidate region, if False all class ids will be zero. Default
            value is True.
            dice_threshold (int): threshold for Dice index, all regions that higher than this
            threshold will be considered as massses.

        Returns:
            pandas.DataFrame: features of selected number of contours.    
        '''

        if self.has_masses:
            f_c = get_img_features(self._image_data,
                                   mask_ground_truth=self.cropped_ground_truth,
                                   contour_max_number=contour_max_number,
                                   train=train,
                                   dice_threshold=dice_threshold)
        else:
            f_c = get_img_features(self._image_data,
                                   contour_max_number=contour_max_number,
                                   train=train,
                                   dice_threshold=dice_threshold)

        candidates_features, self._regions_tpr, self._regions_fpr, f_names = f_c
        # save feature names to the static variable ones
        if MammogramImage.feature_names is None:
            MammogramImage.feature_names = f_names

        return candidates_features

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
        self._image_data = image * mask
        non_zero_pxls = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(non_zero_pxls)
        self._bounding_rect = {"x": x, "y": y, "width": w, "height": h}
        self._image_data = self._image_data[y:y+h, x:x+w]

    def _read_ground_truth(self, ground_truth_path):
        '''Reads and adds ground truth image to the object'''

        self._image_ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
