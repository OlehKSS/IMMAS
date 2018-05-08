import multiprocessing as mp

import cv2
import numpy as np
from pandas import DataFrame
from pandas import concat
import traceback

from .binarization import get_candidates_mask
from .geometry import get_geom_features
from .intensity import get_itensity_features

def get_img_features(img, mask_ground_truth=None, contour_max_number=10, train=True):
    '''
    Function calculates features of the given image. Class id for the true positive is 1,
    and for the false positive (not masses) -1. Regions of interest that have area less than
    2500 (50 by 50) will be ignored.

    Args:
        img (numpy.array): image, which features to find
        mask_ground_truth (np.array): mask for extracting mass region, default is None.
        contour_max_number (int): maximum number of contours (without groundtruth) 
        to take into account, default is 10. In case you do not want to limit the number 
        of contours provide None as the parameter value.
        train (bool): if True, dataframe of features will be returned with correct class
        ids assigned to each candidate region, if False all class ids will be zero. Default
        value is True.

    Returns:
        (pandas.DataFrame, [opencv.contour]): features of selected contours 
        and list of contours.   
    '''

    img_thresh = get_candidates_mask(img)

    number_of_masses = 0
    min_area = 50 * 50

    if train and (not (mask_ground_truth is None)):
        # delete mass region from the mask
        _, mask_ground_truth = cv2.threshold(mask_ground_truth, 0, 1, cv2.THRESH_BINARY)
        img_thresh = img_thresh * (1 - mask_ground_truth)

        _, mass_contours, _ = cv2.findContours(mask_ground_truth,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        number_of_masses = len(mass_contours) 

    _, contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # select only contours with area higher than min area
    contours = [c for c in contours if cv2.contourArea(c) > min_area]
    # sort obtained contours by area in descending order
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    # how many contours should we investigate
    if (contour_max_number is None) or (len(contours) < contour_max_number):
        # in case we do not have enough contours in our image
        # or we do not want to limit the number of contours to select
        # we do not need to change controus array in this case
        contour_max_number = len(contours)
    else:
        # select biggest contours, according to provided max number of contours
        contours = contours[:contour_max_number]

    geom_features = get_geom_features(contours[0])
    intens_features = get_itensity_features(img, contours[0])
    features = list(geom_features.values()) + list(intens_features.values())
    features_names = list(geom_features.keys()) + list(intens_features.keys())
    # append class identificator
    features_names.append("class_id")
    # size of features array:
    # rows = number of non-mass contours + number of mass contours
    # cols = number of features + 1 (for class id)
    arr_features = np.zeros((contour_max_number + number_of_masses, 
                            len(features) + 1), 
                            dtype=float)
    arr_features[0, :-1] = features
    # since we work here with false positives
    arr_features[0, -1] = -1

    for index, contour in enumerate(contours):
        if index != 0:
            geom_features = get_geom_features(contour)
            intens_features = get_itensity_features(img, contour)
            features = list(geom_features.values()) + list(intens_features.values())

            arr_features[index, :-1] = features
            # since we work here with false positives
            arr_features[index, -1] = -1

    if train and (not (mask_ground_truth is None)):
        # appends mass contours, true positive
        for index, contour in enumerate(mass_contours):
                geom_features = get_geom_features(contour)
                intens_features = get_itensity_features(img, contour)
                features = list(geom_features.values()) + list(intens_features.values())
                # we append this data to the tail, so we need to add number of contours 
                # added previously
                arr_features[contour_max_number + index, :-1] = features
                # since we work here with false positives
                arr_features[contour_max_number + index, -1] = 1

        # add groundtruth contours to the output as well
        contours = contours + mass_contours        

    if not train:
        # class ids should be 0 if we need to return test data frame of features
        arr_features[:, -1] = 0

    return (DataFrame(arr_features, columns=features_names), contours)


def get_dataset_features(data, contour_max_number=10, train=True):
    '''
    Function returns list of features for all of the mammograms.

    Args:
        data ([MammogramImage]): list (iterable) of the mammograms from dataset.
        contour_max_number (int): maximum number of contours (without groundtruth) 
        to take into account, default is 10. In case you do not want to limit the number 
        of contours provide None as the parameter value.
        train (bool): if True, dataframe of features will be returned with correct class
        ids assigned to each candidate region, if False all class ids will be zero. Default
        value is True.

    Returns:
        pandas.DataFrame: feature of all images combined in one data table.    
    '''

    feat_data_frames = []

    for mm in data:
        try:
            mm.read_data()
            features, _ = mm.get_img_features(contour_max_number, train=train)
            feat_data_frames.append(features)
        except Exception as e:
            print(f"Caught an exception, mammogram {mm.file_name}")
            print(type(e))
            print(e.args)
            print(e)
            print(traceback.format_exc())  

    return concat(feat_data_frames, ignore_index=True)


def get_dataset_features_parallel(data, contour_max_number=10, train=True, processes=4):
    '''
    Function returns list of features for all of the mammograms.

    Args:
        data ([MammogramImage]): list (iterable) of the mammograms from dataset.
        contour_max_number (int): maximum number of contours (without groundtruth) 
        to take into account, default is 10. In case you do not want to limit the number 
        of contours provide None as the parameter value.
        train (bool): if True, dataframe of features will be returned with correct class
        ids assigned to each candidate region, if False all class ids will be zero. Default
        value is True.
        processes (int): number of processes to create for code execution.

    Returns:
        pandas.DataFrame: feature of all images combined in one data table.    
    '''

    pool = mp.Pool(processes=processes)

    feat_data_frames = [pool.apply(helper,args=(mm, contour_max_number, train)) for mm in data] 

    return concat(feat_data_frames, ignore_index=True)


def helper(img, contour_max_number, train):
    '''Helper function for running dataset feature extraction in parallel.'''
    try:
        img.read_data()
        features, _ = img.get_img_features(contour_max_number, train=train)

        return features

    except Exception as e:
        print(f"Caught an exception, mammogram {img.file_name}")
        print(type(e))
        print(e.args)
        print(e)
        print(traceback.format_exc()) 
