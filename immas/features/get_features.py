import multiprocessing as mp

import cv2
import numpy as np
from pandas import DataFrame
from pandas import concat
import traceback

from .binarization import get_candidates_mask
from .geometry import get_geom_features
from .intensity import get_itensity_features

from ..constants import DICE_INDEX_DEFAULT_THRESHOLD, CLASS_ID_POS, CLASS_ID_NEG, MIN_ROI_AREA
from ..classification import get_rois


def get_img_features(img,
                     mask_ground_truth=None,
                     contour_max_number=10,
                     train=True,
                     dice_threshold=DICE_INDEX_DEFAULT_THRESHOLD):
    '''
    Function calculates features of the given image. Class id for the true positive is 1,
    and for the false positive (not masses) -1. Regions of interest that have area less than
    2500 will be ignored.

    Args:
        img (numpy.array): image, which features to find
        mask_ground_truth (np.array): mask for extracting mass region, default is None.
        contour_max_number (int): maximum number of contours (without groundtruth) 
        to take into account, default is 10. In case you do not want to limit the number 
        of contours provide None as the parameter value.
        train (bool): if True, dataframe of features will be returned with correct class
        ids assigned to each candidate region, if False all class ids will be zero. Default
        value is True.
        dice_threshold (int): threshold for Dice index, all regions that higher than this
        threshold will be considered as massses.

    Returns:
        (pandas.DataFrame, [dict], [str]): features of selected contours, list of
        true positive and false positive regions of interest, list of
        feature names. Each region is represented as an instance of class dict with fields:
        {
            "class_id: 1 for a mass, -1 for non-mass,
            "contour": opencv defined contour,
            "dice_index": maximum of the Dice index for the region,
            "features": numpy array of features
        } 
    '''

    img_thresh = get_candidates_mask(img)

    number_of_masses = 0
    min_area = MIN_ROI_AREA

    if train:
        id_tpr = CLASS_ID_POS
        id_fpr = CLASS_ID_NEG
    else:
        id_tpr = 0
        id_fpr = 0

    regions_tpr, regions_fpr = get_rois(img_thresh, mask_ground_truth, dice_threshold)

    number_of_masses = len(regions_tpr)

    # select only contours with area higher than min area
    regions_fpr = [c for c in regions_fpr if cv2.contourArea(c["contour"]) > min_area]
    # sort obtained contours by area in descending order
    regions_fpr.sort(key=lambda c: cv2.contourArea(c["contour"]), reverse=True)

    # how many contours should we investigate
    if (contour_max_number is None) or (len(regions_fpr) < contour_max_number):
        # in case we do not have enough contours in our image
        # or we do not want to limit the number of contours to select
        # we do not need to change controus array in this case
        contour_max_number = len(regions_fpr)
    else:
        # select biggest contours, according to provided max number of contours
        regions_fpr = regions_fpr[:contour_max_number]

    arr_features = None

    for index, region in enumerate(regions_fpr):
        contour = region["contour"]

        if index == 0:
            geom_features = get_geom_features(contour)
            intens_features = get_itensity_features(img, contour)

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
        else:
            geom_features = get_geom_features(contour)
            intens_features = get_itensity_features(img, contour)
            features = list(geom_features.values()) + list(intens_features.values())

        region["features"] = features
        arr_features[index, :-1] = features
        # since we work here with false positives
        arr_features[index, -1] = id_fpr

    for index, region in enumerate(regions_tpr):
        contour = region["contour"]

        geom_features = get_geom_features(contour)
        intens_features = get_itensity_features(img, contour)
        features = list(geom_features.values()) + list(intens_features.values())

        if (arr_features is None) and (index == 0):
            features_names = list(geom_features.keys()) + list(intens_features.keys())
            # append class identificator
            features_names.append("class_id")
            # size of features array:
            # rows = number of non-mass contours + number of mass contours
            # cols = number of features + 1 (for class id)
            arr_features = np.zeros((contour_max_number + number_of_masses,
                                    len(features) + 1),
                                    dtype=float)

        region["features"] = features
        # we append this data to the tail, so we need to add number of contours 
        # added previously
        arr_features[contour_max_number + index, :-1] = features
        # since we work here with false positives
        arr_features[contour_max_number + index, -1] = id_tpr

    features_df = DataFrame(arr_features, columns=features_names)
    regions = regions_tpr + regions_fpr
    # class_id will be excluded from feature names in the output
    return (features_df, regions, features_names[:-1])


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
            features = mm.get_img_features(contour_max_number, train=train)
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
        features = img.get_img_features(contour_max_number, train=train)

        return features

    except Exception as e:
        print(f"Caught an exception, mammogram {img.file_name}")
        print(type(e))
        print(e.args)
        print(e)
        print(traceback.format_exc())
