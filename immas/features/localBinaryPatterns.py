# -*- coding: utf-8 -*-
"""
Created on Sun May 20 15:07:49 2018

@author: dono_
"""
import cv2
from skimage.feature import local_binary_pattern
import numpy as np

def get_roi_resizing (img, contour, roi_size=256):
    '''
    Returns region of interest (roi) from a contour by generating a bounding
    box and resizing it to roi_size

    Args:
        img (np.array): image where contour is defined.
        contour ([(int, int)]): list of point that from a contour.
        roi_size (int): dimension of roi  
        

    Returns:
        roi(np.array): roi image
    '''
    
    # generate bounding rectangle of the contour
    x,y,w,h = cv2.boundingRect(contour)
    rect = img[y:int(y+h), x:int(x+w)]
    # generate roi
    dim = (roi_size,roi_size)
    roi = cv2.resize(rect, dim, interpolation = cv2.INTER_AREA)
    
    return roi

def get_roi_cropping (img, contour, roi_size=256):
    '''
    Returns region of interest (roi) from a contour by calculating its centroid
    and generating a box of roi_size around it

    Args:
        img (np.array): image where contour is defined.
        contour ([(int, int)]): list of point that from a contour.
        roi_size (int): dimension of roi  
        

    Returns:
        roi(np.array): roi image
    '''
    roi_radius = int(roi_size/2)
    # find centroid
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    # generate roi
    pt1 = (int(cx-roi_radius), int(cy-roi_radius))
    pt2 = (int(cx+roi_radius), int(cy+roi_radius))
    roi = img[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    
    return roi

def get_LBP (roi, P=8, R=1, METHOD='uniform', block_size=5):
    '''
    Computation of rotation invatiant local binary patterns for a region of 
    interest. 

    Args:
        roi (np.array): roi image
        P (int): number of neighbors 
        R (int): radius
        Method (int): lbp method
        block_size (int): window size for splitting the image 

    Returns:
        LBP_features: directory containig the concatenated histogram
    '''
    
    #hist = []
    #labels = []
    LBP_features = dict()
    counter = 0
    n_bins = int(P+R+1)
    for r in range(0, roi.shape[0] -block_size , block_size):
        for c in range(0, roi.shape[1] -block_size, block_size):
            window = roi[r:r+block_size,c:c+block_size]
            lbp = local_binary_pattern(window, P, R, METHOD)
            hist_current, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))  
            #hist = np.concatenate([hist,hist_current])
            for n in range(0, hist_current.shape[0]):
                label = "LBP"+str(counter)
                LBP_features[label] = hist_current[n]
                counter = counter + 1
     
    return LBP_features