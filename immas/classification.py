import cv2
import numpy as np
import pandas as pd
from . import basic_functions
from .constants import DICE_INDEX_DEFAULT_THRESHOLD, CLASS_ID_POS, CLASS_ID_NEG
import bisect
from sklearn.metrics import auc, classification_report, confusion_matrix, accuracy_score, matthews_corrcoef, roc_curve, make_scorer, roc_auc_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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
    TP_value = np.amax(segmented_images) + np.amax(groundtruth_images)
    TP = (segData == TP_value).sum()  # found a true positive: segmentation result and groundtruth match(both are positive)
    segData_FP = 2. * segmented_images + groundtruth_images
    segData_FN = segmented_images + 2. * groundtruth_images
    FP = (segData_FP == 2 * np.amax(segmented_images)).sum() # found a false positive: segmentation result and groundtruth mismatch
    FN = (segData_FN == 2 * np.amax(groundtruth_images)).sum() # found a false negative: segmentation result and groundtruth mismatch
    return 2*TP/(2*TP+FP+FN)  # according to the definition of DICE similarity score


def find_match(m, visual_result = "no"):
    '''
        Determines if a mass candidate is a match or not.
        
        Args:
        m (Mammogram object): classified image with associated regions
        visual_result (string): yes/no to display accuracy for candidates and groundtruth images
        
        Returns:
        num_TP (int): number of true positives regions in one image
        num_FP (int): number of false positives regions in one image
        num_mass_grd (int): number of masses in the groundtruth
        '''
    _, groundtruth_contours, _ = cv2.findContours(m.cropped_ground_truth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_mass_grd = len(groundtruth_contours)
    num_TP = 0
    num_FP = 0
    for r in m.regions:
        if r["class_id"] == CLASS_ID_POS:
            segmented_mask = np.zeros(m.image_data.shape, dtype='uint8')
            cv2.drawContours(segmented_mask, [r["countour"]], -1, 255, thickness=cv2.FILLED)
            DICE = np.zeros(len(groundtruth_contours))
            for j in range(0, len(groundtruth_contours)):
                groundtruth_mask = np.zeros(img.image_data.shape, dtype='uint8')
                cv2.drawContours(groundtruth_mask, [groundtruth_contours[j]], -1, 255, thickness=cv2.FILLED)
                DICE[j] = dice_similarity(segmented_mask,groundtruth_mask)
            if np.amax(DICE) >= DICE_INDEX_DEFAULT_THRESHOLD:
                num_TP = num_TP + 1
            else:
                num_FP = num_FP + 1

    if visual_result == "yes":
        segmented_mask = np.zeros(m.image_data.shape, dtype='uint8')
        for r in m.regions:
            if class_id == CLASS_ID_POS:
                cv2.drawContours(segmented_mask, [r["countour"]], -1, 255, thickness=cv2.FILLED)
        basic_functions.accuracy(segmented_mask,m.cropped_ground_truth,"yes")

    return num_TP, num_FP, num_mass_grd


def get_rois(mask, mask_ground_truth=None, dice_threshold=DICE_INDEX_DEFAULT_THRESHOLD):
    '''
    Function for obtaining regions of interest from the mammogram.
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
            groundtruth_mask = np.zeros(mask.shape, dtype='uint8')
            cv2.drawContours(groundtruth_mask, 
                            [contour_gt], 
                            -1, 
                            255, 
                            thickness=cv2.FILLED)

            gt_masks.append(groundtruth_mask)
        
        for seg_contour in segmented_contours:
            segmented_mask = np.zeros(mask.shape, dtype='uint8')
            cv2.drawContours(segmented_mask, [seg_contour], -1, 255, thickness=cv2.FILLED)

            dice_indices = np.zeros(len(gt_masks))

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

def load_features_data (file_path):
    '''
    Loads features dataframe given a file_path and returns two dataframes containing features randomly
    divided, with all regions of the same image in the same dataset
        
    Args:
        file_path: string containing path + filename

    Returns:
        data_train, data_test: pandas dataframes
    '''
    
    data = pd.read_csv(file_path, index_col=0)
    data_per_imgfile = list(data['img_name'].unique())
    np.random.shuffle(data_per_imgfile)

    test_set_perc = 0.5
    test_set_len = int(test_set_perc * len(data_per_imgfile))
    print(f"Number of images in the dataset 01: {test_set_len}")

    train_imgs = data_per_imgfile[test_set_len:]
    test_imgs = data_per_imgfile[:test_set_len]
    print(f"Number of images in the dataset 02: {len(train_imgs)}")

    # sampling of the original data set according to images in train/test subsets
    data_train = data[data["img_name"].isin(train_imgs)]
    data_test = data[data["img_name"].isin(test_imgs)]
    print(f"Number of regions in dataset 01: {len(data_train)}")
    print(f"Number of regions in dataset 02: {len(data_test)}")

    return data_train, data_test

def line(x_coords, y_coords):
    """
    Given a pair of coordinates (x1,y2), (x2,y2), define the line equation. Note that this is the entire line vs. t
    the line segment.

    Parameters
    ----------
    x_coords: Numpy array of 2 points corresponding to x1,x2
    x_coords: Numpy array of 2 points corresponding to y1,y2

    Returns
    -------
    (Gradient, intercept) tuple pair
    """    
    if (x_coords.shape[0] < 2) or (y_coords.shape[0] < 2):
        raise ValueError('At least 2 points are needed to compute'
                         ' area under curve, but x.shape = %s' % p1.shape)
    if ((x_coords[0]-x_coords[1]) == 0):
        raise ValueError("gradient is infinity")
    gradient = (y_coords[0]-y_coords[1])/(x_coords[0]-x_coords[1])
    intercept = y_coords[0] - gradient*1.0*x_coords[0]
    return (gradient, intercept)

def x_val_line_intercept(gradient, intercept, x_val):
    """
    Given a x=X_val vertical line, what is the intersection point of that line with the 
    line defined by the gradient and intercept. Note: This can be further improved by using line
    segments.

    Parameters
    ----------
    gradient
    intercept

    Returns
    -------
    (x_val, y) corresponding to the intercepted point. Note that this will always return a result.
    There is no check for whether the x_val is within the bounds of the line segment.
    """    
    y = gradient*x_val + intercept
    return (x_val, y)

def get_fpr_tpr_for_thresh(fpr, tpr, thresh):
    """
    Derive the partial ROC curve to the point based on the fpr threshold.

    Parameters
    ----------
    fpr: Numpy array of the sorted FPR points that represent the entirety of the ROC.
    tpr: Numpy array of the sorted TPR points that represent the entirety of the ROC.
    thresh: The threshold based on the FPR to extract the partial ROC based to that value of the threshold.

    Returns
    -------
    thresh_fpr: The FPR points that represent the partial ROC to the point of the fpr threshold.
    thresh_tpr: The TPR points that represent the partial ROC to the point of the fpr threshold
    """    
    p = bisect.bisect_left(fpr, thresh)
    thresh_fpr = fpr[:p+1].copy()
    thresh_tpr = tpr[:p+1].copy()
    g, i = line(fpr[p-1:p+1], tpr[p-1:p+1])
    new_point = x_val_line_intercept(g, i, thresh)
    thresh_fpr[p] = new_point[0]
    thresh_tpr[p] = new_point[1]
    return thresh_fpr, thresh_tpr

def partial_auc_score(fpr, tpr, upper_limit=1):
    """
    Derive the AUC based of the partial FROC curve from FPPI=0 to FPR=upper_limit threshold.

    Parameters
    ----------
    fppi: numpy array of false positive per image.
    tpr: numpy array of true positive rate.
    upper_limit: The threshold based on the FPPI to extract the partial FROC based to that value of the threshold.

    Returns
    -------
    AUC of the partial ROC. A value that ranges from 0 to 1.
    """        
    fpr_thresh, tpr_thresh = get_fpr_tpr_for_thresh(fpr, tpr, upper_limit)
    return auc(fpr_thresh, tpr_thresh)

def ROC_to_FROC(full_prob, false_positive_rate, true_positive_rate, full_auc):
    """
    Uses the output of the ROC curve to build the FROC curve, correctly scaling the x and y axis
    Calculates the area under the FROC curve for FPPI between 0 and 1

    Parameters
    ----------
    full_prob: numpy array of probabilities
    false_positive_rate: numpy array of false positive rate (output of ROC curve)
    true_positive_rate: numpy array of true positive rate (output of ROC curve)
    full_auc: float; total area under the ROC curve

    Returns
    -------
    partial_AUC: float; partial area for FPPI between 0 and 1
    false_positive_rate: numpy array of false positive per image corrected for FROC curve
    true_positive_rate: numpy array of true positive rate corrected for FROC curve
    """
    # Counts to adjust the TPR and to create the False Positive per Image
    unique, counts = np.unique(full_prob[:,-1], return_counts=True)
    num_img = 410
    num_pos_img = 115
    regions = full_prob[:,-1].shape[0]
    pos_reg = counts[1]
    neg_reg = counts[0]
    neg_reg_per_img = neg_reg / num_img
    false_positive_rate = false_positive_rate * neg_reg_per_img
    true_positive_rate = true_positive_rate * pos_reg / num_pos_img

    # Calculates Partial AUC
    partial_AUC = partial_auc_score(false_positive_rate, true_positive_rate, 1)

    # Plots the FROC Curve
    plt.title('Free Response ROC Curve')
    plt.plot(false_positive_rate, true_positive_rate, 'b',label='Partial AUC (FPPI = 0:1) = %0.2f'% partial_AUC)
    plt.legend(loc='lower right')
    plt.xlim([-0,5])
    plt.ylim([-0,1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive per Image (FPPI)')
    plt.grid(color='k', linestyle='dotted', linewidth=0.5, alpha=0.5)
    plt.show()

    print('Area under the original ROC curve for our classifier: %0.2f'% full_auc)
    print('Partial area under the FROC curve for FPPI between 0 and 1: %0.5f'% partial_AUC)
    return partial_AUC, false_positive_rate, true_positive_rate

def all_feat_no_LBP(kernel):
    if kernel == 'rbf':
        svclassifier = SVC(C=10, class_weight={1: 20}, gamma=0.0001, kernel='rbf', probability=True)
    elif kernel =='sigmoid':
        svclassifier = SVC(C=7, class_weight={1:15}, gamma=0.001, kernel='sigmoid', probability=True)
    elif kernel == 'linear':
        svclassifier = SVC(C=0.001, class_weight={1:20}, kernel='linear', probability=True)
    else:
        svclassifier = SVC(C=0.5, class_weight={1: 10}, gamma=0.01, kernel='poly', degree=1, coef0=1.0, probability=True)
    return svclassifier

def all_LBP(kernel):
    if kernel == 'rbf':
        svclassifier = SVC(C=0.01, class_weight={1: 10}, gamma=0.001, kernel='rbf', probability=True)
    elif kernel =='sigmoid':
        svclassifier = SVC(C=0.1, class_weight={1:10}, gamma=0.0001, kernel='sigmoid', probability=True)
    elif kernel == 'linear':
        svclassifier = SVC(C=0.001, class_weight='balanced', kernel='linear', probability=True)
    else:
        svclassifier = SVC(C=0.001, class_weight={1: 10}, gamma=0.001, kernel='poly', degree=3, coef0=0.5, probability=True)
    return svclassifier

def geom_feat(kernel):
    if kernel == 'rbf':
        svclassifier = SVC(C=10, class_weight={1: 20}, gamma=0.0001, kernel='rbf', probability=True)
    elif kernel =='sigmoid':
        svclassifier = SVC(C=7, class_weight={1:15}, gamma=0.001, kernel='sigmoid', probability=True)
    elif kernel == 'linear':
        svclassifier = SVC(C=0.001, class_weight={1:20}, kernel='linear', probability=True)
    else:
        svclassifier = SVC(C=0.5, class_weight={1: 10}, gamma=0.01, kernel='poly', degree=1, coef0=1.0, probability=True)
    return svclassifier

def intens_feat(kernel):
    if kernel == 'rbf':
        svclassifier = SVC(C=10, class_weight={1: 20}, gamma=0.0001, kernel='rbf', probability=True)
    elif kernel =='sigmoid':
        svclassifier = SVC(C=7, class_weight={1:15}, gamma=0.001, kernel='sigmoid', probability=True)
    elif kernel == 'linear':
        svclassifier = SVC(C=0.001, class_weight={1:20}, kernel='linear', probability=True)
    else:
        svclassifier = SVC(C=0.5, class_weight={1: 10}, gamma=0.01, kernel='poly', degree=1, coef0=1.0, probability=True)
    return svclassifier

def noGLCM_feat(kernel):
    if kernel == 'rbf':
        svclassifier = SVC(C=10, class_weight={1: 20}, gamma=0.0001, kernel='rbf', probability=True)
    elif kernel =='sigmoid':
        svclassifier = SVC(C=7, class_weight={1:15}, gamma=0.001, kernel='sigmoid', probability=True)
    elif kernel == 'linear':
        svclassifier = SVC(C=0.001, class_weight={1:20}, kernel='linear', probability=True)
    else:
        svclassifier = SVC(C=0.5, class_weight={1: 10}, gamma=0.01, kernel='poly', degree=1, coef0=1.0, probability=True)
    return svclassifier

def lbp_feat(kernel):
    if kernel == 'rbf':
        svclassifier = SVC(C=10, class_weight={1: 20}, gamma=0.0001, kernel='rbf', probability=True)
    elif kernel =='sigmoid':
        svclassifier = SVC(C=7, class_weight={1:15}, gamma=0.001, kernel='sigmoid', probability=True)
    elif kernel == 'linear':
        svclassifier = SVC(C=0.001, class_weight={1:20}, kernel='linear', probability=True)
    else:
        svclassifier = SVC(C=0.5, class_weight={1: 10}, gamma=0.01, kernel='poly', degree=1, coef0=1.0, probability=True)
    return svclassifier

def run_SVM (dataset01, dataset02, kernel='rbf', features='all_except_LBP'):
    """
    Runs SVM using otpimal parameters according to the features used and the desired kernel
    Prints the FROC curve, the area under the ROC curve and the partial area under the FROC curve
    for FPPI between 0 and 1

    Parameters
    ----------
    dataset01, dataset01: numpy arrays containing features and labels for each dataset
    features: string indicating which features were used. Default: all_except_LBP
    kernel: desired kernel to use in the SVM. Default: rbf
    
    Returns
    -------
    full_probabilities: numpy array of probabilities
    full_auc: float representing the full area under the ROC curve
    partial_auc: float representing the partial area under the FROC curve for FPPI between 0 and 1
    FROC_fpr, FROC_tpr: false positive per image and true positive rate numpy arrays corrected for the FROC curve
    
    """     
    features_dic = {
        'all_except_LBP': all_feat_no_LBP,
        'all_with_LBP': all_LBP,
        'geometrical': geom_feat,
        'intensity': intens_feat,
        'intensity_no_GLCM': noGLCM_feat,
        'lbp': lbp_feat,
    }

    # Get the function from features dictionary
    func = features_dic.get(features)
    # Execute the function
    svclassifier = func(kernel)

    dataset01_data = dataset01[:,:-1]
    dataset01_labels = dataset01[:,-1]
    dataset02_data = dataset02[:,:-1]
    dataset02_labels = dataset02[:,-1]

    # Trains classifier in DataSet01 and tests in DataSet02
    svclassifier.fit(dataset01_data, dataset01_labels)
    prob1 = svclassifier.predict_proba(dataset02_data)
    prob1 = np.column_stack((prob1,dataset02_labels))

    # Trains classifier in DataSet02 and tests in DataSet01
    svclassifier.fit(dataset02_data, dataset02_labels)
    prob2 = svclassifier.predict_proba(dataset01_data)
    prob2 = np.column_stack((prob2,dataset01_labels))

    # Calculate the probabilities taking both tests into account
    full_probabilities = np.concatenate((prob1,prob2),axis=0)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(full_probabilities[:,-1], full_probabilities[:,1], pos_label=1, drop_intermediate=True)
    full_auc = auc(false_positive_rate, true_positive_rate)
    partial_auc, FROC_fpr, FROC_tpr = ROC_to_FROC(full_probabilities, false_positive_rate, true_positive_rate, full_auc)

    return full_probabilities, full_auc, partial_auc, FROC_fpr, FROC_tpr
