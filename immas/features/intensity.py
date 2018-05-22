import cv2
import numpy as np
from math import pi, sqrt
from skimage import feature
import scipy


def get_itensity_features(img, contour):
    '''
    Returns intensity features of the given contour (except NCPS -> geometric).
    Available features now are mean intensity, standard deviation,
    smoothness and skewness.

    Args:
        img (np.array): image where contour is defined.
        contour ([(int, int)]): list of point that from a contour.

    Returns:
        dict: dictionary with feature names and values.
    '''

    x, y, width, height = cv2.boundingRect(contour)
    mask = np.zeros(img.shape, dtype='uint8')
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    mass_candidate = img * mask
    # cropped mass from the image, without background
    mass_candidate = mass_candidate[y:y + height, x:x + width]

    if (img.dtype == np.uint8):
        hist_size = 2 ** 8
    elif (img.dtype == np.uint16):
        hist_size = 2 ** 16

    bins = range(0, hist_size + 1)
    hist, _ = np.histogram(mass_candidate, bins=bins)
    # cropping of the 0s from histogram, because they are mask leftovers
    hist = hist[1:-1]
    # fixing length of bins due to numpy specifics of histogram calculation
    bins = bins[1:-2]
    # histogram normalization
    hist = hist * (1 / sum(hist))

    mean_intens = get_mean_intensity(hist, bins)
    std_dev = get_standard_deviation(hist, bins, mean_intens)
    skewness, kurtosis = get_skewness_kurtosis(hist, bins, mean_intens)

    smoothness = 1 - (1 / (1 + std_dev * std_dev))

    correlation, contrast, uniformity, homogeneity, energy, dissimilarity = get_GLCM_descriptors(mass_candidate)

    # NCPS Calculation
    # Coordinates of the centroid of the image
    moments = cv2.moments(contour)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    area = cv2.contourArea(contour)
    intensities = np.zeros((len(contour), 1))
    for i in range(0, len(contour)):
        x = contour[:, 0][i][0]
        y = contour[:, 0][i][1]
        intensities[i] = img[y, x]
    min_gray_coor = np.argmax(intensities)
    a = contour[:, 0][min_gray_coor][0]
    b = contour[:, 0][min_gray_coor][1]
    NCPS = sqrt((cx - a) ** 2 + (cy - b) ** 2) / area

    # Gradient calculation
    # Find normalized radial length
    radial_length = np.zeros((len(contour), 1))
    for i in range(0, len(contour)):
        radial_length[i] = sqrt((cx - contour[:, 0][i][0]) ** 2 + (cy - contour[:, 0][i][1]) ** 2)
    centroid_intensity = img[cy, cx]
    gradient = np.zeros((len(contour), 1))
    for i in range(0, len(contour)):
        y1 = contour[:, 0][i][1] - cy
        theta = np.arcsin(y1 / radial_length[i])
        if contour[:, 0][i][1] < cy and contour[:, 0][i][0] > cx:
            x2 = contour[:, 0][i][0] + abs(int(10 * np.cos(theta)))
            y2 = contour[:, 0][i][1] - abs(int(10 * np.sin(theta)))
        elif contour[:, 0][i][1] < cy and contour[:, 0][i][0] < cx:
            x2 = contour[:, 0][i][0] - abs(int(10 * np.cos(theta)))
            y2 = contour[:, 0][i][1] - abs(int(10 * np.sin(theta)))
        elif contour[:, 0][i][1] > cy and contour[:, 0][i][0] < cx:
            x2 = contour[:, 0][i][0] - abs(int(10 * np.cos(theta)))
            y2 = contour[:, 0][i][1] + abs(int(10 * np.sin(theta)))
        else:
            x2 = contour[:, 0][i][0] + abs(int(10 * np.cos(theta)))
            y2 = contour[:, 0][i][1] + abs(int(10 * np.sin(theta)))
        gradient[i] = centroid_intensity - img[y2, x2]
    gradient_mean = np.mean(gradient)
    gradient_SD = np.std(gradient)
    gradient_skewness = scipy.stats.skew(gradient)

    return {"NCPS": NCPS,
            "mean_intensity": mean_intens,
            "standard_deviation": std_dev,
            "smoothness": smoothness,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "correlation": correlation,
            "contrast": contrast,
            "uniformity": uniformity,
            "homogeneity": homogeneity,
            "energy": energy,
            "dissimilarity": dissimilarity,
            "gradient mean": gradient_mean,
            "gradient SD": gradient_SD,
            "gradient skewness": gradient_skewness}


def get_mean_intensity(histogram, bins):
    '''
    Returns mean intensity.

    Args:
        histogram ([int]): histogram as list of ints.
        bins ([int]): list (iterable) of the gray-levels for histogram.

    Returns:
        float: mean intensity.    
    '''

    mean_intensity = 0

    for index, gray_level in enumerate(bins):
        mean_intensity = mean_intensity + gray_level * histogram[index]

    return mean_intensity


def get_standard_deviation(histogram, bins, mean_intensity=None):
    '''
    Returns standard deviation of the image intensity.

    Args:
        histogram ([int]): histogram as list of ints.
        bins ([int]): list (iterable) of the gray-levels for histogram.
        mean_intensity (float): mean intensity

    Returns:
        float: standard deviation.    
    '''

    if not mean_intensity:
        mean_intensity = get_mean_intensity(histogram, bins)

    variance = 0

    for index, gray_level in enumerate(bins):
        variance = variance + (gray_level - mean_intensity) \
                   * (gray_level - mean_intensity) * histogram[index]

    return sqrt(variance)


def get_skewness_kurtosis(histogram, bins, mean_intensity=None):
    '''
    Returns skewness of the image.

    Args:
        histogram ([int]): histogram as list of ints.
        bins ([int]): list (iterable) of the gray-levels for histogram.
        mean_intensity (float): mean intensity

    Returns:
        float: skewness.    
    '''

    if not mean_intensity:
        mean_intensity = get_mean_intensity(histogram, bins)

    skewness = 0
    kurtosis = 0

    for index, gray_level in enumerate(bins):
        skewness = skewness + ((gray_level - mean_intensity) ** 3) * histogram[index]
        kurtosis = kurtosis + ((gray_level - mean_intensity) ** 4) * histogram[index]

    return skewness, kurtosis


def get_GLCM_descriptors(img):
    '''
    Returns statistical descriptors based on Grey Level Co-occurrence Matrix (GLCM)

    Args:
        img (np.array): image (mass candidate).

    Returns:
        float: correlation
        float: contrast
        float: uniformity
        float: homogeneity
        float: energy
        float: dissimilarity
    '''
    binned = (img / 1024).astype('uint8')
    GLCM = feature.greycomatrix(binned, [1], [0], levels=binned.max() + 1, normed=True)
    correlation = feature.greycoprops(GLCM, prop='correlation')
    contrast = feature.greycoprops(GLCM, prop='contrast')
    uniformity = feature.greycoprops(GLCM, prop='ASM')
    homogeneity = feature.greycoprops(GLCM, prop='homogeneity')
    energy = feature.greycoprops(GLCM, prop='energy')
    dissimilarity = feature.greycoprops(GLCM, prop='dissimilarity')

    return correlation, contrast, uniformity, homogeneity, energy, dissimilarity
