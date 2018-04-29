import cv2
import numpy as np
from math import pi, sqrt

def get_itensity_features(img, contour):
    '''
    Returns intensity features of the given contour.
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
    mass_candidate = mass_candidate[y:y+height, x:x+width]

    if (img.dtype == np.uint8):
        hist_size = 2**8
    elif (img.dtype == np.uint16):
        hist_size = 2**16

    bins = range(0, hist_size+1)    
    hist, _ = np.histogram(mass_candidate, bins=bins)
    # cropping of the 0s from histogram, because they are mask leftovers
    hist = hist[1:-1]
    # fixing length of bins due to numpy specifics of histogram calculation
    bins = bins[1:-2]
    # histogram normalization
    hist = hist * (1/sum(hist))

    mean_intens = get_mean_intensity(hist, bins)
    std_dev = get_standard_deviation(hist, bins, mean_intens)
    skewness = get_skewness(hist, bins, mean_intens)

    smoothness = 1 - (1 / (1 + std_dev*std_dev))

    return {"mean_intensity": mean_intens, 
            "standard_deviation": std_dev, 
            "smoothness": smoothness,
            "skewness": skewness}


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


def get_skewness(histogram, bins, mean_intensity=None):
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

    for index, gray_level in enumerate(bins):
        skewness = skewness + ((gray_level - mean_intensity)**3) * histogram[index]

    return skewness
