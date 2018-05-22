import cv2, numpy
from math import pi, sqrt
from sys import float_info
import scipy

def get_geom_features(contour):
    '''
    Returns geometric features of the given contour.
    Available features now are perimeter, area, circularity and ac ratio.

    Args:
        contour ([(int, int)]): list of point that from a contour.

    Returns:
        dict: dictionary with feature names and values.
    '''

    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)


    # Coordinates of the centroid of the image
    moments = cv2.moments(contour)
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    # Find normalized radial length
    radial_length = numpy.zeros((len(contour), 1))
    for i in range(0,len(contour)):
        radial_length[i] = sqrt((cx - contour[:, 0][i][0]) ** 2 + (cy - contour[:, 0][i][1]) ** 2)
    max_radial_length = numpy.amax(radial_length)
    normalized_radial_length = radial_length / max_radial_length
    mean_RL = numpy.mean(radial_length)
    mean_NRL = numpy.mean(normalized_radial_length)
    SD_NRL = numpy.std(normalized_radial_length)
    ratio_SD_NRL_and_mean_RL = SD_NRL/mean_RL
    entropy_NRL = scipy.stats.entropy(normalized_radial_length)


    shape_factor = 0
    if area != 0:
        circularity = (perimeter*perimeter) / (4*pi*area)
        shape_factor = (perimeter**2)/area
    else:
        circularity = float_info.max
    # ac ratio
    ac = area / circularity

    return {"perimeter": perimeter, 
            "area": area, 
            "circularity": circularity,
            "ac": ac,
            'shape_factor': shape_factor,
            "mean NRL": mean_NRL,
            "SD NRL": SD_NRL,
            "Ratio SD NRL and mean RL": ratio_SD_NRL_and_mean_RL,
            "entropy NRL": entropy_NRL}