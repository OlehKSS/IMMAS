import cv2
from math import pi
from sys import float_info

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

    if area != 0:
        circularity = (perimeter*perimeter) / (4*pi*area)
    else:
        circularity = float_info.max
    # ac ratio
    ac = area / circularity

    return {"perimeter": perimeter, 
            "area": area, 
            "circularity": 
            circularity, 
            "ac": ac}
            