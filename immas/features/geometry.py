import cv2
from math import pi, sqrt
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

    moments = cv2.moments(contour)
    # Coordinates of the centroid of the image
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    for i in len(contour)
        radial_length = sqrt((cx - contour[i][1])**2 + (cy - contour[i][2])**2)



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
            "circularity": 
            circularity, 
            "ac": ac,
            'shape_factor': shape_factor}
            