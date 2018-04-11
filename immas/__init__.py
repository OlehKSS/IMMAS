import cv2
from mammogram import MammogramImage

path_image = "/home/okozyn/Projects/AIA-2018/dataset/images/20587080_b6a4f750c6df4f90_MG_R_ML_ANON.tif"
path_mask = "/home/okozyn/Projects/AIA-2018/dataset/masks/20587080_b6a4f750c6df4f90_MG_R_ML_ANON.mask.png"
pectoral_muscle = "/home/okozyn/Projects/AIA-2018/dataset/pectoral_muscle_masks/20587080_b6a4f750c6df4f90_MG_R_ML_ANON.tif"
mm = MammogramImage(path_image, path_mask, pmuscle_mask_path=pectoral_muscle)

cv2.namedWindow("pectoral", cv2.WINDOW_NORMAL)
cv2.imshow("pectoral", mm.image_data)
cv2.waitKey(0)

def test():
    print("Hello World!")