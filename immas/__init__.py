import cv2
from immas.io import read_dataset

# I will probably need original uncropped image for later comparison with ground truth, or masks
data_set = read_dataset(image_folder="/home/okozyn/Projects/AIA-2018/dataset/images",
            mask_folder="/home/okozyn/Projects/AIA-2018/dataset/masks",
            results_folder="/home/okozyn/Projects/AIA-2018/dataset/groundtruth",
            pmuscle_mask_folder="/home/okozyn/Projects/AIA-2018/dataset/pectoral_muscle_masks")

print(len(data_set))

img = data_set[4]
img.read_data()

print(img.uncropped_image.shape)

# cv2.namedWindow("data", cv2.WINDOW_NORMAL)
# cv2.imshow("data", img.image_data)
# cv2.waitKey(0)

cv2.namedWindow("uncropped data", cv2.WINDOW_NORMAL)
cv2.imshow("uncropped data", img.uncropped_image)
cv2.waitKey(0)

def test():
    print("Hello World!")