import cv2
import matplotlib.pyplot as plt

def show_image_cv(img, window_name):
    '''
        Shows images using cv2 imshow in smaller windows. Close by pressing any key
        
        Args:
        img: image file.
        window_name (string): name of window to be displayed
        
        Returns:
        0
        '''
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.resizeWindow(window_name, 332, 408)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def show_image_plt(img, image_name):
    '''
        Shows images using matplotlib plt.
        
        Args:
        img: image file.
        image_name (string): title of image to be displayed
        
        Returns:
        0
        '''
    plt.imshow(img, cmap="gray")
    plt.axis('off')
    plt.title(image_name)
    plt.show()

