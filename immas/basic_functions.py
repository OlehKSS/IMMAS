import cv2

def show_image(img, window_name):
    '''
        Shows images using cv2 imshow in smaller windows. Close by pressing any key
        
        Args:
        img (uint8): image file.
        window_name (string): name of window to be displayed
        
        Returns:
        0
        '''
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 332, 408)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    return 0
