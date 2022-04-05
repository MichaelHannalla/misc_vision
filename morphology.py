import cv2
import numpy as np

if __name__ == "__main__":

    img_path = 'images/circles.png'

    # default format of opencv is BGR not RGB
    image_bgr = cv2.imread(img_path) 

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((7,7),np.uint8)       # notice the effect of changing kernel size

    image_gray_dilated = cv2.dilate(image_gray, kernel) 
    image_gray_eroded = cv2.erode(image_gray, kernel)
    
    cv2.imshow("original image", image_gray)
    cv2.imshow("dilated image", image_gray_dilated)
    cv2.imshow("eroded image", image_gray_eroded)

    cv2.waitKey(0)                          # press any key to exit
