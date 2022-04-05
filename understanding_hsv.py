import cv2
import numpy as np

if __name__ == "__main__":

    # create an image
    img_hsv = np.zeros((512, 512, 3)).astype(np.float32)

    # The equivalent of pure red in RGB space in HSV
    # is 0 hue, 255 saturation, 255 value
    img_hsv[:,:,0] = 0 # notice the broadcasting
    img_hsv[:,:,1] = 255
    img_hsv[:,:,2] = 255
    # TRY OTHER COLORS

    # convert to visible BGR
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow("window", img_bgr)
    cv2.waitKey(0)