import cv2
import rospy
import numpy as np
import cv_bridge
from sensor_msgs.msg import Image

def nothing(x):
    pass # A function that does nothing

if __name__ == "__main__":

    vid_path = "videos/color_thresholding_demo.mp4"
    cap = cv2.VideoCapture(vid_path)

    cv2.namedWindow('Trackbars')
    # create trackbars for color change
    # This is only used for finding the value
    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    while cap.isOpened():
    
        ret, frame = cap.read()

        if ret:

            image_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Get the trackbar locations
            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")
            
            # Represent the upper and lower bounds as arrays because that's what OpenCV expects
            lower_ = np.array([l_h, l_s, l_v])
            upper_ = np.array([u_h, u_s, u_v])
            mask = cv2.inRange(image_hsv, lower_, upper_)       # Perform the thresholding

            res_m = cv2.bitwise_and(frame, frame, mask= mask)   

            # We don't need to visualize contours here, uncomment if wanted.
            contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image=res_m, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)   

            cv2.imshow('frame', frame)                       # show the original frame
            cv2.imshow('result thresholded', res_m)          # show the resulting thresholded frame
            
            # Check if the user has pressed ESC key
            c = cv2.waitKey(1)
            if c == 27:
                break   # exit if ESC is pressed
