import cv2
import numpy as np

def visualize(corners, ids, rvecs, tvecs, rejected):
    
    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):

            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID),
                (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[INFO] ArUco marker ID: {}".format(markerID))
        
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(image, camera_mtx, dist_coeffs, rvec, tvec, 0.1)

        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        
if __name__ == "__main__":
    
    # Read the image and specify the family of AruCo markers to be detected
    image_path = "images/aruco_image.png"
    aruco_type = cv2.aruco.DICT_6X6_250
    image = cv2.imread(image_path)

    # Initialize OpenCV AruCo detector and do the detection
    aruco_dict = cv2.aruco.Dictionary_get(aruco_type)
    aruco_params = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, aruco_dict,
        parameters=aruco_params) 

    # Specify the intrinsic matrix of the camera taking the images
    # Warning: These are dummy numbers
    camera_mtx = np.array(
        [[503.68477354,          0,              313.67563672],
         [  0,                   503.37989265,   243.25575731],
         [  0,                   0,                         1]]
    )

    # Specify the intrinsic matrix of the camera taking the images
    dist_coeffs = np.array(
        [[ 2.08346330e-01, -4.68650248e-01,  4.51081812e-04, -1.93373844e-03, 2.37592300e-01]]
    )

    # Estimate relative position of the aruco-marker with respect to the camera frame
    marker_length_m = 0.25
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length_m, camera_mtx, dist_coeffs)
    print("---------------------------------------------------")
    print("Rotational vectors to markers (roll pitch yaw):")
    print(rvecs)
    print("---------------------------------------------------")
    print("Translational vectors to markers (x y z):")
    print(tvecs)
    print("---------------------------------------------------")
        
    # Visualize
    visualize(corners, ids, rvecs, tvecs, rejected)

# References: 
# https://pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
# https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html