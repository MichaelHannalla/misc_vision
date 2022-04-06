import os
import cv2
import numpy as np 

if __name__ == "__main__":
    
    # Chessboard dimensions and calibration data
    checkerboard_shape = (6, 9)       # Number of corners in checkerboard
    square_size_mm = 30
    mm = 1e-3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, square_size_mm, mm)    # Criteria
    objpoints = []                    # list that has the object (world coordinates) points
    imgpoints = []                    # list that has the pixel locations of these object points
    
    # These two lines to get a combination of all coordinates
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # Paying attention to the next two lines is not necessary!
    objp = np.zeros((1, checkerboard_shape[0] * checkerboard_shape[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerboard_shape[0], 0:checkerboard_shape[1]].T.reshape(-1, 2)
    
    # The directory which contains the chessboard images
    calib_images_directory = "calib"
    calib_images_paths = os.listdir(calib_images_directory)
    
    # Read the images inside the directory
    for calib_image_path in calib_images_paths:
        
        calib_image_path_ = os.path.join(calib_images_directory, calib_image_path)
        calib_image = cv2.imread(calib_image_path_)
        calib_image_gray = cv2.cvtColor(calib_image, cv2.COLOR_BGR2GRAY)

        # Get the chessboard corners from the current image
        ret, corners = cv2.findChessboardCorners(calib_image_gray, checkerboard_shape, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:

            # add the world points to the list
            objpoints.append(objp)

            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(calib_image_gray, corners, (11,11),(-1,-1), criteria)
            
            # add the refined image points to the list
            imgpoints.append(corners2)

            # # Draw and display the corners
            chessboard_calib_image = cv2.drawChessboardCorners(calib_image, checkerboard_shape, corners2, ret)
            cv2.imshow("calibration process", chessboard_calib_image)
            cv2.waitKey(2) # wait for two-seconds

    # After we got the objpoints and imgpoints
    # We need to give these 2 lists that has (img-obj) pairs to the solver to get the calibration
    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    image_shape = calib_image_gray.shape[::-1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    print("Camera matrix (intrinsics) :")
    print(mtx)
    print("\nDistortion cooeficients :")
    print(dist)