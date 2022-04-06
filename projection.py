import cv2
import numpy as np 

if __name__ == "__main__":

    # read the image 
    image_bgr = cv2.imread("images/sd_scene.png")
    
    # the intrinsic parameters of the camera that took this shot
    K = [9.037596e+02, 0.000000e+00, 6.957519e+02, 0.000000e+00, 9.019653e+02, 2.242509e+02, 0.000000e+00, 0.000000e+00, 1.000000e+00]
    K = np.reshape(K, (3,3))    # we reshape to make it a 2D numpy matrix
    print("--------------------------------------------------------")
    print("Intrinsic parameters [K] of the camera that took this shot")
    print(K)
    print("--------------------------------------------------------")

    # we'll define here that the world frame is itself the camera frame 
    # hence identity rotation and zero translation in the extrinsic matrix [R|T]
    rotation = np.eye(3)
    translation = np.zeros((3,1))
    extrinsic = np.hstack((rotation, translation))
    print("--------------------------------------------------------")
    print("Extrinsic parameters")
    print(extrinsic)
    print("--------------------------------------------------------")
    
    # Defining the world point that we are trying to understand where it will fall inside the image
    world_point = np.array([[0.25],
                            [0.25],
                            [3],
                            [1]])
    print("--------------------------------------------------------")
    print("World point we're trying to project inside the image")
    print(world_point)
    print("--------------------------------------------------------")


    # Projecting the point on the image
    pixel_point = K @ extrinsic @ world_point
    u_, v_, z= pixel_point[0,0], pixel_point[1,0], pixel_point[2,0]
    u = int(u_/z)
    v = int(v_/z)
    print("--------------------------------------------------------")
    print("Coordinates of that point projected on the image plane")
    print("u:{}, v:{}".format(u,v))
    print("--------------------------------------------------------")
    

    # Draw a circle around the pixel with green color
    radius = 3 
    thickness = 3
    cv2.circle(image_bgr, (u,v), radius, (0,255,0), thickness)
    cv2.imshow("frame", image_bgr)
    cv2.waitKey(0)