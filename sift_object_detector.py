import cv2
import numpy as np

if __name__ == "__main__":

    # Read the target image
    target_object_path = "object.png"
    target_object_image_bgr = cv2.imread(target_object_path)
    target_object_image_bgr = cv2.resize(target_object_image_bgr, 
        (target_object_image_bgr.shape[1], target_object_image_bgr.shape[0]))
    target_object_image_gray = cv2.cvtColor(target_object_image_bgr, cv2.COLOR_BGR2GRAY)

    # Read the lookup image
    lookup_image_path = "lookup.png"
    lookup_image_bgr = cv2.imread(lookup_image_path)
    lookup_image_bgr = cv2.resize(lookup_image_bgr, 
        (lookup_image_bgr.shape[1], lookup_image_bgr.shape[0]))
    lookup_image_gray = cv2.cvtColor(lookup_image_bgr, cv2.COLOR_BGR2GRAY)

    # Create the SIFT feature extractor
    sift = cv2.ORB_create()

    # Detect SIFT features
    target_object_image_kp, target_object_image_des = sift.detectAndCompute(target_object_image_gray, None)
    lookup_image_kp, lookup_image_des = sift.detectAndCompute(lookup_image_gray, None)
    target_object_image_keypoints = cv2.drawKeypoints(target_object_image_gray, target_object_image_kp, target_object_image_bgr)
    lookup_image_keypoints = cv2.drawKeypoints(lookup_image_gray, lookup_image_kp, lookup_image_bgr)
    
    # Initialize the Matcher for matching the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(lookup_image_des, target_object_image_des, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # Draw the matches
    matches_img = cv2.drawMatchesKnn(lookup_image_bgr, lookup_image_kp,
        target_object_image_bgr, target_object_image_kp, good, None)

    cv2.imshow("target object", target_object_image_keypoints)
    cv2.imshow("lookup image", lookup_image_keypoints)
    cv2.imshow("matches image", matches_img)
    cv2.waitKey(0)