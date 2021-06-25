import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import operator

import cv2
from cv2 import selectROI

def euclideanDistance(pixel, img):
    #img_filtered = cv2.medianBlur(img, 3)
    img_mean = np.mean(img, axis=(0,1))
    #distance_img = np.sqrt(np.sum((img_filtered - pixel)**2, axis = 2))
    distance_img = np.sqrt(np.sum((img_mean - pixel)**2))
    return distance_img

def getNeighbors(img, objbox, backbox):                                          # selectROI return (x, y, w, h)
    segmented_img = np.zeros_like(img)
    obj_cropped = img[objbox[1] : objbox[1]+objbox[3], objbox[0] : objbox[0]+objbox[2], :]
    bg_cropped = img[backbox[1] : backbox[1]+backbox[3], backbox[0] : backbox[0]+backbox[2], :]
    obj_mean = np.mean(obj_cropped, axis=(0,1))
    bg_mean = np.mean(bg_cropped, axis=(0,1))
    
    for u in range(img.shape[0]):
        for v in range(img.shape[1]):

            dist_to_object = euclideanDistance(img[u, v], obj_cropped)
            min_to_obj = np.amin(dist_to_object)
            
            dist_to_bg = euclideanDistance(img[u, v], bg_cropped)
            min_to_bg = np.amin(dist_to_bg)

            if min_to_obj < min_to_bg:                                          # Pixel classified as object
                segmented_img[u, v] = obj_mean
            else:
                segmented_img[u, v] = bg_mean

    return segmented_img

def main():
    # Load the first image
    image = Image.open('TestingImages/3096.jpg')
    data_1 = np.asarray(image)
    data_1 = cv2.cvtColor(data_1, cv2.COLOR_RGB2BGR)

    background_bbox = selectROI("Select background", data_1)
    object_bbox = selectROI("Select object", data_1)
    cv2.destroyAllWindows()

    classified_1 = getNeighbors(data_1, object_bbox, background_bbox)
    cv2.imshow("Segmented Image", classified_1)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()