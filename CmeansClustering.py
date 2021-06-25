import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import operator

import cv2
from cv2 import selectROI

def euclideanDistance(pixel1, pixel2):
    distance = np.sqrt(np.sum((pixel1 - pixel2)**2))
    return distance

def euclideanDistanceImage(pixel, img):
    distance_img = np.sqrt(np.sum((img - pixel)**2, axis = 2))
    return distance_img

def cmeanCluster(img, c):
    # Initialize clusters by random means
    means = np.random.randint(low=0, high = 255, size = (c,3))
    new_means = means

    while True:
        clustered_img = np.zeros_like(img)
        d_to_means = []
        for i in range(c):
            d_to_means.append(euclideanDistanceImage(pixel = new_means[i], img=img))

        dmin = np.min(d_to_means, axis = 0)
        wheremin = np.equal(d_to_means, dmin)

        for i in range(len(wheremin)):
            new_means[i] = np.mean(img[wheremin[i]], axis = 0)
            clustered_img[wheremin[i]] = new_means[i]
        
        if np.linalg.norm(new_means - means) == 0:
            break                              # Finished mean clustering

        print("did one loop")
    
    return clustered_img

def main():
    # Load the first image
    image = Image.open('TestingImages/159029.jpg')
    data_1 = np.asarray(image)

    classified_1 = cmeanCluster(data_1, c=3)
    classified_1 = cv2.cvtColor(classified_1, cv2.COLOR_RGB2BGR)
    cv2.imshow("Segmented Image", classified_1)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()