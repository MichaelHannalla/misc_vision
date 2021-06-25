from PIL import Image
import numpy as np
from sklearn import svm

import cv2
from cv2 import selectROI

def euclideanDistance(pixel, img):
    img_mean = np.mean(img, axis=(0,1))
    distance_img = np.sqrt(np.sum((img_mean - pixel)**2))
    return distance_img

def fitSVM(img, objbox, backbox):                                               # selectROI return (x, y, w, h)

    segmented_img = np.zeros_like(img)

    obj_cropped = img[objbox[1] : objbox[1]+objbox[3], objbox[0] : objbox[0]+objbox[2], :]
    bg_cropped = img[backbox[1] : backbox[1]+backbox[3], backbox[0] : backbox[0]+backbox[2], :]
    obj_flattened = obj_cropped.reshape(-1, 3)
    bg_flattened = bg_cropped.reshape(-1, 3)

    obj_mean = np.mean(obj_cropped, axis=(0,1))
    bg_mean = np.mean(bg_cropped, axis=(0,1))

    train_x = np.vstack((obj_flattened, bg_flattened))
    labels_y = np.vstack((np.ones((len(obj_flattened),1)), np.zeros((len(bg_flattened),1)))).reshape(-1)

    model = svm.SVC()
    model.fit(train_x, labels_y)
    
    img_flattened = img.reshape(-1, 3)
    predicted = model.predict(img_flattened).reshape(img.shape[0], img.shape[1])

    for i in range(predicted.shape[0]):
        for j in range(predicted.shape[1]):
            if predicted[i, j] == 1:
                segmented_img[i, j] = obj_mean
            else:
                segmented_img[i, j] = bg_mean
    
    return segmented_img

def main():
    # Load the first image
    image = Image.open('TestingImages/67079.jpg')
    data_1 = np.asarray(image)
    data_1 = cv2.cvtColor(data_1, cv2.COLOR_RGB2BGR)

    background_bbox = selectROI("Select background", data_1)
    object_bbox = selectROI("Select object", data_1)
    cv2.destroyAllWindows()

    classified_1 = fitSVM(data_1, object_bbox, background_bbox)
    
    cv2.imshow("Segmented Image", classified_1)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()