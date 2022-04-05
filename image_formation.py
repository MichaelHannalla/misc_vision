import cv2

if __name__ == "__main__":

    img_path = 'images/image_formation_example.png'
    image_bgr = cv2.imread(img_path) # default format of opencv is BGR not RGB

    print("----------------------------------------------------")
    print("Type of image_bgr: {}".format(type(image_bgr)))
    print("Dimensions of the image_bgr: ".format(image_bgr.ndim))
    print("Total shape of image_bgr: {}".format(image_bgr.shape))
    print("Image height: {}".format(image_bgr.shape[0]))
    print("Image width: {}".format(image_bgr.shape[1]))
    print("Image channels: {}".format(image_bgr.shape[2]))
    print("Size of the image_bgr array: {}".format(image_bgr.size))
    print("----------------------------------------------------")

    # converting the image to grayscale
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    print("Type of image_gray: {}".format(type(image_gray)))
    print("Dimensions of the image_gray: ".format(image_gray.ndim))
    print("Total shape of image_gray: {}".format(image_gray.shape))
    print("----------------------------------------------------")    

    # To show the image_bgr and image_gray
    cv2.imshow("<Window name in here> 1", image_bgr)     
    cv2.imshow("<Window name in here> 2", image_gray)     

    cv2.waitKey(0)          # Waits for any key to bypass this