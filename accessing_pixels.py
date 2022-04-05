import cv2

if __name__ == "__main__":

    img_path = 'images/image_formation_example.png'
    # default format of opencv is BGR not RGB
    image_bgr = cv2.imread(img_path) 

    # Acces the pixel at row 0, column 0
    pixel = image_bgr[0,0]      
    (b, g, r) = pixel
    print("Blue, Green and Red values at (0,0): ", format((b, g, r)))