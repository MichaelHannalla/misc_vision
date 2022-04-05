import cv2
import numpy as np

if __name__ == "__main__":

    img_path = 'images/image_formation_example.png'
    image_bgr = cv2.imread(img_path) # default format of opencv is BGR not RGB

    # Create a random 512 x 512 image
    image_bgr = np.random.randint(low=0, high=100, 
        size=(512, 512,3)).astype(np.float32) 
    
    # imagine what does this do (check numpy integer indexing)
    image_bgr[:80, :40] = [0,0,0]

    # notice the change in coordinate representation convention
    cv2.rectangle(image_bgr, (0, 0), (80, 40), 
        color= (0, 255, 0), thickness=2) # this draws a rectangle

    cv2.imshow("window", image_bgr)
    cv2.waitKey(0)      # press any key to exit

