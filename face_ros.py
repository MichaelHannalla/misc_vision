import cv2
import rospy
from std_msgs.msg import Int32

if __name__ == "__main__":

    # Load the face detector model
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

    # Load the demo video
    vid_path = "videos/face_demo.mp4"
    cap = cv2.VideoCapture(vid_path)

    # Initialize ROS node
    rospy.init_node("face_detector_node")
    pub = rospy.Publisher("face_detector/status", Int32, queue_size=10)

    while cap.isOpened():    
        ret, frame = cap.read()

        # Reset each frame
        face_detected_in_this_frame = False

        if ret:

            image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       
            
            # scale_factor: Parameter specifying how much the image size is reduced at each image scale.
            scale_factor = 1.3 
            # min_neighbours: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
            min_neighbours = 5

            # Perform the faces detection
            faces = face_cascade.detectMultiScale(image_gray, scale_factor, min_neighbours)
            
            for (x,y,w,h) in faces:
                
                # Draw a rectangle at the detected location of the face
                cv2.rectangle(image_gray, (x,y), (x+w,y+h), (255, 255, 255) ,2)     

                # Set the bool face_detected_in_this_frame to True
                face_detected_in_this_frame = True

            # Publish the status of the detector
            if face_detected_in_this_frame:
                pub.publish(Int32(1))
            else:
                pub.publish(Int32(0))

            # Show the frame
            cv2.imshow('frame', image_gray)
            
            # Check if the user has pressed ESC key
            c = cv2.waitKey(1)

            if c == 27:
                cv2.destroyAllWindows()
                break   # exit if ESC is pressed
