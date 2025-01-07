import cv2
from reciving_feed import generating_frames_array
import requests

url = "http://192.168.1.172:8000/camera"

response = requests.get(url, stream=True)

haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
 


# Call the generator function
for frame in generating_frames_array(response):
    if frame is not None:
        cv2.imshow('Video Stream', frame)  # Display the current frame

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=9)

        # Iterating through rectangles of detected faces 
        for (x, y, w, h) in faces_rect: 
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
        
        cv2.imshow('Detected faces', frame) 



        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup resources
cv2.destroyAllWindows()