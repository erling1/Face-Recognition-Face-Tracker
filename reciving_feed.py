import cv2
import requests
import numpy as np

"""url = "http://192.168.1.172:8000/camera"

response = requests.get(url, stream=True)

if response.status_code == 200:
    print("respons ok")
        # Read the incoming byte chunks"""

def generating_frames_array(response):
    bytes_stream = b''  # Initialize an empty byte stream

    for chunk in response.iter_content(chunk_size=1024):
        bytes_stream += chunk  # Add the new chunk to the byte stream
        a = bytes_stream.find(b'\xff\xd8')  # Start of JPEG frame
        b = bytes_stream.find(b'\xff\xd9')  # End of JPEG frame
        
        if a != -1 and b != -1:
            jpg = bytes_stream[a:b+2]  # Extract the JPEG frame
            bytes_stream = bytes_stream[b+2:]  # Remove the processed frame from the buffer
            
            # Decode JPEG bytes to an OpenCV image
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            
            if frame is not None:
                yield frame  # Yield the frame instead of returning
                



                    #cv2.imshow('Video Stream', frame)
                
                
                
                
                #if cv2.waitKey(1) == 27:  # Exit on pressing ESC
                #    break

    #cv2.destroyAllWindows()
    
        