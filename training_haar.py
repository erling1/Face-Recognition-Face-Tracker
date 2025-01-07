import cv2
import numpy as np
import os


class FaceDetector():

    def __init__(self):
        #Creating an instance of LBPHFaceRecognizer, which will be trained on our custom dataset.
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        #Creating an instance of our face detector 
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    def images_and_labels(self, folder_path):
        training_images = []
        ids = []
        for filename in os.listdir(folder_path):
            #reading Image
            img = cv2.imread(os.path.join(folder_path,filename),cv2.IMREAD_GRAYSCALE)
            rectangle_faces_coords = self.face_detector.detectMultiScale(img)

            #getting name of the person in the picture
            words = filename.split('_')
            ID_person = int(words[-1])

            ids.append(ID_person)
            training_images.append(rectangle_faces_coords)

            faces = []

            for (x,y,w,h) in rectangle_faces_coords:
                faces.append(img[y:y+h, x:x+w])

        return faces,ids  
    

    def train_model(self, faces, labels, path):

        self.recognizer.train(faces, np.array(labels))
        # Save the trained model to a file
        self.recognizer.save(f'{path}.xml')

        """# Create the 'trainer' folder if it doesn't exist
        if not os.path.exists("trainer"):
            os.makedirs("trainer")
        # Save the model into 'trainer/trainer.yml'
        self.recognizer.write('trainer/trainer.yml')"""


detector = FaceDetector

faces, ids = detector.images_and_labels()






