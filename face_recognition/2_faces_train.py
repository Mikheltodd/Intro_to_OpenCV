import os
import cv2 as cv
import numpy as np

# Create a list of people (the name of the folders) 

# people = ['Ana De Armas', 'Jim Carrey', 'Keanu Reeves', 'Terry Crews', 'Zooey Deschanel']

# You can also create an empty list and loop over every folder

dir = r'D:\OneDrive - UPB\UPB\Cursos\Python\Notebooks\Intro_to_OpenCV\face_recognition\photographs'

people = []

for i in os.listdir(dir):
    people.append(i)

print(people)

features = []
labels = []
haar_cascade = cv.CascadeClassifier('haar_face.xml')


def create_train():
    for person in people:
        path = os.path.join(dir, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img_array = cv.imread(img_path)
            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                #faces region of interest
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print("Training done ----------")

# print(f'Length of the features = {len(features)}')
# print(f'Length of the labels = {len(labels)}')

features = np.array(features, dtype='object')
labels = np.array(labels)

# LBPH (Local Binary Pattern Histogram) is a Face-Recognition algorithm it is used to recognize the face of a person
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer on the features list and the labels list
face_recognizer.train(features, labels)

# This is done to avoid training everytime we change the images. We keep a "copy" of the trained model. 
face_recognizer.save('face_trained.yml')

np.save('features.npy', features)
np.save('labels.npy', labels)