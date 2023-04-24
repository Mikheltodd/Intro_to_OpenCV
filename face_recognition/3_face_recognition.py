import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Ana De Armas', 'Jim Carrey', 'Keanu Reeves', 'Terry Crews', 'Zooey Deschanel']
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Use a different photo for validation
img = cv.imread(r'D:\OneDrive - UPB\UPB\Cursos\Python\Notebooks\Intro_to_OpenCV\face_recognition\validation\2.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Person', gray)

# Detect the face in the image
faces_rect = haar_cascade.detectMultiScale(gray,  1.1, 3)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), thickness=2)
    cv.putText(img, str(round(confidence, 2)) + '%', (x+w,y+h), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected Face', img)

cv.waitKey(0)