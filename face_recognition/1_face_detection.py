# Face Detection is performed using classifiers. 
# A classifier is an algorithm that decides whether a given image is positive or negative if a face is present or not. 
# It needs to be trained on 1.000 and 10.000 of images with and without faces. OpenCV comes with pre trained classifiers.
# Haar Cascade (basic) - https://github.com/opencv/opencv/tree/master/data/haarcascades

import cv2 as cv

img = cv.imread('./Faces/faces_0.jpeg')

desired_height = 700
aspect_ratio = desired_height/img.shape[0]
desired_width = int(img.shape[1]*aspect_ratio)
dim = (desired_width, desired_height)
img = cv.resize(img, dsize=dim, interpolation=cv.INTER_AREA)
cv.imshow('Faces', img)

# Face detection does not involve skin tone or colors present in the image. Haar cascades look at an object in an image and using the edges tries to determine whether it's a face or not. 

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray Image', gray)

haar_cascade = cv.CascadeClassifier('haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)


cv.waitKey(0)