import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm.notebook import tqdm as tq
import cv2
import sys

mypath= 'test/'
dataset = 'test_data/'
cascPath = "lbpcascade_animeface.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

file_name = []
full_path = []
for path, subdirs, files in os.walk(mypath):
    for name in files:
        full_path.append(os.path.join(path, name)) 
        file_name.append(name)

for name,full in zip(file_name,full_path):
    image = cv2.imread(full)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if os.path.exists(os.path.join(dataset)) == False:
        os.makedirs(os.path.join(dataset))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # saving faces according to detected coordinates 
        sub_face = image[y:y+h, x:x+w]
        #FaceFileName = "cropped_face/face_" + str(y+x) + ".jpg" # folder path and random name image
        file_dest = os.path.join(dataset, name)
        print("Copy",full,"to",file_dest)
        cv2.imwrite(file_dest, sub_face)    