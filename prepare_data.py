import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
from tqdm.notebook import tqdm as tq
import cv2
import sys

mypath= 'vtuber/'
dataset = 'dataset/'
cascPath = "lbpcascade_animeface.xml"
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

file_name = []
tag = []
full_path = []
for path, subdirs, files in os.walk(mypath):
    for name in files:
        full_path.append(os.path.join(path, name)) 
        tag.append(path.split('/')[-1])        
        file_name.append(name)

for path,name,full in zip(tag,file_name,full_path):
    file_dest = os.path.join(dataset,path, name)    
    if os.path.exists(os.path.join(dataset,path)) == False:
        os.makedirs(os.path.join(dataset,path))

    if os.path.isfile(file_dest) == True:
        continue

    image = cv2.imread(full)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(e)
        continue
    
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
        
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        # saving faces according to detected coordinates 
        sub_face = image[y:y+h, x:x+w]
        #FaceFileName = "cropped_face/face_" + str(y+x) + ".jpg" # folder path and random name image
        print("Copy",full,"to",file_dest)
        cv2.imwrite(file_dest, sub_face)    

"""
df = pd.DataFrame({"path":full_path,'file_name':file_name,"tag":tag})
df.groupby(['tag']).size()
df.head()

X= df['path']
y= df['tag']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=300)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=100)

df_tr = pd.DataFrame({'path':X_train
              ,'tag':y_train
             ,'set':'train'})

df_te = pd.DataFrame({'path':X_test
              ,'tag':y_test
             ,'set':'test'})

df_val = pd.DataFrame({'path':X_val
              ,'tag':y_val
             ,'set':'validation'})

print('train size', len(df_tr))
print('val size', len(df_te))
print('test size', len(df_val))

df_all = df_tr.append([df_te,df_val]).reset_index(drop=1)

print('===================================================== \n')
print(df_all.groupby(['set','tag']).size(),'\n')

print('===================================================== \n')

#cek sample datanya
df_all.sample(3)

datasource_path = "vtuber/"
dataset_path = "dataset/"

for index, row in tq(df_all.iterrows()):
    #detect filepath
    file_path = row['path']
    print(file_path)

    if os.path.exists(file_path) == False:
            file_path = os.path.join(datasource_path,row['tag'],row['image'].split('.')[0])            
    
    #make folder destination dirs
    if os.path.exists(os.path.join(dataset_path,row['set'],row['tag'])) == False:
        os.makedirs(os.path.join(dataset_path,row['set'],row['tag']))
    
    #define file dest
    destination_file_name = file_path.split('/')[-1]
    file_dest = os.path.join(dataset_path,row['set'],row['tag'],destination_file_name)
    
    #copy file from source to dest
    if os.path.exists(file_dest) == False:
        # Read the image
        image = cv2.imread(file_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # saving faces according to detected coordinates 
            sub_face = image[y:y+h, x:x+w]
            #FaceFileName = "cropped_face/face_" + str(y+x) + ".jpg" # folder path and random name image
            cv2.imwrite(file_dest, sub_face)
"""