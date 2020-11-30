# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 15:31:05 2020

@author: Deni
"""

import os
import cv2
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import time as tm
import datetime
#Excels
x = datetime.datetime.now()
a = x.year
b = x.month
c = x.day
date = str(a) + '-' + str(b) + '-' + str(c)

d = x.hour
e = x.minute
f = x.second
time = str(d) + ':' + str(e) + ':' + str(f)

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Defining a function that will do the detections
def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.3,minNeighbors=5)
    return faces,gray_img

def train_classifier(faces,face_ID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    print(face_recognizer)
    face_recognizer.train(faces,np.array(face_ID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h)=face
    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)

def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,4,(255,0,0),3)
    


def labels_for_training_data(directory):
    faces=[]
    faceID=[]
    
    for path,subdirnames,filenames in os.walk(directory):
        
        for filename in filenames:
                if filename.startswith("."):
                    print("Skipping system files")
                    continue
                id=os.path.basename(path)
                img_path=os.path.join(path,filename)
                print("img_path",img_path)
                print("id:",id)
                test_img=cv2.imread(img_path)
                if test_img is None:
                      print("Image not loaded properly")
                     
                      continue
                faces_rect,gray_img=faceDetection(test_img)
                if len(faces_rect)!=1:
                      continue

                (x,y,w,h)=faces_rect[0]
                rol_gray=gray_img[y:y+w,x:x+h]
                faces.append(rol_gray)
                faceID.append(int(id))
    return faces,faceID

def trainingProcess():

    faces,faceID = labels_for_training_data('imagesAttandance/')
    face_recognizer = train_classifier(faces, faceID)
    face_recognizer.save('trainingdata.yml')
    status_train = "Done"
    return status_train
    
def openCamera():
    faces,faceID = labels_for_training_data('imagesAttandance/')
    face_recognizer = train_classifier(faces, faceID)
    face_recognizer.read('trainingdata.yml')
    name={0:"Deni", 1 : "Natario", 2 : "Johan"} 
    toogle = 0
    cap=cv2.VideoCapture(0)
    while True:
        ret,test_img=cap.read()
        faces_detected,gray_img=faceDetection(test_img)
        for (x,y,w,h) in faces_detected:
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=1)
            pass
        resized_img=cv2.resize(test_img,(1000,700))
        cv2.imshow("face detection ",resized_img)
        cv2.waitKey(10)
        for face in faces_detected:
            (x,y,w,h)=face
            rol_gray=gray_img[y:y+h,x:x+h]
            label,confidence=face_recognizer.predict(rol_gray)
            print("confidence",confidence)
            print("label",label)
            draw_rect(test_img,face)
            predict_name=name[label]
            put_text(test_img,predict_name,x,y)
            print(name[label])
            if confidence<55:
                put_text(test_img,predict_name,x,y)
                print('confidence lebih dari 55')
                df = pd.DataFrame([[date,time,predict_name]])
                toogle = 1
                
        resized_img=cv2.resize(test_img,(1000,700))
        cv2.imshow("face detection ",resized_img)
        
        if toogle == 1:
            writer = pd.ExcelWriter('dataexcel.xlsx',engine = 'openpyxl')
            writer.book = load_workbook('dataexcel.xlsx')
            writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
            reader = pd.read_excel(r'dataexcel.xlsx')
            df.to_excel(writer,index=False,header=False,startrow=len(reader)+1)
            writer.close()
            tm.sleep(3)
            break
        
        if cv2.waitKey(10) == ord('q'):
           break
       
    cap.release()
    cv2.destroyAllWindows()
    return predict_name