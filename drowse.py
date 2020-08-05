import cv2
import os
from keras.models import load_model
import numpy as np
from threading import Thread
import argparse
import time
import playsound

#Accept audio file from path in disk and play it
def sound_alarm(path):
        playsound.playsound(path)

#Reading files for detecting face and both eyes
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

#Accepting the name of alarm file as an argument --alarm alarm.mp3
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="",help="path alarm .WAV file")
args = vars(ap.parse_args())


lbl=['Close','Open']

#Loading the pretrained model
model = load_model('models/cnnCat2.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

COUNTER=0
ALARM_ON=False
while(True):

    #Reading the captured frame
    ret, frame = cap.read()
    height,width = frame.shape[:2] 
    
    #Converting it to grayscale for better classification
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detecting face and both eyes from the caputred frame
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    
    #Resizing the frame
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )
     
    #Classifying both eyes as open or closed based on prediction of the model
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break   
    
    
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>15):
        #person is feeling sleepy so we beep the alarm
        # cv2.imwrite(os.path.join(path,'image.jpg'),frame)

        # if the alarm is not on, turn it on
            # check to see if an alarm file was supplied,
            # and if so, start a thread to have the alarm
            # sound played in the background
        if args["alarm"] != "":
            t = Thread(target=sound_alarm,args=(args["alarm"],))
            t.deamon = True
            t.start()
        # draw an alarm on the frame
        cv2.putText(frame, "DROWSINESS ALERT! WAKE UP!!", (40, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
