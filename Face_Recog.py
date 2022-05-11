import cv2 #OpenCV
import numpy as np #convert Image to Numerical array 
import os 
import RPi.GPIO as GPIO
from time import sleep
from PIL import Image #Pillow lib for handling images

led = LED(23)

labels = ["Harry", "Em",] #Names of people

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Haarcascade classifier
recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load("face-trainner.yml") #Load YML file created from face traniner

cap = cv2.VideoCapture(0) #Get vidoe feed from the Camera

while(True):
    led.off() #LED is off
    ret, img = cap.read() # Break video into frames 
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert Video frame to Greyscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, ) #Recognize faces
    for (x, y, w, h) in faces:
    	roi_gray = gray[y:y+h, x:x+w] #Convert Face to greyscale
    	
    	id_, confidence = recognizer.predict(roi_gray) #recognize the Face
    	
    
    	if (confidence<100):
            font = cv2.FONT_HERSHEY_SIMPLEX #Font style
            name = labels[id_] #Acquire name from the List using ID number
            confidence = "{0}%".format(round(100 - confidence)) #confidence rating 
            led.on() #LED turns on if face is a match
            
            cv2.putText(img, name, (x+5,y-5), font, 1, (255,255,255), 2) #Place name around face
            cv2.putText(img, str(confidence), (x+5,y+h-5), font,1, (225,225,0), 2) #Place confidence rating around face
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #Place box around face
        else:
            name = "Unkown" #Place name around face 
            font = cv2.FONT_HERSHEY_SIMPLEX #Font style
            confidence = "{0}%".format(round(100 - confidence)) #Confidence rating
            led.off() #LED off
            
            cv2.putText(img, name, (x+5,y-5), font, 1, (255,255,255), 2) #Place name around face
            cv2.putText(img, str(confidence), (x+5,y+h-5), font,1, (225,225,0), 2) #place confidence rating around face
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #place box around face 

    cv2.imshow('Preview',img) #Display the Video
    if cv2.waitKey(20) & 0xFF == ord('q'):
    	break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
