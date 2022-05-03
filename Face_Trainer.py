#train with the faces folder to create a YAML file

import cv2 #OpenCV
import numpy as np #convert Images to Numerical array 
import os  
from PIL import Image #Pillow lib for handling images

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #Haarcascade classifier
recognizer = cv2.face.createLBPHFaceRecognizer()

Face_ID = -1 
pev_person_name = ""
y_ID = []
x_train = []

Face_Images = os.path.join(os.getcwd(), "Face_Images") #Telling program where to look for saved face images 
print (Face_Images)

for root, dirs, files in os.walk(Face_Images): #go to the face image directory 
    for file in files: #check every directory
            if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"): #image files ending with jpeg,jpg or png 
			path = os.path.join(root, file)
			person_name = os.path.basename(root)
			print(path, person_name)

			
			if pev_person_name!=person_name: #Check if the name of person has changed 
				Face_ID=Face_ID+1 #If yes increment the ID count 
				pev_person_name = person_name

			
			Gery_Image = Image.open(path).convert("L") # convert image to greysclae using Pillow
			Crop_Image = Gery_Image.resize( (800,800) , Image.ANTIALIAS) #Crop the Grey Image to 800*800 
			Final_Image = np.array(Crop_Image, "uint8")
			#print(Numpy_Image)
			faces = face_cascade.detectMultiScale(Final_Image, scaleFactor=1.5, minNeighbors=5) #Detect face in all images 
			print (Face_ID,faces)

			for (x,y,w,h) in faces:
				roi = Final_Image[y:y+h, x:x+w] #crop the Region of Interest (ROI)
				x_train.append(roi)
				y_ID.append(Face_ID)

recognizer.train(x_train, np.array(y_ID)) #Create a Matrix of Training data 
recognizer.save("face-trainner.yml") #Save the matrix as YML file 
