# -*- coding: utf-8 -*-

# pip3 install matplotlib
# pip3 install numpy
# pip3 install  opencv-contrib-python

import numpy as np
import cv2
import matplotlib as plot


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        #print(check) #prints true as long as the webcam is running
        #print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            #img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            #img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            #print("Processing image...")
            #img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            #print("Converting RGB image to grayscale...")
            #gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            #print("Converted RGB image to grayscale...")
            #print("Resizing image to 28x28 scale...")
            #img_ = cv2.resize(gray,(28,28))
            #print("Resized...")
            #img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            print("Image saved!")
        
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

img = cv2.imread('saved_img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

scaleFactor = 1.01
minNeighbors = 100
faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)


#print(faces)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('Wynik detekcji',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('done')
