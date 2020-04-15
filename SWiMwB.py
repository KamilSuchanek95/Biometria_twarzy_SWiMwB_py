# -*- coding: utf-8 -*-
# pip3 install matplotlib
# pip3 install numpy
# pip3 install  opencv-contrib-python

import numpy as np
import cv2
import matplotlib as plot
import ipdb

class Face_detector:

    def __init__(self):
        self.photo = []
        self.faces = []
 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    def get_photo(self):
        key = cv2.waitKey(1) # obiekt przycisku dla okna kamerki 1ms
        webcam = cv2.VideoCapture(0) # obiekt kamerki
        while True:
            try:
                check, frame = webcam.read() # odczytaj okno
                cv2.imshow("Capturing", frame) # wyświetl okno
                key = cv2.waitKey(1) # obiekt przycisku, czekaj 1ms z oknem
                if key == ord('s'):  # jeśli "s" to zapisz zdjęcie
                    cv2.imwrite(filename="images\\image_object_" + str(id(self)) + '.jpg', img=frame)
                    webcam.release()
                    cv2.waitKey(1650)
                    cv2.destroyAllWindows()
                    print("Image saved!")
                    break
                elif key == ord('q'): # jeśli "q" to wyłącz kamerkę
                    print("Turning off camera.")
                    webcam.release()
                    print("Camera off.")
                    print("Program ended.")
                    cv2.destroyAllWindows()
                    break
            except(KeyboardInterrupt): # jeśli ctrl+C albo coś innego również zakończ
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break

    def detect_face(self):
        img = cv2.imread('images\\image_object_' + str(id(self)) + '.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        scaleFactor = 1.3
        minNeighbors = 4
        self.faces = self.face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)
        return self.faces

    def show_detected_faces(self): # to pozostałość po testowaniu, póki co zostanie.
        img = cv2.imread('images/image_object_' + str(id(self)) + '.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for (x,y,w,h) in self.faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('Wynik detekcji', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('done [show_detected_faces]')
