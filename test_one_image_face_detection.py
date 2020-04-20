import cv2
import os
import numpy as np
import ipdb

def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

img = cv2.imread("test_images/test2_s5.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# załadowanie klasyfikatora
face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# wykrycie twarzy
faces = face_cas.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
ipdb.set_trace()
draw_rectangle(img,faces[0])
cv2.imshow("Kto to?", img)
cv2.waitKey(0)

ipdb.set_trace();

