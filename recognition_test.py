import cv2
import os
import numpy as np
import ipdb
import datetime
import pre_SWiMwB as TS

## bez szkolenia:
subjects = os.listdir("training_images")

""" odznaczyc wlasciwe @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"""
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
# face_recognizer.read('models/fisherfaces/model_BioIDFisherfaces_256-2020-06-09_06-06-30.xml') 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('models/lbph/model_BioID_LBPH-2020-06-10_10-20-11.xml')
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer.read('models/eigenfaces/model_BioID_eigenfaces_256-2020-06-10_10-05-31.xml')
""" ... @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ """

### testy i srednie p
p = 0
for subject in os.listdir('test_images'):
#	if subject.startswith("."):
#		continue;
	oo = os.listdir('test_images/' + subject)
	for t in oo:
		test_img = cv2.imread("test_images/" + subject + "/" + t)
		predicted_img, how_much, who = TS.predict(test_img, face_recognizer, subjects, eq = 1)
		if predicted_img is None:
			continue
		p = p + how_much
		print(f'{subject:s}:	{how_much:f},	rozpoznano:{who:s}\n')
		# cv2.imshow("Kto to?", predicted_img)
		# cv2.waitKey(0)
		#with open("models/fisherfaces/test_BioIDFisherfaces_256-2020-06-09_06-06-30" + ".csv", "a") as file:
		#with open("models/eigenfaces/test_BioID_eigenfaces_256-2020-06-10_10-05-31" + ".csv", "a") as file:
		with open("models/lbph/test_BioID_LBPH-2020-06-10_10-20-11" + ".csv", "a") as file:
			file.write(subject + ',' + who + "\n")
	p = p / len(oo)
	#with open("models/fisherfaces/p-mean_BioIDFisherfaces_256-2020-06-09_06-06-30" + ".csv", "a") as file:
	#with open("models/eigenfaces/p-mean_BioID_eigenfaces_256-2020-06-10_10-05-31" + ".csv", "a") as file:
	with open("models/lbph/p-mean_BioID_LBPH-2020-06-10_10-20-11" + ".csv", "a") as file:
	 	file.write(subject + ',' + str(p) + "\n")
	p = 0
print('koniec testow')
#ipdb.set_trace()
