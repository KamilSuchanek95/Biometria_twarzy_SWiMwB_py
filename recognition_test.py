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
face_recognizer.read('models/lbph/model_BioID_LBPH-2020-06-13_14-32-22.xml')
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer.read('models/eigenfaces/model_BioID_eigenfaces_256-2020-06-10_10-05-31.xml')
""" ... @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ """

### testy i odleglosci pozytywne/negatywne
p = []
n = []
#with open("models/fisherfaces/p-positive_BioIDFisherfaces_256-2020-06-09_06-06-30" + ".csv", "a") as file:
#with open("models/eigenfaces/p-positive_BioID_eigenfaces_256-2020-06-10_10-05-31" + ".csv", "a") as file:
with open("models/lbph/p-positive_BioID_LBPH-2020-06-13_14-32-22" + ".csv", "a") as file:
	file.write('subject, minimum, mean, maximum\n')

#with open("models/fisherfaces/p-negative_BioIDFisherfaces_256-2020-06-09_06-06-30" + ".csv", "a") as file:
#with open("models/eigenfaces/p-negative_BioID_eigenfaces_256-2020-06-10_10-05-31" + ".csv", "a") as file:
with open("models/lbph/p-negative_BioID_LBPH-2020-06-13_14-32-22" + ".csv", "a") as file:
	file.write('subject, minimum, mean, maximum\n')
for subject in os.listdir('test_images'):
#	if subject.startswith("."):
#		continue;
	oo = os.listdir('test_images/' + subject)
	for t in oo:
		test_img = cv2.imread("test_images/" + subject + "/" + t)
		predicted_img, how_much, who = TS.predict(test_img, face_recognizer, subjects, eq = 0)
		if predicted_img is None:
			continue
		if who == subject:
			p.append(how_much)
		else:
			n.append(how_much)
		# print(f'{subject:s}:	{how_much:f},	rozpoznano:{who:s}\n')
		# cv2.imshow("Kto to?", predicted_img)
		# cv2.waitKey(0)
		#with open("models/fisherfaces/test_BioIDFisherfaces_256-2020-06-09_06-06-30" + ".csv", "a") as file:
		#with open("models/eigenfaces/test_BioID_eigenfaces_256-2020-06-10_10-05-31" + ".csv", "a") as file:
		with open("models/lbph/test_BioID_LBPH-2020-06-13_14-32-22" + ".csv", "a") as file:
			file.write(subject + ',' + who + "\n")
	#with open("models/fisherfaces/p-positive_BioIDFisherfaces_256-2020-06-09_06-06-30" + ".csv", "a") as file:
	#with open("models/eigenfaces/p-positive_BioID_eigenfaces_256-2020-06-10_10-05-31" + ".csv", "a") as file:
	with open("models/lbph/p-positive_BioID_LBPH-2020-06-13_14-32-22" + ".csv", "a") as file:
		if len(p) > 0:
	 		file.write(subject + ',' + str(np.min(p)) + ',' + str(np.mean(p)) + ',' + str(np.max(p)) + "\n")
		else:
	 		file.write(subject + ',0,0,0\n')
	#with open("models/fisherfaces/p-negative_BioIDFisherfaces_256-2020-06-09_06-06-30" + ".csv", "a") as file:
	#with open("models/eigenfaces/p-negative_BioID_eigenfaces_256-2020-06-10_10-05-31" + ".csv", "a") as file:
	with open("models/lbph/p-negative_BioID_LBPH-2020-06-13_14-32-22" + ".csv", "a") as file:
		if len(n) > 0:
	 		file.write(subject + ',' + str(np.min(n)) + ',' + str(np.mean(n)) + ',' + str(np.max(n)) + "\n")
		else:
	 		file.write(subject + ',0,0,0\n')
	p = []
	n = []
print('koniec testow')
#ipdb.set_trace()
