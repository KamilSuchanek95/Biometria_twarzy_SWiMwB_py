import cv2
import os
import numpy as np
import ipdb
import datetime
import pre_SWiMwB as TS

### Reading subjects from file:
# with open("models/subjects_METODA-DATA.csv", "r") as file:
# 	for f in file:
# 		subjects = f.split(',')
# 		subjects = subjects[0:-1]  

## razem z szkoleniem: eq =0 dla lbph, dla innych eq =1
faces, labels, subjects = TS.prepare_training_data("training_images", eq = 0)

""" odznaczyc wlasciwe @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ """
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
""" ... @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ """
ipdb.set_trace();

face_recognizer.train(faces, np.array(labels))
ts = datetime.datetime.now()
date_str = "{}".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))

""" odznaczyc wlasciwe @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ """
# face_recognizer.save('models/model_BioID_eigenfaces_256-' + date_str + '.xml')
face_recognizer.save('models/model_BioID_LBPH-' + date_str + '.xml')
# face_recognizer.save('models/model_BioIDFisherfaces_256-' + date_str + '.xml')
""" ... @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ """

""" odznaczyc wlasciwe @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ """
# with open("models/subjects_eigenfaces-" + date_str + ".csv", "w") as file:
with open("models/subjects_BioID_LBPH-" + date_str + ".csv", "w") as file:
# with open("models/subjects_BioIDFisherfaces_256-" + date_str + ".csv", "w") as file:
	for n in subjects:
		file.write(n)
		file.write(',')

