# -*- coding: utf-8 -*-

# pip3 install  opencv-contrib-python
from program.helpers.program_helper import *

import cv2
import os
import numpy

class Face_detector:

    def __init__(self, classifier_path):
        if os.path.exists(classifier_path):
            self.cascade = cv2.CascadeClassifier(classifier_path)
        else:
            self.cascade = None
        self.photo = []
        self.face = []


    def detect_face(self, image, scale_factor = 1.05, min_neighbors = 4):
        #return None if image is None else None
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
        if (len(faces) == 0):
            return None
        return self.choose_the_widest_frame(faces, gray)

    def choose_the_widest_frame(self, faces, gray):
        widths = faces.transpose()[2] # [[x,y,w,h],[x,y,w,h]...] => [[x,x...],[y,y...],[w,w...],[h,h...]] => arr[2] => [w,w...]
        index_of_max_width = widths.argmax()
        x, y, w, h = faces[index_of_max_width]
        self.face = gray[y: y + w, x: x + h]
        return self.face


class Face_recognitor():

    def __init__(self, face_detector, algorithm = 'lbph'):
        self.algorithm = algorithm.lower()
        self.face_recognizer = self.select_recognizer(self.algorithm)
        self.face_detector = face_detector
        self.subjects = []

    def roi_must_be_square(self, algorithm):
        if algorithm == 'lbph':
            return False
        else:
            return True

    def select_recognizer(self, algorithm):
        switcher = {
                'lbph':         cv2.face.LBPHFaceRecognizer_create(),
                'fisherface':   cv2.face.FisherFaceRecognizer_create(),
                'eigenface':    cv2.face.EigenFaceRecognizer_create()
        }
        return switcher.get(algorithm, 'Invalid algorithm name.')
    
    def load_subjects(self, subjects_path):
        with open(subjects_path, "r") as f: subjects = f.readlines()[0]
        self.subjects = subjects.split(',')

    def create_subjects_file(self, models_path):
        with open(path(models_path, self.algorithm + '_subjects.csv'), "w") as f: f.write(','.join(self.subjects))

    def read_model(self, model_path, subjects_path):
        self.face_recognizer.read(model_path)
        self.load_subjects(subjects_path)

    def prepare_training_data(self, training_data_folder_path):
        self.subjects = listdir_with_glob(training_data_folder_path)
        faces, labels, label = [], [], -1 
        for dir_name in self.subjects:
            label += 1
            subject_dir_path = path(training_data_folder_path, dir_name)
            subject_images_names = listdir_with_glob(subject_dir_path)
            for image_name in subject_images_names:
                image = cv2.imread(path(subject_dir_path, image_name))
                face = self.face_detector.detect_face(image)
                if face is not None:
                    if self.roi_must_be_square(self.algorithm):
                        face = cv2.resize(face, (256, 256))
                    faces.append(face)
                    labels.append(label)
                else:
                    None
        return faces, labels


    def train_model(self, training_data_folder_path, models_path):
        faces, labels = self.prepare_training_data(training_data_folder_path)
        self.face_recognizer.train(faces, numpy.array(labels))
        self.face_recognizer.save(path(models_path, self.algorithm + '_model' + '.xml'))
        self.create_subjects_file(models_path)

    def predict(self, image):
        face = self.face_detector.detect_face(image) 
        if face is None:
            return None, None, None
        if self.roi_must_be_square(self.algorithm):
            face = cv2.resize(face, (256, 256))
        label, how_much = self.face_recognizer.predict(face)
        label_text = self.subjects[label]
        return image, how_much, label_text



# # This Class is unused for now jet
# class Camera:

#     def __init__(self, name, camera = 0):
#         self.name = name
#         self.image = []
#         self.cam = cv2.VideoCapture(camera)
#         return None if not self.is_device_work(self.cam) else None
        
#         return self

#     def is_device_work(self, cam):
#         if cam is None or not cam.isOpened(): 
#             return False
#         else:
#             return True

#     def end_frameing(self):
#         print("Turning off camera.")
#         self.cam.release()
#         print("Camera off.")
#         cv2.destroyAllWindows()

#     def create_image_path(self):
#         return path(Camera.IMAGES_PATH, self.name + '_' + str(self.number_or_files_with_name(Camera.IMAGES_PATH) + 1) + '.jpg')

#     def number_or_files_with_name(self, dir):
#         # return len(next(os.walk(dir))[2])
#         return len(glob.glob(dir,"*{self.name}_*.jpg"))

#     def get_image(self):
#         return None if not self.is_device_work(self.cam) else None
#         while True:
#             try:
#                 check, frame = self.cam.read() # odczytaj okno
#                 cv2.imshow("Capturing", frame) # wyświetl okno
#                 key = cv2.waitKey(1) # obiekt przycisku, czekaj 1ms z oknem
#                 if key == ord('s'):  # jeśli "s" to zapisz zdjęcie
#                     self.image = frame
#                     cv2.imwrite(filename=self.create_image_path(self.name), img=frame)
#                     self.end_frameing()
#                     # cv2.waitKey(1650)
#                     print("Image saved!")
#                     break
#                 elif key == ord('q'): # jeśli "q" to wyłącz kamerkę
#                     self.image = None
#                     self.end_frameing()
#                     break
#             except(KeyboardInterrupt): # jeśli ctrl+C albo coś innego również zakończ
#                 self.image = None
#                 self.end_frameing()
#                 break
#         return self.image
