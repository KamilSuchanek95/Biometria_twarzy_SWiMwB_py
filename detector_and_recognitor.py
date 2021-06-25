# -*- coding: utf-8 -*-

import numpy as np


#new
import os
import glob


# pip3 install  opencv-contrib-python
import cv2
from tkinter import messagebox
import tkinter

# # This Class is unused for now jet
# class Camera:
    
#     CORE_PATH   =   os.path.dirname(os.path.realpath(__file__)) # used below too
#     RESOURCES_PATH  = os.path.join(CORE_PATH, 'resources')
#     IMAGES_PATH = os.path.join(RESOURCES_PATH, 'images')

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
#         return os.path.join(Camera.IMAGES_PATH, self.name + '_' + str(self.number_or_files_with_name(Camera.IMAGES_PATH) + 1) + '.jpg')

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


class Face_detector:
    CORE_PATH   =   os.path.dirname(os.path.realpath(__file__))
    RESOURCES_PATH  = os.path.join(CORE_PATH, 'resources')
    MODELS_FILES_PATH   = os.path.join(RESOURCES_PATH, 'models')
    HAAR_CASCADE_FILE_PATH  = os.path.join(MODELS_FILES_PATH, 'haarcascade_frontalface_default.xml')
    def __init__(self, classifier_path = HAAR_CASCADE_FILE_PATH):
        self.cascade = cv2.CascadeClassifier(classifier_path)
        # TODO: check if clasiffier file exists
        self.photo = []
        self.face = []

    def detect_face(self, image):
        #return None if image is None else None
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
        # jesli nie wykryto twarzy zwroc None
        if (len(faces) == 0):
            return None
        # koordynaty ROI (najszerszego)
        tr = np.transpose(faces)[2]
        max_width_roi_idx, = np.where(np.max(tr) == tr)
        (x, y, w, h) = faces[max_width_roi_idx[0]]
        # zwroc ROI w skali szarości oraz koordynaty
        self.face = gray[y: y + w, x: x + h]
        return self.face


class Face_recognitor():
    CORE_PATH   =   os.path.dirname(os.path.realpath(__file__))
    RESOURCES_PATH  = os.path.join(CORE_PATH, 'resources')
    MODELS_FILES_PATH   = os.path.join(RESOURCES_PATH, 'models')
    TRAINING_DATA_FOLDER_PATH = os.path.join(RESOURCES_PATH, 'training_images')

    def __init__(self, algorithm = 'lbph'):
        self.algorithm = algorithm.lower()
        self.face_recognizer = self.select_recognizer(self.algorithm)
        self.face_detector = Face_detector()
        self.subjects = []

    def roi_must_be_square(self, algorithm):
        if algorithm != 'lbph':
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
        with open(subjects_path, "r") as f: lines = f.readlines
        for l in lines:
            self.subjects = l.split(',')[0:-1]

    def create_subjects_file(self, subjects, models_path = MODELS_FILES_PATH):
        with open(os.path.join(models_path, self.algorithm + '_subjects.csv'), "w") as file:
            for n in subjects:
                file.write(n)
                file.write(',')

    def read_model(self, model_path, subjects_path):
        self.face_recognizer.read(model_path)
        self.load_subjects(subjects_path)

    # przygotowanie wykrytych twarzy do uczenia modelu
    def prepare_training_data(self, training_data_folder_path = TRAINING_DATA_FOLDER_PATH):
        # w naszym folderze training_images beda foldery s0 s1 s2 s3 dla kazdej osoby ktora chcemy rozpoznawac,
        # oraz dla nieznanych, ktorych mozna zdefiniowac jako "typowe" nieznane twarze z bardzo malym zbiorem uczacym.
        # ewentualne wykluczenie "nieznanych" mozna tez oprzec o zbyt duza niepewnosc zwracana jako "how_much".
        # Warto po przeszkoleniu modelu, wygenerować "typowe" wartosci niepewnosci dla predykcji obrazow testowych, aby
        # potem sugerowac sie nimi w celu stwierdzenia, czy niepewnosc jest dosc mala aby wskazanie konkretnej tozsamosci
        # bylo wiarygodne.
        licznik = 0
        dirs = os.listdir(training_data_folder_path)
        self.subjects = dirs
        faces, labels, label = [], [], -1 
        for dir_name in dirs:
            # zakladamy, ze kazdy folder ma w nazwie tylko jedna litere 's' a po niej numer zatwierdzanej tozsamosci,
            # zastepujemy 's' => '' i uzyskujemy identyfikator tozsamosci.
            #label = int(dir_name.replace("s", ""))
            label = label + 1
            # skladanie sciezki do pliku ./training_images/s0 ...
            subject_dir_path = os.path.join(training_data_folder_path, dir_name)
            # odczytujemy liste zdjec w lokacji wskazanej w linii wyzej
            subject_images_names = os.listdir(subject_dir_path)
            # petla po obrazach w celu detekcji twarzy, w przypadku jej wykrycia, dodanie owej do listy wykrytych, owa lista
            # posluzy do nauki modelu
            for image_name in subject_images_names:
                # ignorujemy pliki systemowe '.' '..'
                if image_name.startswith("."):
                    continue
                # skladamy sciezke do zdjecia z nazwa owego
                image_path = os.path.join(subject_dir_path, image_name)
                # wczytanie owego do zmiennej
                image = cv2.imread(image_path)
                # wykrycie twarzy
                face = self.face_detector.detect_face(image)
                # jeśli w istocie ją wykryto, dodawanie do listy twarzy oraz jej identyfikatora
                if face is not None:
                    licznik +=1
                    print( str(licznik) + ' Tak ' + image_name)
                    if self.roi_must_be_square(self.algorithm):
                        face = cv2.resize(face, (256, 256))
                    faces.append(face)
                    labels.append(label)
                else:
                    licznik +=1
                    print(str(licznik) + ' Nie '+image_name)
                    None
        # zwroc twarze z identyfikatorami
        return faces, labels, self.subjects


    def train_model(self, training_data_folder_path = TRAINING_DATA_FOLDER_PATH, models_path = MODELS_FILES_PATH):
        faces, labels, subjects = self.prepare_training_data(training_data_folder_path)
        self.face_recognizer.train(faces, np.array(labels))
        self.face_recognizer.save(os.path.join(models_path, self.algorithm + '_model' + '.xml'))
        self.create_subjects_file(subjects, models_path)

    # rozpoznawanie osoby na zdjeciu i podpisywanie
    def predict(self, image):
        #img = test_img.copy() # kopia obrazu
        # face = self.face_detector.detect_face(img) # wykrywanie twarzy
        #label, how_much = self.face_recognizer.predict(img) # rozpoznawanie
        #label_text = self.subjects[label] # odszukanie tozsamosci po identyfikatorze
        #return how_much, label_text # zwroc odpisany obraz, niepewnosc oraz tozsamosc
        # img = image.copy()  # kopia obrazu
        face = self.face_detector.detect_face(image)  # wykrywanie twarzy
        if face is None:
            return None, None, None
        if self.roi_must_be_square(self.algorithm):
            face = cv2.resize(face, (256, 256))
        label, how_much = self.face_recognizer.predict(face)  # rozpoznawanie
        label_text = self.subjects[label]  # odszukanie tozsamosci po identyfikatorze
        return image, how_much, label_text  # zwróć podpisany obraz, niepewnosc oraz tozsamosc
