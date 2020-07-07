# -*- coding: utf-8 -*-
# pip3 install matplotlib
# pip3 install numpy
# pip3 install  opencv-contrib-python

import numpy as np
import cv2
# import matplotlib as plot
# import ipdb
import os

class Face_detector:

    def __init__(self):
        self.photo = []
        self.face = []
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    def get_photo(self):
        key = cv2.waitKey(1) # obiekt przycisku dla okna kamerki 1ms
        webcam = cv2.VideoCapture(0) # obiekt kamerki
        while True:
            try:
                check, frame = webcam.read() # odczytaj okno
                cv2.imshow("Capturing", frame) # wyświetl okno
                key = cv2.waitKey(1) # obiekt przycisku, czekaj 1ms z oknem
                if key == ord('s'):  # jeśli "s" to zapisz zdjęcie
                    self.photo = frame
                    cv2.imwrite(filename="images/image_object_" + str(id(self)) + '.jpg', img=frame)
                    webcam.release()
                    cv2.waitKey(1650)
                    cv2.destroyAllWindows()
                    print("Image saved!")
                    break
                elif key == ord('q'): # jeśli "q" to wyłącz kamerkę
                    self.photo = None
                    print("Turning off camera.")
                    webcam.release()
                    print("Camera off.")
                    print("Program ended.")
                    cv2.destroyAllWindows()
                    break
            except(KeyboardInterrupt): # jeśli ctrl+C albo coś innego również zakończ
                self.photo = None
                print("Turning off camera.")
                webcam.release()
                print("Camera off.")
                print("Program ended.")
                cv2.destroyAllWindows()
                break
        return self.photo

    def detect_face(self, image = None):
        if image is None:
            img = cv2.imread('images/image_object_' + str(id(self)) + '.jpg')
        else:
            img = image
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
        # jesli nie wykryto twarzy zwroc None
        if (len(faces) == 0):
            return None
        # koordynaty ROI
        tr = np.transpose(faces)[2]
        max_width_roi_idx, = np.where(np.max(tr) == tr)
        (x, y, w, h) = faces[max_width_roi_idx[0]]
        # zwroc ROI w skali szarości oraz koordynaty
        self.face = gray[y: y + w, x: x + h]
        return self.face


class Face_recognitor():

    def __init__(self, algorithm='lbph'):
        self.algorithm = algorithm.lower()
        if(self.algorithm == 'eigenface'):
            self.face_recognizer = cv2.face.EigenFaceRecognizer_create()
        elif(self.algorithm == 'fisherface'):
            self.face_recognizer = cv2.face.FisherFaceRecognizer_create()
        else:
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_detector = Face_detector()
        self.subjects = []



    def read_model(self, model_path, subjects_path):
        self.face_recognizer.read(model_path)
        with open(subjects_path, "r") as file:
            for f in file:
                sub = f.split(',')
                self.subjects = sub[0:-1]


    # przygotowanie wykrytych twarzy do uczenia modelu
    def prepare_training_data(self, data_folder_path, eq = 0):
        # w naszym folderze training_images beda foldery s0 s1 s2 s3 dla kazdej osoby ktora chcemy rozpoznawac,
        # oraz dla nieznanych, ktorych mozna zdefiniowac jako "typowe" nieznane twarze z bardzo malym zbiorem uczacym.
        # ewentualne wykluczenie "nieznanych" mozna tez oprzec o zbyt duza niepewnosc zwracana jako "how_much".
        # Warto po przeszkoleniu modelu, wygenerować "typowe" wartosci niepewnosci dla predykcji obrazow testowych, aby
        # potem sugerowac sie nimi w celu stwierdzenia, czy niepewnosc jest dosc mala aby wskazanie konkretnej tozsamosci
        # bylo wiarygodne.
        dirs = os.listdir(data_folder_path)
        self.subjects = dirs
        faces = []
        labels = []
        label = -1
        for dir_name in dirs:
            # zakladamy, ze kazdy folder ma w nazwie tylko jedna litere 's' a po niej numer zatwierdzanej tozsamosci,
            # zastepujemy 's' => '' i uzyskujemy identyfikator tozsamosci.
            #label = int(dir_name.replace("s", ""))
            label = label + 1
            # skladanie sciezki do pliku ./training_images/s0 ...
            subject_dir_path = data_folder_path + "/" + dir_name
            # odczytujemy liste zdjec w lokacji wskazanej w linii wyzej
            subject_images_names = os.listdir(subject_dir_path)
            # petla po obrazach w celu detekcji twarzy, w przypadku jej wykrycia, dodanie owej do listy wykrytych, owa lista
            # posluzy do nauki modelu
            for image_name in subject_images_names:
                # ignorujemy pliki systemowe '.' '..'
                if image_name.startswith("."):
                    continue
                # skladamy sciezke do zdjecia z nazwa owego
                image_path = subject_dir_path + "/" + image_name
                # wczytanie owego do zmiennej
                image = cv2.imread(image_path)
                # wykrycie twarzy
                face = self.face_detector.detect_face(image)
                # jeśli w istocie ją wykryto, dodawanie do listy twarzy oraz jej identyfikatora
                if face is not None:
                    if eq:
                        face = cv2.resize(face, (256, 256))
                    faces.append(face)
                    labels.append(label)
        # zwroc twarze z identyfikatorami
        return faces, labels, self.subjects


    def train_model(self, path = "training_images"):
        if self.algorithm == 'lbph':
            faces, labels, subjects = self.prepare_training_data(path, eq=0)
        else:
            faces, labels, subjects = self.prepare_training_data(path, eq=1)
        self.face_recognizer.train(faces, np.array(labels))
        # ts = datetime.datetime.now()
        # ipdb.set_trace();
        # date_str = "{}".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        self.face_recognizer.save('models/' + self.algorithm + '_model' + '.xml')
        with open('models/' + self.algorithm + '_subjects.csv', "w") as file:
            for n in subjects:
                file.write(n)
                file.write(',')

    # rozpoznawanie osoby na zdjeciu i podpisywanie
    def predict(self, img, eq):
        #img = test_img.copy() # kopia obrazu
        # face = self.face_detector.detect_face(img) # wykrywanie twarzy
        #label, how_much = self.face_recognizer.predict(img) # rozpoznawanie
        #label_text = self.subjects[label] # odszukanie tozsamosci po identyfikatorze
        #return how_much, label_text # zwroc odpisany obraz, niepewnosc oraz tozsamosc
        img = img.copy()  # kopia obrazu
        face = self.face_detector.detect_face(img)  # wykrywanie twarzy
        if face is None:
            return None, None, None
        if eq:
            face = cv2.resize(face, (256, 256))
        label, how_much = self.face_recognizer.predict(face)  # rozpoznawanie
        label_text = self.subjects[label]  # odszukanie tozsamosci po identyfikatorze
        return img, how_much, label_text  # zwróć podpisany obraz, niepewnosc oraz tozsamosc
