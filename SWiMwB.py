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


class Face_recognitor:

    def __init__(self):
        self.photo = []
        self.faces = []
 
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')



def detect_face (img):
    # konwersja na skale szarosci
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # załadowanie klasyfikatora
    face_cas = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # wykrycie twarzy
    faces = face_cas.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # jesli nie wykryto twarzy zwroc None
    if (len(faces) == 0):
        return None, None
    # koordynaty ROI
    (x, y, w, h) = faces[0]
    # zwroc ROI w skali szarości oraz koordynaty
    return gray[y: y+w, x: x+h], faces[0]


# przygotowanie wykrytych twarzy do uczenia modelu
def prepare_training_data(data_folder_path):
    # w naszym folderze training_images beda foldery s0 s1 s2 s3 dla kazdej osoby ktora chcemy rozpoznawac,
    # oraz dla nieznanych, ktorych mozna zdefiniowac jako "typowe" nieznane twarze z bardzo malym zbiorem uczacym.
    # ewentualne wykluczenie "nieznanych" mozna tez oprzec o zbyt duza niepewnosc zwracana jako "how_much".
    # Warto po przeszkoleniu modelu, wygenerować "typowe" wartosci niepewnosci dla predykcji obrazow testowych, aby
    # potem sugerowac sie nimi w celu stwierdzenia, czy niepewnosc jest dosc mala aby wskazanie konkretnej tozsamosci
    # bylo wiarygodne.
    dirs = os.listdir(data_folder_path)
    subjects = dirs
    faces = []
    labels = []
    label = -1;
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
                continue;
            # skladamy sciezke do zdjecia z nazwa owego
            image_path = subject_dir_path + "/" + image_name
            # wczytanie owego do zmiennej
            image = cv2.imread(image_path)
            # wykrycie twarzy
            face, rect = detect_face(image)
            # jeśli w istocie ją wykryto, dodawanie do listy twarzy oraz jej identyfikatora
            if face is not None:
                # dodanie ROI twarzy do listy
                faces.append(face)
                # dodanie identyfikatora twarzy
                labels.append(label)
    # zwroc twarze z identyfikatorami
    return faces, labels, subjects


def prepare_training_data_eigenfaces(data_folder_path):
    # w naszym folderze training_images beda foldery s0 s1 s2 s3 dla kazdej osoby ktora chcemy rozpoznawac,
    # oraz dla nieznanych, ktorych mozna zdefiniowac jako "typowe" nieznane twarze z bardzo malym zbiorem uczacym.
    # ewentualne wykluczenie "nieznanych" mozna tez oprzec o zbyt duza niepewnosc zwracana jako "how_much".
    # Warto po przeszkoleniu modelu, wygenerować "typowe" wartosci niepewnosci dla predykcji obrazow testowych, aby
    # potem sugerowac sie nimi w celu stwierdzenia, czy niepewnosc jest dosc mala aby wskazanie konkretnej tozsamosci
    # bylo wiarygodne.
    dirs = os.listdir(data_folder_path)
    subjects = dirs
    faces = []
    labels = []
    label = -1;
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
                continue;
            # skladamy sciezke do zdjecia z nazwa owego
            image_path = subject_dir_path + "/" + image_name
            # wczytanie owego do zmiennej
            image = cv2.imread(image_path)
            # wykrycie twarzy
            face, rect = detect_face(image)
            # jeśli w istocie ją wykryto, dodawanie do listy twarzy oraz jej identyfikatora
            if face is not None:
                face = cv2.resize(face, (100, 100))
                # dodanie ROI twarzy do listy
                faces.append(face)
                # dodanie identyfikatora twarzy
                labels.append(label)
    # zwroc twarze z identyfikatorami
    return faces, labels, subjects

# rysowanie kwadratu
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)


# podpisywanie kwadratu
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


# rozpoznawanie osoby na zdjeciu i podpisywanie
def predict(test_img):
    img = test_img.copy() # kopia obrazu
    face, rect = detect_face(img) # wykrywanie twarzy
    label, how_much = face_recognizer.predict(face) # rozpoznawanie

    label_text = subjects[label] # odszukanie tozsamosci po identyfikatorze
    # obrysowanie i podpisanie
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img, how_much, label_text # zwróć podpisany obraz, niepewnosc oraz tozsamosc


def predict_eigenfaces(test_img):
    img = test_img.copy() # kopia obrazu
    face, rect = detect_face(img) # wykrywanie twarzy
    face = cv2.resize(face, (100, 100))
    label, how_much = face_recognizer.predict(face) # rozpoznawanie
    label_text = subjects[label] # odszukanie tozsamosci po identyfikatorze
    # obrysowanie i podpisanie
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    return img, how_much, label_text # zwróć podpisany obraz, niepewnosc oraz tozsamosc


