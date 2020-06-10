import cv2
import os
import numpy as np
import ipdb
import datetime

# wykrywanie twarzy i zwrocenie ROI w skali szarosci
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
def prepare_training_data(data_folder_path, eq = 0):
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
		# label = int(dir_name.replace("s", ""))
		label = label + 1 # nie zawsze foldery to s0 s1.. teraz sa roznie, s17, s10, na dodatek python sortuje je inaczej niz nakazuje rzeczywisty porzadek.
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
				if eq:
					face = cv2.resize(face, (256, 256))
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
def predict(test_img, face_recognizer, subjects, eq = 0):
	img = test_img.copy() # kopia obrazu
	face, rect = detect_face(img) # wykrywanie twarzy
	if face is None:
		return None, None, None
	if eq:
		face = cv2.resize(face, (256, 256))
	label, how_much = face_recognizer.predict(face) # rozpoznawanie
	label_text = subjects[label] # odszukanie tozsamosci po identyfikatorze
	# obrysowanie i podpisanie
	draw_rectangle(img, rect)
	draw_text(img, label_text, rect[0], rect[1]-5)
	return img, how_much, label_text # zwróć podpisany obraz, niepewnosc oraz tozsamosc
