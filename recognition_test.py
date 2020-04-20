import cv2
import os
import numpy as np
import ipdb
import datetime
# lista rozpoznawalnych osob
# subjects = ['s0', 's1', 's10', 's17', 's2', 's4', 's5', 's8', 's9'] 

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
				# dodanie ROI twarzy do listy
				faces.append(face)
				# dodanie identyfikatora twarzy
				labels.append(label)
	# zwroc twarze z identyfikatorami
	return faces, labels, subjects


def prepare_training_data_equal(data_folder_path):
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
def predict(test_img):
	img = test_img.copy() # kopia obrazu
	face, rect = detect_face(img) # wykrywanie twarzy
	label, how_much = face_recognizer.predict(face) # rozpoznawanie

	label_text = subjects[label] # odszukanie tozsamosci po identyfikatorze
	# obrysowanie i podpisanie
	draw_rectangle(img, rect)
	draw_text(img, label_text, rect[0], rect[1]-5)
	return img, how_much, label_text # zwróć podpisany obraz, niepewnosc oraz tozsamosc


def predict_equal(test_img):
	img = test_img.copy() # kopia obrazu
	face, rect = detect_face(img) # wykrywanie twarzy
	face = cv2.resize(face, (256, 256))
	label, how_much = face_recognizer.predict(face) # rozpoznawanie
	label_text = subjects[label] # odszukanie tozsamosci po identyfikatorze
	# obrysowanie i podpisanie
	draw_rectangle(img, rect)
	draw_text(img, label_text, rect[0], rect[1]-5)
	return img, how_much, label_text # zwróć podpisany obraz, niepewnosc oraz tozsamosc



### Reading subjects from file:
# with open("models/subjects_METODA-DATA.csv", "r") as file:
# 	for f in file:
# 		subjects = f.split(',')
# 		subjects = subjects[0:-1]  

#########################################################
#-------------------------------------------------------#
#########################################################
### Sekcja przeznaczona dla metody Eigenfaces
## razem z szkoleniem:
# faces, labels, subjects = prepare_training_data_equal("training_images")
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer.train(faces, np.array(labels))
# ts = datetime.datetime.now()
# date_str = "{}".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
# face_recognizer.save('models/model_BioID_eigenfaces_256-' + date_str + '.xml')
# with open("models/subjects_eigenfaces-" + date_str + ".csv", "w") as file:
# 	for n in subjects:
# 		file.write(n)
# 		file.write(',')
## bez szkolenia:
# subjects = os.listdir("training_images")
# face_recognizer = cv2.face.EigenFaceRecognizer_create()
# face_recognizer.read('model_BioID_eigenfaces_256-DATA.xml')
#########################################################
#-------------------------------------------------------#
#########################################################
### Sekcja przeznaczona dla metody LBPH
## razem z szkoleniem:
# faces, labels, subjects = prepare_training_data("training_images")
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.train(faces, np.array(labels))
# ts = datetime.datetime.now()
# date_str = "{}".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
# face_recognizer.save('models/model_BioID_LBPH-' + date_str + '.xml')
# with open("models/subjects_BioID_LBPH-" + date_str + ".csv", "w") as file:
# 	for n in subjects:
# 		file.write(n)
# 		file.write(',')
## bez szkolenia:
# subjects = os.listdir("training_images")
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# face_recognizer.read('model_BioID_LBPH-DATA.xml')
#########################################################
#-------------------------------------------------------#
#########################################################
### Sekcja przeznaczona dla metody Fisherfaces
## razem z szkoleniem:
faces, labels, subjects = prepare_training_data_equal("training_images")
face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
ts = datetime.datetime.now()
date_str = "{}".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
face_recognizer.save('models/model_BioIDFisherfaces_256-' + date_str + '.xml')
with open("models/subjects_BioIDFisherfaces-" + date_str + ".csv", "w") as file:
	for n in subjects:
		file.write(n)
		file.write(',')
## bez szkolenia:
# subjects = os.listdir("training_images")
# face_recognizer = cv2.face.FisherFaceRecognizer_create()
# face_recognizer.read('model_BioID_Fisherfaces_256-DATA.xml')


### testy automatyczne dla metod Fisherfaces i Eigenfaces:
for subject in os.listdir('test_images'):
	if subject.startswith("."):
		continue;
	test_img = cv2.imread("test_images/" + subject)
	predicted_img, how_much, who = predict_equal(test_img)
	print(f'{subject:s}:	{how_much:f},	rozpoznano:{who:s}\n')
	# cv2.imshow("Kto to?", predicted_img)
	# cv2.waitKey(0)
print('koniec testow')

### testy automatyczne dla metody LPBH:
# for subject in os.listdir('test_images'):
# 	if subject.startswith("."):
# 		continue;
# 	test_img = cv2.imread("test_images/" + subject)
# 	predicted_img, how_much, who = predict(test_img)
# 	print(f'{subject:s}:	{how_much:f},	rozpoznano:{who:s}\n')
# 	# cv2.imshow("Kto to?", predicted_img)
# 	# cv2.waitKey(0)
# print('koniec testow')