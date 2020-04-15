import cv2
import os
import numpy as np

# lista rozpoznawalnych osob
subjects = ['kamil', 'nieznany1', 'kolegaAsi', 'michal']

# wykrywanie twarzy i zwrocenie ROI w skali szarosci
def detect_face (img):
	# konwersja na skale szarosci
	gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
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
	faces = []
	labels = []
	for dir_name in dirs:
		# zakladamy, ze kazdy folder ma w nazwie tylko jedna litere 's' a po niej numer zatwierdzanej tozsamosci,
		# zastepujemy 's' => '' i uzyskujemy identyfikator tozsamosci.
		label = int(dir_name.replace("s", ""))
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
	return faces, labels


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

# przygotowanie danych do uczenia
faces, labels = prepare_training_data("training_images")

# utworzenie obiektu klasyfikatora LBPH
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
# szkolenie modelu
face_recognizer.train(faces, np.array(labels))

# obrazy testowe
test_img1 = cv2.imread("test_images/test1_kamil.jpg")
test_img2 = cv2.imread("test_images/test1_michal.jpg")
test_img3 = cv2.imread("test_images/jakasBaba1.jpg")
test_img4 = cv2.imread("test_images/lysyfacet.jpeg")
test_img5 = cv2.imread("test_images/test1_kolegaAsi.jpg")
test_img6 = cv2.imread("test_images/test1_nieznany1.jpg")
# sprawdzenie
predicted_img1, how_much1, who1 = predict(test_img1)
predicted_img2, how_much2, who2 = predict(test_img2)
predicted_img3, how_much3, who3 = predict(test_img3)
predicted_img4, how_much4, who4 = predict(test_img4)
predicted_img5, how_much5, who5 = predict(test_img5)
predicted_img6, how_much6, who6 = predict(test_img6)
print(f'kamil:		{how_much1:f},	rozpoznano:{who1:s}\n'
	  f'michal:		{how_much2:f},	rozpoznano:{who2:s}\n'
	  f'jakasBaba:	{how_much3:f},	rozpoznano:{who3:s}\n'
	  f'lysyFacet:	{how_much4:f},	rozpoznano:{who4:s}\n'
	  f'kolegaAsi:	{how_much5:f},	rozpoznano:{who5:s}\n'
	  f'nieznany:	{how_much6:f},	rozpoznano:{who6:s}\n')
print("done!")


# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
# ax1.imshow(cv2.cvtColor(predicted_img1, cv2.COLOR_BGR2RGB))
# ax2.imshow(cv2.cvtColor(predicted_img2, cv2.COLOR_BGR2RGB))
# cv2.imshow("Czy to Kamil", predicted_img1)
# cv2.imshow("Czy to kolega Asi", predicted_img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(1)
# cv2.destroyAllWindows()
