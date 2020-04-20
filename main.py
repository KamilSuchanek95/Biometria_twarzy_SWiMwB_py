import SWiMwB as FD

fd = FD.Face_detector() # obiekt detektora
fd.get_photo() # pobieranie zdjecia z kamerki
fd.detect_face() # wykrywanie twarzy
fd.show_detected_faces()


print('done!')