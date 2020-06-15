import SWiMwB as FD

fd = FD.Face_detector() # obiekt detektora
fd.get_photo() # pobieranie zdjecia z kamerki
fd.detect_face() # wykrywanie twarzy
print(fd.face)
if fd.face is not None:
    fr = FD.Face_recognitor('lbph')

    #fr.read_model('models/lbph/model_BioID_LBPH-2020-06-13_14-32-22.xml','models/lbph/subjects_BioID_LBPH-2020-06-13_14-32-22.csv')
    fr.train_model()
    (how_much, label) = fr.predict(fd.face)
    print('wykryto: ' + label)

else:
    print('nie wykryto twarzy!')

print('done!')