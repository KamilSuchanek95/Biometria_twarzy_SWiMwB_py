import SWiMwB as FD

fd = FD.Face_detector()
fd.get_photo()
print(fd.face_cascade)
fd.detect_face()
fd.show_detected_faces()


print('done!')