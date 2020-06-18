import SWiMwB as s
import tkinter as tki
import cv2
import threading
from PIL import Image, ImageTk


class RecognitionApp:

    def __init__(self):
        self.detector = s.Face_detector()
        self.recognitor = s.Face_recognitor()
        self.identities = {}
        self.toggle_camera = 1
        self.webcam = cv2.VideoCapture(0)

        # ustawienia okna
        self.window = tki.Tk()
        self.window.wm_title("Prototyp aplikacji systemu biometrycznego")
        self.window.config(background="#FF44FF")
        self.window.protocol("WM_DELETE_WINDOW", self.onClose)

        self.frame = tki.Frame(self.window, width=1000, height=500)
        self.frame.grid(row=0,column=0,columnspan=10,rowspan=10)

        self.camera_label = tki.Label(self.window)
        self.camera_label.grid(row=1,column=1,columnspan=9,rowspan=9)

        # elementy GUI
        button_camera_on_of = tki.Button(self.window, text="Turn on the camera",
                                         command=self.buttonCamera)
        button_camera_on_of.grid(row=1, column=6, columnspan=2, rowspan=2, sticky='nsew')
        button_camera_on_of = tki.Button(self.window, text="Turn off the camera",
                                         command=self.turnOffCamera)
        button_camera_on_of.grid(row=4, column=6, columnspan=2, rowspan=2, sticky='nsew')
        button_get_image = tki.Button(self.window, text="Get the image",
                                         command=self.getImage)
        button_get_image.grid(row=7, column=6, columnspan=2, rowspan=2, sticky='nsew')


    def turnOnCamera(self):
        self.webcam = cv2.VideoCapture(0)
        check, image = self.webcam.read()
        if check:
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.camera_label.image = image
            self.camera_label.config(image=image)
        self.window.after(30, self.turnOnCamera)


    def turnOffCamera(self):
        if self.turnOnCamera is not None:
            self.window.after_cancel(self.turnOnCamera)
            # self.turnOnCamera = None
        self.webcam.release()
        self.toggle_camera = self.toggle_camera * (-1)


    def buttonCamera(self):
        if self.toggle_camera>0:
            self.toggle_camera = self.toggle_camera * (-1)
            self.turnOnCamera()


    def getImage(self):
        if self.turnOnCamera is not None:
            self.window.after_cancel(self.turnOnCamera)
            self.turnOnCamera = None
        check1, image1 = self.webcam.read()
        cv2.imwrite(filename="images/image_object_" + str(id(check1)) + '.jpg', img=image1)


    def onClose(self):
            print("[INFO] closing...")
            self.turnOffCamera()
            self.window.quit()
            self.window.destroy()


appli = RecognitionApp()
appli.window.mainloop()