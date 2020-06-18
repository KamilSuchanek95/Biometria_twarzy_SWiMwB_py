import SWiMwB as s
import tkinter as tki
from tkinter import messagebox
import cv2
import datetime
from PIL import Image, ImageTk


class RecognitionApp:

    def __init__(self):
        self.detector = s.Face_detector()
        self.recognitor = s.Face_recognitor()
        self.identities = {}
        self.toggle_camera = 1
        self.webcam = cv2.VideoCapture(0)
        self.off = 0
        self.current_image = []

        # ustawienia okna
        self.window = tki.Tk()
        self.window.wm_title("Prototyp aplikacji systemu biometrycznego")
        self.window.config(background="#0048FF")
        self.window.protocol("WM_DELETE_WINDOW", self.onClose)

        self.frame = tki.Frame(self.window, width=1000, height=500)
        self.frame.grid(row=0,column=0,columnspan=10,rowspan=10)

        self.identity = tki.Entry(self.window, width = 10)
        self.identity.grid(row=0, column=0, columnspan=1, rowspan=1, sticky='nsew')

        self.camera_label = tki.Label(self.window)
        self.camera_label.grid(row=1,column=0,columnspan=1,rowspan=1, sticky='nsew')

        # elementy GUI
        button_camera_on = tki.Button(self.window, text="Turn on the camera", command=self.buttonCamera)
        button_camera_on.grid(row=1, column=7, columnspan=4, rowspan=2, sticky='nsew')
        button_camera_off = tki.Button(self.window, text="Turn off the camera", command=self.turnOffCamera)
        button_camera_off.grid(row=4, column=7, columnspan=4, rowspan=2, sticky='nsew')
        button_get_image = tki.Button(self.window, text="Get the image", command=self.getImage)
        button_get_image.grid(row=7, column=7, columnspan=4, rowspan=2, sticky='nsew')


    def turnOnCamera(self):
        if self.off:
            self.webcam = cv2.VideoCapture(0)
            self.off = 0
        else:
            check, self.current_image = self.webcam.read()
            if check:
                image = Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))
                image = ImageTk.PhotoImage(image)
                self.camera_label.image = image
                self.camera_label.config(image=image)
            self.window.after(30, self.turnOnCamera)


    def buttonCamera(self):
        if self.toggle_camera > 0:
            self.toggle_camera = self.toggle_camera * (-1)
            self.turnOnCamera()


    def turnOffCamera(self):
        # if camera is turned on
        if self.toggle_camera < 0:
            self.toggle_camera = self.toggle_camera * (-1)
            # turn off loop
            self.window.after_cancel(self.turnOnCamera)
            self.webcam.release()
            self.off = 1


    def getImage(self):
        # stop the streaming
        # self.window.after_cancel(self.turnOnCamera)
        # get a photo and save
        # check1, image1 = self.webcam.read()
        ts = datetime.datetime.now()  # grab the current timestamp
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        filename = "images/" + self.identity.get() + '_' + filename
        cv2.imwrite(filename=filename, img=self.current_image) # "images/image_object_" + str(id(check1)) + '.jpg'
        # check, that streaming is off
        # self.off = 1
        messagebox.showinfo('Message title', 'The image was saved in ' + filename)


    def onClose(self):
            print("[INFO] closing...")
            self.turnOffCamera()
            self.window.quit()
            self.window.destroy()


appli = RecognitionApp()
appli.window.mainloop()