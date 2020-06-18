import SWiMwB as s
import tkinter as tki
from tkinter import messagebox, ttk
import cv2
import datetime
from PIL import Image, ImageTk


class RecognitionApp:

    def __init__(self, root, img):
        # globals
        self.detector = s.Face_detector()
        self.recognitor = s.Face_recognitor()
        self.identities = {}
        self.toggle_camera = 1
        self.webcam = cv2.VideoCapture(0)
        self.off = 0
        self.current_image = img
        self.init_image = img

        # settings of figure
        self.window = root #tki.Tk()
        self.window.wm_title("Prototyp aplikacji systemu biometrycznego")
        self.window.config(background="#004890")
        self.window.protocol("WM_DELETE_WINDOW", self.onClose)

        self.tab_control = ttk.Notebook(self.window)
        self.recognition_tab = ttk.Frame(self.tab_control)
        self.configuration_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.recognition_tab, text='Recognition panel')
        self.tab_control.add(self.configuration_tab, text='Configuration panel')
        self.tab_control.grid(column=0,row=0)

        self.frame = tki.Frame(self.recognition_tab, width=700, height=1000,background="#004890")
        self.frame.grid(row=0,column=0,rowspan=20)

        self.identity = tki.Entry(self.recognition_tab, width = 10, borderwidth=4, relief="solid")
        self.identity.grid(row=1, column=0, rowspan=1, sticky='nsew')

        self.identity_label = tki.Label(self.recognition_tab, text = 'Enter Your identity: ', borderwidth=4, relief="groove")
        self.identity_label.grid(row=0, column=0, rowspan=1, sticky='nsew', pady=2, padx=2)

        self.camera_label = tki.Label(self.recognition_tab, image = img)
        self.camera_label.grid(row=2,column=0, sticky='nsew')

        self.button_camera_on = tki.Button(self.recognition_tab, text="Turn on the camera", command=self.buttonCamera, borderwidth=4, relief='ridge')
        self.button_camera_on.grid(row=3, column=0, rowspan=2, sticky='nsew')
        self.button_camera_off = tki.Button(self.recognition_tab, text="Turn off the camera", command=self.turnOffCamera, borderwidth=4, relief='ridge')
        self.button_camera_off.grid(row=5, column=0, rowspan=2, sticky='nsew')
        self.button_get_image = tki.Button(self.recognition_tab, text="Get the image", command=self.getImage, borderwidth=4, relief='ridge')
        self.button_get_image.grid(row=7, column=0, rowspan=2, sticky='nsew')

        self.sep = ttk.Separator(self.recognition_tab, orient="vertical")
        self.sep.grid(column=1, sticky='ns', columnspan=2)

        self.button_temperature = tki.Button(self.recognition_tab, text="Temp", command=self.temperature, borderwidth=4,
                                     relief='solid', background="#F55255")
        self.button_temperature.grid(row=0, column=3, rowspan=1, sticky='nsew')
        self.button_ecg = tki.Button(self.recognition_tab, text="ECG", command=self.ecg, borderwidth=4,
                                     relief='solid', background="#904899")
        self.button_ecg.grid(row=1, column=3, rowspan=1, sticky='nsew')

        # configuration TopLevel app
        # self.top = tki.Toplevel(self.window)
        # self.top.wm_title("Configuracja rozpoznawania twarzy")
        # self.top.config(background="#004893")
        self.top_button_trainModel = tki.Button(self.configuration_tab, text='Train model', command=self.train_model, borderwidth=4, relief='ridge')
        self.top_button_trainModel.grid(row=2, column=3, rowspan=2, sticky='nsew')
        self.top_button_loadModel = tki.Button(self.configuration_tab, text='Load model', command=self.load_model, borderwidth=4, relief='ridge')
        self.top_button_loadModel.grid(row=4, column=3, rowspan=2, sticky='nsew')
        self.top_button_empty = tki.Button(self.configuration_tab, text='Empty', command=self.empty, borderwidth=4, relief='ridge')
        self.top_button_empty.grid(row=6, column=3, rowspan=2, sticky='nsew')


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
            self.camera_label.config(image = self.init_image)


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


    def temperature(self):
        1
    def ecg(self):
        1

    # configuration TopLevel metods
    def train_model(self):
        1

    def load_model(self):
        2

    def empty(self):
        3


root = tki.Tk()
img = ImageTk.PhotoImage(Image.open('init_image.jpg'))
appli = RecognitionApp(root, img)
appli.window.mainloop()