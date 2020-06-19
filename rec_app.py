import SWiMwB as s
import tkinter as tki
from tkinter import messagebox, ttk, filedialog
from functools import partial
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

        # ustawienia okna aplikacji
        self.window = root
        self.window.wm_title("Prototyp aplikacji systemu biometrycznego")
        self.window.config(background="#004890")
        self.window.protocol("WM_DELETE_WINDOW", self.onClose)
        # ustawienia paneli aplikacji
        self.tab_control = ttk.Notebook(self.window)
        self.recognition_tab = ttk.Frame(self.tab_control)
        self.rec_configuration_tab = ttk.Frame(self.tab_control)
        self.temp_configuration_tab = ttk.Frame(self.tab_control)
        self.ecg_configuration_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.recognition_tab, text='Recognition panel')
        self.tab_control.add(self.rec_configuration_tab, text='Face recognition configuration panel')
        self.tab_control.add(self.temp_configuration_tab, text='Temperature verification configudration panel')
        self.tab_control.add(self.ecg_configuration_tab, text='ECG recognition configuration panel')
        self.tab_control.grid(column=0,row=0)

        """ Kontrolki w panelu "Recognition panel" """
        # podgląd kamerki
        self.camera_label = tki.Label(self.recognition_tab, image = img)
        self.camera_label.grid(row=0,column=0, sticky='nsew')
        # rozdzielacz podglądu od przycisków
        self.sep = ttk.Separator(self.recognition_tab, orient="vertical")
        self.sep.grid(column=1, sticky='ns', columnspan=2)
        # ramka przycisków, aby grupowala je w pionie, nadal w jednym wierszu, który również zawiera podgląd
        self.frame_controls = tki.Frame(self.recognition_tab, background="#F55255")
        self.frame_controls.grid(row=0, column=3,columnspan=1, sticky='nsew')
        # kontrolki w panelu bocznym (ramce)
        self.identity_label = tki.Label(self.frame_controls, text = 'Enter Your identity: ', borderwidth=4, relief="groove")
        self.identity_label.grid(sticky='nsew')
        self.identity = tki.Entry(self.frame_controls, width=10, borderwidth=4, relief="solid")
        self.identity.grid(sticky='nsew')
        self.button_camera_on = tki.Button(self.frame_controls, text="Turn on the camera", command=self.buttonCamera, borderwidth=4, relief='ridge')
        self.button_camera_on.grid(sticky='nsew')
        self.button_camera_off = tki.Button(self.frame_controls, text="Turn off the camera", command=self.turnOffCamera, borderwidth=4, relief='ridge')
        self.button_camera_off.grid(sticky='nsew')
        self.button_get_image = tki.Button(self.frame_controls, text="Get the image", command=self.getImage, borderwidth=4, relief='ridge')
        self.button_get_image.grid(sticky='nsew')

        """ Kontrolki w panelu "Face recognition configuration panel" """
        self.frame_training = tki.Frame(self.rec_configuration_tab, background="#F50055")
        self.frame_training.grid(row=0, column=0,columnspan=3, sticky='nsew')
        self.training_images_label = tki.Label(self.frame_training, text="Training images path: ")
        self.training_images_label.grid(row=0, column=0, columnspan=2, sticky='nsew')
        self.training_path_entry = tki.Entry(self.frame_training, text='.../training_images', borderwidth=4, relief='ridge')
        self.training_path_entry.grid(row=1, column=0, columnspan=3, sticky='nsew')
        self.training_path_select_button = tki.Button(self.frame_training, command=partial(self.getPath, self.training_path_entry), text='Select path or entry below:')
        self.training_path_select_button.grid(row=0, column=2,sticky='nsew')
        self.training_path_entry = tki.Entry(self.frame_training, text='.../training_images', borderwidth=4, relief='ridge')
        self.training_path_entry.grid(row=1, column=0, columnspan=3, sticky='nsew')

        self.button_loadModel = tki.Button(self.frame_training, text='Load model', command=self.load_model, borderwidth=4, relief='ridge')
        self.button_loadModel.grid(row=4, column=3, rowspan=2, sticky='nsew')
        self.button_empty = tki.Button(self.frame_training, text='Empty', command=self.empty, borderwidth=4, relief='ridge')
        self.button_empty.grid(row=6, column=3, rowspan=2, sticky='nsew')


        """ Kontrolki w panelu "Temperature verification configuration panel" """

        """ Kontrolki w panelu "ECG recognition configuration panel" """



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

    def getPath(self, entry_handle):
        path = filedialog.askdirectory()
        entry_handle.delete(0, tki.END)
        entry_handle.insert(0, path)


root = tki.Tk()
img = ImageTk.PhotoImage(Image.open('init_image.jpg'))
appli = RecognitionApp(root, img)
appli.window.mainloop()