
from gui_helper import *
import cv2
from PIL import Image, ImageTk
import tkinter as tki
from tkinter import messagebox
import datetime


class RecognitionFrame(tki.Frame):


    def __init__(self, container, controller):
        super().__init__(container)
        self.controller = controller

        self.toggle_camera = 1
        self.webcam = cv2.VideoCapture(0)
        self.off = 0
        self.current_image = ImageTk.PhotoImage(Image.open(DEFAULT_IMAGE_PATH))
        self.init_image = self.current_image


        """ Kontrolki w panelu "Recognition panel" """
        """# podgląd kamerki"""
        self.camera_label = tki.Label(self, image = self.current_image)
        self.camera_label.grid(row=0,column=0, sticky='nsew')
        """# rozdzielacz podglądu od przycisków"""

        """# ramka przycisków, aby grupowala je w pionie, nadal w jednym wierszu, który również zawiera podgląd"""
        self.frame_controls = tki.Frame(self, background="#F55255")
        self.frame_controls.grid(row=0, column=3,columnspan=1, sticky='nsew')
        """# kontrolki w panelu bocznym (ramce)"""
        self.identity_label = tki.Label(self.frame_controls, text = 'Enter Your identity: ', borderwidth=4, relief="groove")
        self.identity_label.grid(sticky='nsew')
        self.identity_entry = tki.Entry(self.frame_controls, width=10, borderwidth=4, relief="solid")
        self.identity_entry.grid(sticky='nsew')
        self.button_camera_on = tki.Button(self.frame_controls, text="Turn on the camera", command=self.button_camera, borderwidth=4, relief='ridge')
        self.button_camera_on.grid(sticky='nsew')
        self.button_camera_off = tki.Button(self.frame_controls, text="Turn off the camera", command=self.turn_off_camera, borderwidth=4, relief='ridge')
        self.button_camera_off.grid(sticky='nsew')
        self.button_get_image = tki.Button(self.frame_controls, text="Get the image", command=self.get_image, borderwidth=4, relief='ridge')
        self.button_get_image.grid(sticky='nsew')
        self.button_recognize = tki.Button(self.frame_controls, text="Recognize the face", command=self.recognize_face, borderwidth=4, relief='ridge')
        self.button_recognize.grid(sticky='nsew')


    def turn_on_camera(self):
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
            self.controller.after(30, self.turn_on_camera)

    def button_camera(self):
        if self.toggle_camera > 0:
            self.toggle_camera = self.toggle_camera * (-1)
            self.turn_on_camera()

    def turn_off_camera(self):
        # if camera is turned on
        if self.toggle_camera < 0:
            self.toggle_camera = self.toggle_camera * (-1)
            # turn off loop
            self.controller.after_cancel(self.turn_on_camera)
            self.webcam.release()
            self.off = 1
            self.camera_label.config(image = self.init_image)

    def get_image(self):
        # stop the streaming
        # self.window.after_cancel(self.turnOnCamera)
        # get a photo and save
        # check1, image1 = self.webcam.read()
        ts = datetime.datetime.now()  # grab the current timestamp
        filename = os.path.join(IMAGES_PATH, self.identity_entry.get() + '_' + "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S")))
        cv2.imwrite(filename=filename, img=self.current_image) # "images/image_object_" + str(id(check1)) + '.jpg'
        # check, that streaming is off
        # self.off = 1
        messagebox.showinfo('Message title', 'The image was saved in ' + filename)

    def recognize_face(self):
        img, p_val, subject = self.controller.recognizer.predict(self.current_image)
        if img is None:
            messagebox.showinfo('Recognition imformation', 'No face detected!')
        else:
            if self.identity_entry.get() == self.controller.identities[subject] and self.controller.eukli_distances[subject] >= p_val: # z wymaganą dokładnością...
                    messagebox.showinfo('Recognition result','You are really ' + self.controller.identities[subject] + '.\nAcces allowed.')
            elif self.controller.eukli_distances[subject] >= p_val:
                messagebox.showinfo('Recognition result', 'You are not ' + self.identity_entry.get() + ",\n  But you have been recognized as " + self.controller.identities[subject] + ".\n Access denied.")
                print('p_val:\t\t' + str(p_val) + '\n' +
                      'self.p+val:\t\t' + str(self.controller.eukli_distances[subject]) + '\n' +
                      'entry.ID:\t\t' + self.identity_entry.get() + '\n' +
                      'subject:\t\t' + subject + '\n' +
                      'self.ID:\t\t' + self.controller.identities[subject])
            else:
                messagebox.showinfo('Recognition result','Not recognized approved person\n')
                print('p_val:\t\t' + str(p_val) + '\n' +
                      'self.p+val:\t\t' + str(self.controller.eukli_distances[subject]) + '\n' +
                      'entry.ID:\t\t' + self.identity_entry.get() + '\n' +
                      'subject:\t\t' + subject + '\n' +
                      'self.ID:\t\t' + self.controller.identities[subject])
            ts = datetime.datetime.now()
            filename_date = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            filename = "recognition_as_" + self.identity_entry.get() + '_recognized_as_' + self.controller.identities[subject] + '_' + filename_date
            image_path = os.path.join(IMAGES_PATH, filename)
            cv2.imwrite(filename=image_path, img=self.current_image)
