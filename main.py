#!/usr/bin/python3

#new
from tkinter.filedialog import LoadFileDialog
from supportmodule import *

import detector_and_recognitor as s
import tkinter as tki
from tkinter import messagebox, ttk
from functools import partial
import cv2
import datetime
from PIL import Image, ImageTk
import numpy as np
import ipdb as i

class RecognitionApp:

    def __init__(self, root, img):
        # globals
        self.detector = s.Face_detector()
        self.recognizer = s.Face_recognitor()
        self.subjects= []
        self.identities = {}
        self.p_vals = {}
        self.toggle_camera = 1
        self.webcam = cv2.VideoCapture(0)
        self.off = 0
        self.current_image = img
        self.init_image = img
        
        # ustawienia okna aplikacji
        self.window = root
        self.window.wm_title("Prototyp aplikacji systemu biometrycznego")
        self.window.config(background="#004890")
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)    

        # ustawienia paneli aplikacji    
        self.tab_control = ttk.Notebook(self.window)
        self.recognition_tab = ttk.Frame(self.tab_control)
        self.rec_configuration_tab = ttk.Frame(self.tab_control)
        self.temp_configuration_tab = ttk.Frame(self.tab_control)
        self.ecg_configuration_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.recognition_tab, text='Recognition panel')
        self.tab_control.add(self.rec_configuration_tab, text='Face recognition configuration panel')
        self.tab_control.add(self.temp_configuration_tab, text='Temperature verification configuration panel')
        self.tab_control.add(self.ecg_configuration_tab, text='ECG recognition configuration panel')
        self.tab_control.grid(column=0,row=0)
        """ Kontrolki w panelu "Recognition panel" """
        """# podgląd kamerki"""
        self.camera_label = tki.Label(self.recognition_tab, image = img)
        self.camera_label.grid(row=0,column=0, sticky='nsew')
        """# rozdzielacz podglądu od przycisków"""
        self.sep = ttk.Separator(self.recognition_tab, orient="vertical")
        self.sep.grid(column=1, sticky='ns', columnspan=2)
        """# ramka przycisków, aby grupowala je w pionie, nadal w jednym wierszu, który również zawiera podgląd"""
        self.frame_controls = tki.Frame(self.recognition_tab, background="#F55255")
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
        """ Kontrolki w panelu "Face recognition configuration panel" """
        self.frame_training = tki.Frame(self.rec_configuration_tab, background="#F50055")
        self.frame_training.grid(row=0, column=0,columnspan=3, sticky='nsew')
        # training path settings
        self.training_images_label = tki.Label(self.frame_training, text="Training images path: ")
        self.training_images_label.grid(row=0, column=0, columnspan=2, sticky='nsew')
        self.training_path_entry = tki.Entry(self.frame_training, borderwidth=4, relief='ridge')
        self.training_path_entry.insert(tki.END, "for example: .../training_images")
        self.training_path_entry.grid(row=1, column=0, columnspan=3, sticky='nsew')
        self.training_path_select_button = tki.Button(self.frame_training, command=partial(self.get_path, self.training_path_entry), text='Select path\nor entry below:', relief='solid')
        self.training_path_select_button.grid(row=0, column=2,sticky='nsew')
        """# testing path settings"""
        self.testing_images_label = tki.Label(self.frame_training, text="Testing images path: ")
        self.testing_images_label.grid(row=2, column=0, columnspan=2, sticky='nsew')
        self.testing_path_entry = tki.Entry(self.frame_training, text = '', borderwidth=4, relief='ridge'); self.testing_path_entry.insert(tki.END, "for example: .../testing_images")
        self.testing_path_entry.grid(row=3, column=0, columnspan=3, sticky='nsew')
        self.testing_path_select_button = tki.Button(self.frame_training, command=partial(self.get_path, self.testing_path_entry), text='Select path\nor entry below:', relief='solid')
        self.testing_path_select_button.grid(row=2, column=2,sticky='nsew')
        """# select algorithm"""
        self.alg_var = tki.StringVar(self.frame_training, 'lbph')
        self.alg_lbph_radio = tki.Radiobutton(self.frame_training, text="LBPH", command=self.radio_checked, variable = self.alg_var, val = 'lbph')
        self.alg_lbph_radio.grid(row=4, column=0, sticky='nsew')
        self.alg_fish_radio = tki.Radiobutton(self.frame_training, text="Fisherface", command=self.radio_checked, variable = self.alg_var, val = 'fisherface')
        self.alg_fish_radio.grid(row=4, column=1, sticky='nsew')
        self.alg_eigen_radio = tki.Radiobutton(self.frame_training, text="Eigenface", command=self.radio_checked, variable = self.alg_var, val = 'eigenface')
        self.alg_eigen_radio.grid(row=4, column=2, sticky='nsew')
        """# train model or load model """
        self.create_model_button = tki.Button(self.frame_training, command=self.create_model, text='If You selected necessary paths\nClick and train model', relief='solid')
        self.create_model_button.grid(row=5, column=0, columnspan=2, sticky='nsew')
        self.create_model_button = tki.Button(self.frame_training, command=self.manually_load_model, text='Or load model\nfrom file', relief='solid')
        self.create_model_button.grid(row=5, column=2, columnspan=1, sticky='nsew')
        """# classes of resulting model"""
        self.resulting_classes_label = tki.Message(self.frame_training, 
                                                   text='''Replace the class names "s0, s1, ..." with real identity names by selecting the folder name, 
                                                           e.g. "s0" and entering "Adam Kowalski" instead, without removing a commas, each identity name 
                                                           must be exactly on the position of the folder name being replaced. 
                                                        
                                                           So, having a list: "s0, s1, s10", assuming that s0 is assigned to the identity "Adam Kowalski", 
                                                           s1 to "Marta Brzdż" and s10 to "Lucyna Puf", then the text "s0, s1, s10" must be replaced by 
                                                           "Adam Kowalski, Marta Brzdż, Lucyna Puf" :''')
        self.resulting_classes_label.grid(row=6, column=0, columnspan=2, sticky='nsew')
        self.resulting_class_entry = tki.Entry(self.frame_training, borderwidth=4, relief='ridge'); self.resulting_class_entry.insert(tki.END, "for example: s0,s1,s10,s11,s12,s2,s3")
        self.resulting_class_entry.grid(row=7, column=0, columnspan=3, sticky='nsew')
        self.confirmation_classes_button = tki.Button(self.frame_training, text='Confirm the\nchanged list', relief='solid', command=self.create_identity_classes)
        self.confirmation_classes_button.grid(row=6, column=2,sticky='nsew')
        """# save configurations"""
        self.save_configuration_button = tki.Button(self.frame_training, text='Save configuration', relief='solid', command=self.confirm_model)
        self.save_configuration_button.grid(row=8, column=0, columnspan=3, sticky='nsew')

        self.check_if_program_is_ready()


    def check_if_program_is_ready(self):
        with open(PROGRAM_STATE_FILE_PATH, 'r') as f: r = f.read(); is_set_and_algorithm = r.split(',')
        if float(is_set_and_algorithm[0]) < 1:
            messagebox.showinfo('Start program',
'''You must configure the program!
Go to the face recognition configuration panel 
and train / load the model''')
        else:
            self.alg_var.set(is_set_and_algorithm[1])
            self.load_model_first_time(is_set_and_algorithm[1])

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
            self.window.after(30, self.turn_on_camera)

    def button_camera(self):
        if self.toggle_camera > 0:
            self.toggle_camera = self.toggle_camera * (-1)
            self.turn_on_camera()

    def turn_off_camera(self):
        # if camera is turned on
        if self.toggle_camera < 0:
            self.toggle_camera = self.toggle_camera * (-1)
            # turn off loop
            self.window.after_cancel(self.turn_on_camera)
            self.webcam.release()
            self.off = 1
            self.camera_label.config(image = self.init_image)

    def get_image(self):
        # stop the streaming
        # self.window.after_cancel(self.turnOnCamera)
        # get a photo and save
        # check1, image1 = self.webcam.read()
        ts = datetime.datetime.now()  # grab the current timestamp
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        filename = "images/" + self.identity_entry.get() + '_' + filename
        cv2.imwrite(filename=filename, img=self.current_image) # "images/image_object_" + str(id(check1)) + '.jpg'
        # check, that streaming is off
        # self.off = 1
        messagebox.showinfo('Message title', 'The image was saved in ' + filename)

    def on_close(self):
        print("[INFO] closing...")
        self.turn_off_camera()
        self.window.quit()
        self.window.destroy()

    def recognize_face(self):
        img, p_val, subject = self.recognizer.predict(self.current_image)
        if img is None:
            messagebox.showinfo('Recognition imformation', 'No face detected!')
        else:
            if self.identity_entry.get() == self.identities[subject] and self.p_vals[subject] >= p_val: # z wymaganą dokładnością...
                    messagebox.showinfo('Recognition result','You are really ' + self.identities[subject] + '.\nAcces allowed.')
            elif self.p_vals[subject] >= p_val:
                messagebox.showinfo('Recognition result', 'You are not ' + self.identity_entry.get() + ",\n  But you have been recognized as " + self.identities[subject] + ".\n Access denied.")
                print('p_val:\t\t' + str(p_val) + '\n' +
                      'self.p+val:\t\t' + str(self.p_vals[subject]) + '\n' +
                      'entry.ID:\t\t' + self.identity_entry.get() + '\n' +
                      'subject:\t\t' + subject + '\n' +
                      'self.ID:\t\t' + self.identities[subject])
            else:
                messagebox.showinfo('Recognition result','Not recognized approved person\n')
                print('p_val:\t\t' + str(p_val) + '\n' +
                      'self.p+val:\t\t' + str(self.p_vals[subject]) + '\n' +
                      'entry.ID:\t\t' + self.identity_entry.get() + '\n' +
                      'subject:\t\t' + subject + '\n' +
                      'self.ID:\t\t' + self.identities[subject])
            ts = datetime.datetime.now()
            filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            filename = "images/recognition_as_" + self.identity_entry.get() + '_recognized_as_' + self.identities[subject] + '_' + filename
            cv2.imwrite(filename=filename, img=self.current_image)

    """# Face recognition configuration metods"""

    def get_path(self, entry_handle):
        path = filedialog.askdirectory()
        entry_handle.delete(0, tki.END)
        entry_handle.insert(0, path)

    def create_identity_classes(self):
        fr = open('models/' + str(self.recognizer.algorithm) + '_parameters.csv', 'r')
        if len(fr.readline().split(',')) < 3:
            fr.close()
            identityList = self.resulting_class_entry.get().split(',')
            if(len(identityList)) == len(self.subjects):
                self.identities = dict(zip(self.subjects, identityList))
                if len(self.identities) < 1:
                    self.resulting_class_entry.insert(0, 'You must first train or load a model1!')
                else:
                    fr = open('models/' + str(self.recognizer.algorithm) + '_parameters.csv', 'r')
                    lines = fr.readlines()
                    fr.close()
                    if len(lines)==len(self.subjects):
                        for idx, el in enumerate(lines):
                            lines[idx] = lines[idx][0:-1] + ',' + self.identities[self.subjects[idx]]
                        fw = open('models/' + self.recognizer.algorithm + '_parameters.csv', 'w')
                        for el in lines:
                            fw.write(el + '\n')
                        fw.close()
                    else:
                        self.resulting_class_entry.insert(0, "Parameters file is wrong!")
            else:
                messagebox.showinfo('Error entered Ids list','Lengths of IDs list and subjects is not equal')

    def test_model(self, path): # create name_parameters.csv file with {subject,mean p, identity} columns
        p = []
        fw = open("models/" + self.recognizer.algorithm + "_parameters.csv", "w")
        for subject in os.listdir(path):
            #	if subject.startswith("."):
            #		continue;
            oo = os.listdir(path + '/' + subject)
            for t in oo:
                test_img = cv2.imread(path + '/' + subject + "/" + t)
                if self.recognizer.algorithm == 'lbph':
                    predicted_img, how_much, who = self.recognizer.predict(test_img)
                else:
                    predicted_img, how_much, who = self.recognizer.predict(test_img)
                if predicted_img is None:
                    continue
                if who == subject:
                    p.append(how_much)
            if len(p) > 0:
                fw.write(subject + ',' + str(np.max(p)) + '\n')
                self.p_vals.update({subject: np.max(p)})
            else:
                fw.write(subject + ',' + 0 + '\n')
                self.p_vals.update({subject: 0})
            p = []
        fw.close()

        print('koniec testow')

    def set_pvals_and_identities(self, parameters_path):
        with open(parameters_path, 'r') as f: lines = f.readlines()
        for l in lines:
            list = l.split(',')
            self.p_vals.update({l[0]: float(l[1].strip())})
            self.identities.update({l[0]: l[2].strip()})

    def update_subjects_text_entry(self, subjects_path):
        with open(subjects_path, "r") as f: lines = f.readlines
        for l in lines:
            self.subjects = l.split(',')[0:-1]
            self.resulting_class_entry.delete(0, tki.END)
            self.resulting_class_entry.insert(0, ','.join(self.subjects))

    def load_model(self, algorithm, model_paths):
        [model_path, parameters_path, subjects_path] = model_paths
        self.update_subjects_text_entry(subjects_path)
        self.set_pvals_and_identities(parameters_path)
        try:
            self.recognizer = s.Face_recognitor(algorithm)
            self.recognizer.read_model(model_path = model_path, subjects_path = subjects_path)
            messagebox.showinfo('Load successful', 'Model was loaded!')
        except:
            messagebox.showinfo('Load unsuccessful', 'Wrong selected files')
    
    def load_model_first_time(self, algorithm):
        # algorithm = str(self.alg_var.get())
        model_paths = get_model_data_from_resources(algorithm)
        self.load_model(algorithm, model_paths)

    def manually_load_model(self):
        algorithm = str(self.alg_var.get())
        model_paths = get_malually_model_data_paths(algorithm)
        self.load_model(algorithm, model_paths)
    
    def create_model(self):
        # create model object
        self.recognizer = s.Face_recognitor(self.alg_var.get())
        # train it
        self.recognizer.train_model(self.training_path_entry.get())
        # set subjects
        self.subjects = self.recognizer.subjects
        # entry classes
        self.resulting_class_entry.delete(0, tki.END)
        self.resulting_class_entry.insert(0, ','.join(self.subjects))
        # test model
        self.test_model(self.testing_path_entry.get())

    def radio_checked(self): # unnecessary
        print(self.alg_var.get())

    def confirm_model(self):
        messagebox.showinfo('Complete the face recognition configuration','The configuration has been saved, do not manipulate files inside the "Recognition Systems" and "models" folders.')
        with open('Recognition Systems/set.txt', 'w') as file:
            file.write('1,' + str(self.alg_var.get()))





def start_application():
    root = tki.Tk() # create a GUI object with Tk
    default_image = ImageTk.PhotoImage(Image.open(DEFAULT_IMAGE_PATH))
    appli = RecognitionApp(root, default_image) # create RecognitionApp's instance 
    appli.window.mainloop() # start application

create_resources_if_they_dont_exists()

start_application()

