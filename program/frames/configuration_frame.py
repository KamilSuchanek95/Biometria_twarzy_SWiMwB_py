
from program.helpers.program_helper import *
from program.helpers.gui_helper import *
from program.tools.detector_and_recognitor import *

import os
import glob
import cv2
from functools import partial
import tkinter as tki
from tkinter import messagebox

class ConfigurationFrame(tki.Frame):


    def __init__(self, container, controller):
        super().__init__(container)
        self.controller = controller


        """ Kontrolki w panelu "Face recognition configuration panel" """
        """ # training path settings """
        self.training_images_label = tki.Label(self, text="Training images path: ")
        self.training_images_label.grid(row=0, column=0, columnspan=2, sticky='nsew')
        self.training_path_entry = tki.Entry(self, borderwidth=4, relief='ridge')
        self.training_path_entry.insert(tki.END, "for example: .../training_images")
        self.training_path_entry.grid(row=1, column=0, columnspan=3, sticky='nsew')
        self.training_path_select_button = tki.Button(self, command=partial(self.get_path, self.training_path_entry), text='Select path\nor entry below:', relief='solid')
        self.training_path_select_button.grid(row=0, column=2,sticky='nsew')
        """# testing path settings"""
        self.testing_images_label = tki.Label(self, text="Testing images path: ")
        self.testing_images_label.grid(row=2, column=0, columnspan=2, sticky='nsew')
        self.testing_path_entry = tki.Entry(self, text = '', borderwidth=4, relief='ridge'); self.testing_path_entry.insert(tki.END, "for example: .../testing_images")
        self.testing_path_entry.grid(row=3, column=0, columnspan=3, sticky='nsew')
        self.testing_path_select_button = tki.Button(self, command=partial(self.get_path, self.testing_path_entry), text='Select path\nor entry below:', relief='solid')
        self.testing_path_select_button.grid(row=2, column=2,sticky='nsew')
        """# select algorithm"""
        self.alg_var = tki.StringVar(self, 'lbph')
        self.alg_lbph_radio = tki.Radiobutton(self, text="LBPH", command=self.radio_checked, variable = self.alg_var, val = 'lbph')
        self.alg_lbph_radio.grid(row=4, column=0, sticky='nsew')
        self.alg_fish_radio = tki.Radiobutton(self, text="Fisherface", command=self.radio_checked, variable = self.alg_var, val = 'fisherface')
        self.alg_fish_radio.grid(row=4, column=1, sticky='nsew')
        self.alg_eigen_radio = tki.Radiobutton(self, text="Eigenface", command=self.radio_checked, variable = self.alg_var, val = 'eigenface')
        self.alg_eigen_radio.grid(row=4, column=2, sticky='nsew')
        """# train model or load model """
        self.create_model_button = tki.Button(self, command=self.create_model, text='If You selected necessary paths\nClick and train model', relief='solid')
        self.create_model_button.grid(row=5, column=0, columnspan=2, sticky='nsew')
        self.create_model_button = tki.Button(self, command=self.load_model, text='Or load model\nfrom file', relief='solid')
        self.create_model_button.grid(row=5, column=2, columnspan=1, sticky='nsew')
        """# classes of resulting model"""
        self.resulting_classes_label = tki.Message(self, text=INSTRUCTION_CLASSES_LABEL)
        self.resulting_classes_label.grid(row=6, column=0, columnspan=2, sticky='nsew')
        self.resulting_class_entry = tki.Entry(self, borderwidth=4, relief='ridge'); self.resulting_class_entry.insert(tki.END, "for example: s0,s1,s10,s11,s12,s2,s3")
        self.resulting_class_entry.grid(row=7, column=0, columnspan=3, sticky='nsew')
        self.confirmation_classes_button = tki.Button(self, text='Confirm the\nchanged list', relief='solid', command=self.create_identity_classes)
        self.confirmation_classes_button.grid(row=6, column=2,sticky='nsew')
        """# save configurations"""
        self.save_configuration_button = tki.Button(self, text='Save configuration', relief='solid', command=self.confirm_model)
        self.save_configuration_button.grid(row=8, column=0, columnspan=3, sticky='nsew')


    def check_if_program_is_ready(self):
        with open(PROGRAM_STATE_FILE_PATH, 'r') as f: r = f.read(); is_set_and_algorithm = r.split(',')
        if float(is_set_and_algorithm[0]) < 1:
            messagebox.showinfo('Start program', INFO_MUST_CONFIGURE)
        else:
            self.alg_var.set(is_set_and_algorithm[1])
            self.load_model(is_set_and_algorithm[1])
    
    """# Face recognition configuration metods"""

    def get_path(self, entry_handle):
        path = filedialog.askdirectory()
        entry_handle.delete(0, tki.END)
        entry_handle.insert(0, path)


    def create_identity_classes(self):
        parameters_path = get_model_data_paths_for_algorithm(self.controller.recognizer.algorithm)[1]
        with open(parameters_path, 'r') as f: columns_number = len(f.readline().split(','))
        if columns_number < 3: # dopisz trzecią kolumnę, jeśli jej nie ma.
            identityList = self.resulting_class_entry.get().split(',')
            if(len(identityList)) == len(self.controller.recognizer.subjects):
                self.controller.identities = dict(zip(self.controller.recognizer.subjects, identityList))
                if len(self.controller.identities) < 1:
                    self.resulting_class_entry.insert(0, 'You must first train or load a model1!')
                else:
                    with open(parameters_path, 'r') as f: lines = f.readlines()
                    if len(lines)==len(self.controller.recognizer.subjects):
                        new_lines = []
                        for idx, line in enumerate(lines):
                            new_lines.append(','.join([line.strip(), self.controller.identities[self.controller.recognizer.subjects[idx]]]) + "\n")
                        with open(parameters_path, 'w') as f: f.writelines(new_lines)
                    else:
                        self.resulting_class_entry.insert(0, "Parameters file is wrong!")
            else:
                messagebox.showinfo('Error entered Ids list','Lengths of IDs list and subjects is not equal')


    def create_parameters_file(self, parameters_path, lines):
        with open(parameters_path, "w") as p: p.writelines(lines)

    def test_model(self, test_images_path): # create name_parameters.csv file with {subject,mean p, identity} columns
        parameters_path = get_model_data_paths_for_algorithm(self.controller.recognizer.algorithm)[1]
        parameters_lines_to_save_later = []
        dirs_test_images = listdir_with_glob(test_images_path)
        for subject_images_dir in dirs_test_images:
            eukli_distances = []
            subject_images = listdir_with_glob(path(test_images_path, subject_images_dir))
            for image in subject_images:
                test_img = cv2.imread(path(subject_images_dir, image))
                predicted_img, eukli_distance, who = self.controller.recognizer.predict(test_img)
                if predicted_img is None: #if recognized nothing
                    continue
                if who == subject_images_dir: # if recognized correctly
                    eukli_distances.append(eukli_distance)
            if len(eukli_distances) > 0:
                parameters_lines_to_save_later.append(subject_images_dir + ',' + str(max(eukli_distances)) + '\n')
                self.controller.eukli_distances.update({subject_images_dir: max(eukli_distances)})
            else:
                parameters_lines_to_save_later.append(subject_images_dir + ',' + '0' + '\n')
                self.controller.eukli_distances.update({subject_images_dir: 0})
        self.create_parameters_file(parameters_path, parameters_lines_to_save_later)
        print('End of tests.')

    def set_eukli_distances_and_identities(self, parameters_path):
        with open(parameters_path, 'r') as f: lines = f.readlines()
        for line in lines:
            params = line.split(',')
            self.controller.eukli_distances.update({params[0]: float(params[1].strip())})
            self.controller.identities.update({params[0]: params[2].strip()})
            # in one line we have: "subject, eukli_distance, identity"

    def update_subjects_text_entry(self, subjects_path):
        self.controller.recognizer.load_subjects(subjects_path)
        self.resulting_class_entry.delete(0, tki.END)
        self.resulting_class_entry.insert(0, ','.join(self.controller.recognizer.subjects))

    def load_model(self, algorithm = None):
        if algorithm is None: 
            algorithm = self.alg_var.get()
            model_path, parameters_path, subjects_path = get_malually_model_data_paths(algorithm)
        else:
            model_path, parameters_path, subjects_path = get_model_data_paths_for_algorithm(algorithm)
        try:
            detector = Face_detector(CLASSIFIER_FILE_PATH)
            self.controller.recognizer = Face_recognitor(detector, algorithm)
            self.update_subjects_text_entry(subjects_path)
            self.set_eukli_distances_and_identities(parameters_path)
            self.controller.recognizer.read_model(model_path, subjects_path)
            messagebox.showinfo('Load successful', 'Model was loaded!')
        except:
            messagebox.showinfo('Load unsuccessful', 'Wrong selected files')
    
    def create_model(self):
        detector = Face_detector(CLASSIFIER_FILE_PATH)
        self.controller.recognizer = Face_recognitor(detector, self.alg_var.get())
        self.controller.recognizer.train_model(self.training_path_entry.get(), MODELS_FILES_PATH)
        self.resulting_class_entry.delete(0, tki.END)
        self.resulting_class_entry.insert(0, ','.join(self.controller.recognizer.subjects))
        self.test_model(self.testing_path_entry.get())

    def radio_checked(self): # unnecessary
        print(self.alg_var.get())

    def confirm_model(self):
        messagebox.showinfo('Complete the face recognition configuration','The configuration has been saved, do not manipulate files inside resources folder.')
        with open(PROGRAM_STATE_FILE_PATH, 'w') as file: file.write('1,' + self.alg_var.get())