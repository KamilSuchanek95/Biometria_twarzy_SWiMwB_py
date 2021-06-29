
from gui_helper import *
import cv2
from functools import partial
import tkinter as tki
from tkinter import messagebox
import detector_and_recognitor as s


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
        self.create_model_button = tki.Button(self, command=self.manually_load_model, text='Or load model\nfrom file', relief='solid')
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
            self.load_model_first_time(is_set_and_algorithm[1])
    
    """# Face recognition configuration metods"""

    def get_path(self, entry_handle):
        path = filedialog.askdirectory()
        entry_handle.delete(0, tki.END)
        entry_handle.insert(0, path)


    def create_identity_classes(self):
        parameters_path = get_model_data_paths_for_algorithm(str(self.controller.recognizer.algorithm))[1]
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
                        for idx, line in enumerate(lines):
                            line = ','.join([line.strip(), self.controller.identities[self.controller.recognizer.subjects[idx]]]) + "\n"
                        with open(parameters_path, 'w') as f: f.writelines(lines)
                    else:
                        self.resulting_class_entry.insert(0, "Parameters file is wrong!")
            else:
                messagebox.showinfo('Error entered Ids list','Lengths of IDs list and subjects is not equal')


    def test_model(self, test_images_path): # create name_parameters.csv file with {subject,mean p, identity} columns
        parameters_path = get_model_data_paths_for_algorithm(self.controller.recognizer.algorithm)[1]
        fw = open(parameters_path, "w")
        for subject in os.listdir(test_images_path):
            eukli_distances = []
            subject_images_path = os.path.join(test_images_path, subject)
            for image in os.listdir(subject_images_path):
                image_path = os.path.join(subject_images_path, image)
                test_img = cv2.imread(image_path)
                predicted_img, eukli_distance, who = self.controller.recognizer.predict(test_img)
                if predicted_img is None: #if recognize nothing
                    continue
                if who == subject: # if recognize correctly
                    eukli_distances.append(eukli_distance)
            if len(eukli_distances) > 0:
                fw.write(subject + ',' + str(max(eukli_distances)) + '\n')
                self.controller.eukli_distances.update({subject: max(eukli_distances)})
            else:
                fw.write(subject + ',' + 0 + '\n')
                self.controller.eukli_distances.update({subject: 0})
        fw.close()
        print('End of testing model')

    def set_eukli_distances_and_identities(self, parameters_path):
        with open(parameters_path, 'r') as f: lines = f.readlines()
        # if len(lines[0].split(',')) < 3:
        #     return None
        for line in lines:
            params = line.split(',')
            self.controller.eukli_distances.update({params[0]: float(params[1].strip())})
            self.controller.identities.update({params[0]: params[2].strip()})
            # in one line we have: "subject, eukli_distance, identitie"

    def update_subjects_text_entry(self, subjects_path):
        self.controller.recognizer.load_subjects(subjects_path)
        self.resulting_class_entry.delete(0, tki.END)
        self.resulting_class_entry.insert(0, ','.join(self.controller.recognizer.subjects))

    def load_model(self, algorithm, model_paths):
        try:
            model_path, parameters_path, subjects_path = model_paths
            self.update_subjects_text_entry(subjects_path)
            self.set_eukli_distances_and_identities(parameters_path)
            self.controller.recognizer = s.Face_recognitor(algorithm)
            self.controller.recognizer.read_model(model_path, subjects_path)
            messagebox.showinfo('Load successful', 'Model was loaded!')
        except:
            messagebox.showinfo('Load unsuccessful', 'Wrong selected files')
    
    def load_model_first_time(self, algorithm):
        # algorithm = str(self.alg_var.get())
        model_paths = get_model_data_paths_for_algorithm(algorithm)
        self.load_model(algorithm, model_paths)

    def manually_load_model(self):
        algorithm = str(self.alg_var.get())
        model_paths = get_malually_model_data_paths(algorithm)
        self.load_model(algorithm, model_paths)
    
    def create_model(self):
        # create model object
        self.controller.recognizer = s.Face_recognitor(self.alg_var.get())
        # train it
        self.controller.recognizer.train_model(self.training_path_entry.get())
        # set subjects
        # entry classes
        self.resulting_class_entry.delete(0, tki.END)
        self.resulting_class_entry.insert(0, ','.join(self.controller.recognizer.subjects))
        # test model
        self.test_model(self.testing_path_entry.get())

    def radio_checked(self): # unnecessary
        print(self.alg_var.get())

    def confirm_model(self):
        messagebox.showinfo('Complete the face recognition configuration','The configuration has been saved, do not manipulate files inside resources folder.')
        with open(PROGRAM_STATE_FILE_PATH, 'w') as file:
            file.write('1,' + str(self.alg_var.get()))