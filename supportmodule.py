# module for help organize GUI code and PATHS

import os
from tkinter import filedialog

# Paths
#   core
#    - main.py (and other programms now)
#    - resources (folder for resources)
#        - default (folder for default application files)
#        - images (folder for saved by application pictures)
#        - models (folder for trained in application models .xml)
#        - program (folder maybe for another script files)
#        - program_state (folder for set.txt file includes initial state of program)
CORE_PATH   =   os.path.dirname(os.path.realpath(__file__)) # used below too

RESOURCES_PATH  = os.path.join(CORE_PATH, 'resources')

DEFAULT_FILES_PATH  = os.path.join(RESOURCES_PATH, 'default')

DEFAULT_IMAGE_PATH  = os.path.join(DEFAULT_FILES_PATH, 'DefaultImage.jpg')

IMAGES_PATH             = os.path.join(RESOURCES_PATH, 'images')

MODELS_FILES_PATH   = os.path.join(RESOURCES_PATH, 'models')

FISHERFACE_MODEL_PATH   = os.path.join(MODELS_FILES_PATH, 'fisherface_model.xml')
EIGENFACE_MODEL_PATH    = os.path.join(MODELS_FILES_PATH, 'eigenface_model.xml')
LBPH_MODEL_PATH         = os.path.join(MODELS_FILES_PATH, 'lbph_model.xml')
FISHERFACE_SUBJECTS_PATH    = os.path.join(MODELS_FILES_PATH, 'fisherface_subjects.csv')
EIGENFACE_SUBJECTS_PATH     = os.path.join(MODELS_FILES_PATH, 'eigenface_subjects.csv')
LBPH_SUBJECTS_PATH          = os.path.join(MODELS_FILES_PATH, 'lbph_subjects.csv')
FISHERFACE_PARAMETERS_PATH  = os.path.join(MODELS_FILES_PATH, 'fisherface_parameters.csv')
EIGENFACE_PARAMETERS_PATH   = os.path.join(MODELS_FILES_PATH, 'eigenface_parameters.csv')
LBPH_PARAMETERS_PATH        = os.path.join(MODELS_FILES_PATH, 'lbph_parameters.csv')


PROGRAM_STATE_FOLDER_PATH   = os.path.join(RESOURCES_PATH, 'program_state')
PROGRAM_STATE_FILE_PATH         = os.path.join(PROGRAM_STATE_FOLDER_PATH, 'set.txt')


# Resources objects
# from PIL import Image, ImageTk
# default_image = ImageTk.PhotoImage(Image.open(DEFAULT_IMAGE_PATH))

# Functions

def create_resources_if_they_dont_exists():
    # create folders
    for dir in [MODELS_FILES_PATH, IMAGES_PATH, PROGRAM_STATE_FOLDER_PATH]:
        os.makedirs(dir) if not os.path.exists(dir) else None
    # create set.txt or overwrite if empty
    if not os.path.isfile(PROGRAM_STATE_FILE_PATH):
        with open(PROGRAM_STATE_FILE_PATH, 'a') as f: f.write('0,')
    else:
        with open(PROGRAM_STATE_FILE_PATH, 'r+') as f: 
            not_empty = f.read(1)
            None if not_empty else f.write('0,')
    

def get_model_data_from_resources(algorithm):
    switcher = {
                'lbph':         [LBPH_MODEL_PATH,
                                 LBPH_PARAMETERS_PATH,
                                 LBPH_SUBJECTS_PATH],
                'fisherface':   [FISHERFACE_MODEL_PATH,
                                 FISHERFACE_PARAMETERS_PATH,
                                 FISHERFACE_SUBJECTS_PATH],
                'eigenface':    [EIGENFACE_MODEL_PATH,
                                 EIGENFACE_PARAMETERS_PATH,
                                 EIGENFACE_SUBJECTS_PATH]
    }
    return switcher.get(algorithm, 'Invalid algorithm name.')


def get_malually_model_data_paths(algorithm):
    model_path = filedialog.askopenfile(title = "Select the model .xml file for the " + algorithm + " algorithm", filetypes=(("Text Files", "*.xml"),))
    subjects_path = filedialog.askopenfile(title = "Select the subjects .csv file",     filetypes = [("Text files","*.csv")])
    parameters_path = filedialog.askopenfile(title = "Select the parameters .csv file", filetypes = [("Text files","*.csv")])
    return [model_path.name, parameters_path.name, subjects_path.name]