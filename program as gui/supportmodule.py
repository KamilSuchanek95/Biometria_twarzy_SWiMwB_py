# Constants module

import os


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

def createResourcesIfTheyDontExists():
    # create folders
    for dir in [MODELS_FILES_PATH, IMAGES_PATH, PROGRAM_STATE_FOLDER_PATH]:
        os.makedirs(dir) if not os.path.exists(dir) else None
    # create set.txt
    with open(PROGRAM_STATE_FILE_PATH, 'a+') as f: f.write('0,') if not os.path.isfile(PROGRAM_STATE_FILE_PATH) else None
    

def getModelDataFromResources(algorithm):
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


