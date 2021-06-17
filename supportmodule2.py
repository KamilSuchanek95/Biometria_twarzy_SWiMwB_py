import os



CORE_PATH   =   os.path.dirname(os.path.realpath(__file__)) # used below too

RESOURCES_PATH  = os.path.join(CORE_PATH, 'resources')

# IMAGES_PATH             = os.path.join(RESOURCES_PATH, 'images')

MODELS_FILES_PATH   = os.path.join(RESOURCES_PATH, 'models')
HAAR_CASCADE_FILE_PATH  = os.path.join(MODELS_FILES_PATH, 'haarcascade_frontalface_default.xml')

