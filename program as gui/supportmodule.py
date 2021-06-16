# Constants module

import os


# Paths
CORE_PATH = os.path.dirname(os.path.realpath(__file__))

CORE_DIRS_PATH = [os.path.join(CORE_PATH, 'Recognition Systems'), 
                  os.path.join(CORE_PATH, 'models'), 
                  os.path.join(CORE_PATH, 'images')]

DEFAULT_IMAGE_PATH = os.path.join(CORE_PATH,  'resources', 'default', 'DefaultImage.jpg')
#'resources', 'default',

# Resources objects
# from PIL import Image, ImageTk
# default_image = ImageTk.PhotoImage(Image.open(DEFAULT_IMAGE_PATH))

# Functions

def create_app_folders_if_they_dont_exist():
    for dir in CORE_DIRS_PATH:
      os.makedirs(dir) if not os.path.exists(dir) else None

