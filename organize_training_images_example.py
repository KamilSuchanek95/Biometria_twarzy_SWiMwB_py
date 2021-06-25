# Example preprocessing images for training in my application.

# Scenario: 
#   1. I have the folder with all images, many subjects together
#   2. The images are named with some pattern
#   3. We know which original name of image matches to which personality
#   4. In result, with specify pattern, for example "subject 1 *, subject 2 *..."
#   5.      we organize images with names matched to "subject 1 *" to folder "s1",
#                and next s2, s3 etc.
# the name of images later does not matter, only location. If We want, we can drop manually
# images to specify folders in resources...

from PIL import Image
import ipdb
import shutil
import re
import errno
import os
from tkinter import filedialog

CORE_PATH   =   os.path.dirname(os.path.realpath(__file__))

def clear_work_directories():
    shutil.rmtree(os.path.join(CORE_PATH, 'converted_images'), ignore_errors=True)
    # shutil.rmtree(os.path.join(CORE_PATH, ))
    shutil.rmtree(os.path.join(CORE_PATH, 'new_training_images'), ignore_errors=True)



def take_filenames_matched_to_pattern(images_directory, pattern):
    images_names = os.listdir(images_directory)
    # files is with extension .gif, but it isn't obvious for the program... so:
    matched_names = []
    for image in images_names:
        m = re.search(pattern, image)
        matched_names.append(m.string) if m is not None else None
    return matched_names

def copy_images_remove_dots_and_convert_to_png(images_directory, new_images_folder, matched_names):
    safe_mkdir(new_images_folder)
    for file_name in matched_names:
        image = Image.open(os.path.join(images_directory, file_name))
        new_file_path = os.path.join(new_images_folder, file_name.replace('.','') + '.png')
        image.save(new_file_path, 'png')

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

clear_work_directories()

# patterns
pattern         = re.compile(r"subject\d+[a-zA-Z\.]+")
number_pattern  = re.compile(r"\d\d")


# select folder with images 
images_directory = filedialog.askdirectory(parent=None,
                        title="Wybierz folder z obrazami do nauki modelu", 
                        initialdir=CORE_PATH)

matched_names = take_filenames_matched_to_pattern(images_directory, pattern)

new_images_folder = os.path.join(CORE_PATH, 'converted_images')
copy_images_remove_dots_and_convert_to_png(images_directory, new_images_folder, matched_names)

matched_names = os.listdir(new_images_folder)

matched_names.sort()
first_number =  int(re.findall(number_pattern, matched_names[0])[0])
last_number =   int(re.findall(number_pattern, matched_names[-1])[0])
formated_numbers = [ "%.2d" % i for i in range(first_number, last_number + 1) ]

training_images_path = os.path.join(CORE_PATH, 'new_trainig_images')
safe_mkdir(training_images_path)

combined_string_with_matched_names = ' '.join(matched_names)
for number in formated_numbers:
    new_dir = os.path.join(training_images_path, "s" + str(int(number) - 1 ))
    safe_mkdir(new_dir)
    files_pattern = re.compile("subject" + number + "[a-zA-Z\\.]+")
    files_to_move = re.findall(files_pattern, combined_string_with_matched_names)
    for file in files_to_move:
        old_location = os.path.join(new_images_folder, file)
        new_location = os.path.join(new_dir, file)
        shutil.move(old_location, new_location)



# in result we have folder:
#   new_training_images
#       s01
#           -image1
#           -image2
#               ...
#       s02
#       ...


