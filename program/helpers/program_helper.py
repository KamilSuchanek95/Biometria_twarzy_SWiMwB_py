import os
import glob

def path(*args):
    return os.path.join(*args)

def listdir_with_glob(_path):
    full_dirs = glob.glob(path(_path, '*'))
    return [os.path.basename(dir) for dir in full_dirs]




    