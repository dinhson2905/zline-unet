import os
from os.path import splitext
from os import listdir
from glob import glob

path = 'data/Train/Images/'
ids = [splitext(file)[0] for file in listdir(path) if not file.startswith('.')]
print(len(ids))
idx = ids[2]
print(idx)


img_file = glob(path + idx + '.*')
print(img_file)
# print("Path at terminal when executing this file")
# print(os.getcwd() + "\n")

# print("This file path, relative to os.getcwd()")
# print(__file__ + "\n")

# print("This file full path (following symlinks)")
# full_path = os.path.realpath(__file__)
# print(full_path + "\n")

# print("This file directory and name")
# path, filename = os.path.split(full_path)
# print(path + ' --> ' + filename + "\n")

# print("This file directory only")
# print(os.path.dirname(full_path))