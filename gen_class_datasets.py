import os
import random
from shutil import copyfile

TRAIN_PCT = 0.98
DEV_PCT = 0.01

symbol_images_path = 'symbol_images/'
image_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(symbol_images_path) for f in filenames
               if os.path.splitext(f)[1] == '.png']

random.shuffle(image_files)
total_size = len(image_files)
breakpoints = [
    TRAIN_PCT * total_size,
    (TRAIN_PCT + DEV_PCT) * total_size,
]

train_set = image_files[:breakpoints[0]]
dev_set = image_files[breakpoints[0]:breakpoints[1]]
test_set = image_files[breakpoints[1]:]


def copy_to_set_folder(files, set):
    for file in files:
        dest = 'datasets/{}/{}'.format(
            set,
            file.split(symbol_images_path)[1]
        )
        copyfile(file, dest)


copy_to_set_folder(train_set, 'train')
copy_to_set_folder(dev_set, 'dev')
copy_to_set_folder(test_set, 'test')
