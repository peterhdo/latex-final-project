import os
from shutil import copy2

# reads from ./images, gets all latex symbol classes
# creates a directory for each latex symbol under
# ./datasets/train and copies each symbol example
# to its respective directory

symbols = set()
symbol_to_files = dict()
for dirpath, dirnames, filenames in os.walk("./images"):
    for f in filenames:
        symbol = f.split(";")[0] 
        symbols.add(symbol)
        if symbol in symbol_to_files:
            symbol_to_files[symbol].append(f)
        else:
            symbol_to_files[symbol] = [f]


for symbol in symbols:
    path = os.path.join("./datasets/train/", symbol)
    os.mkdir(path)
    for filename in symbol_to_files[symbol]:
        old_filename_path = os.path.join("./images", filename)
        filename_path = os.path.join(path, filename)
        copy2(old_filename_path, filename_path)


