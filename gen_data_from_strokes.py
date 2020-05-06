#! /usr/local/bin/python3
"""
Helps generate the corresponding symbols given the stroke dataset from detexify.
"""

from PIL import Image, ImageDraw
from numpy import array
import json
import os
import psycopg2 as pg
import math

SYMBOLS_FILE = 'symbols.json'
# CSV of path to image from root directory, bounding box coordinates
BOUNDING_BOXS = 'bounding_boxes.csv'

# width of the rendered stroke (pixels)
STROKE_SAMPLE_WIDTH = 6

def strokes_to_image(image_drawer, strokes):
    """Draws all the strokes of the image."""
    cleaned_strokes = [(int(x[0]), int(x[1])) for x in strokes]
    image_drawer.line(cleaned_strokes, fill='black', width=STROKE_SAMPLE_WIDTH)

def calculate_bounding_box(strokes):
    """
    Calculates the bounding box given the stroke information.
    Returns [(top left coord), (bottom right coord)] 
    """

    # record min top value, max bottom, min left, max right
    top = left = math.inf
    bottom = right = -math.inf

    for s in strokes:
        x, y = int(s[0]), int(s[1])
        top = min(top, y)
        left = min(left, x)
        bottom = max(bottom, y)
        right = max(right, x)
    return [(left,top), (right, bottom)] 


def convert_to_image(sample, image_path='images', with_bounding_box = False, display_image = False):
    """
    Converts strokes to image.

    Args:
    - Sample: sample object from the database
    - image_path: path to save the image (set to None to skip saving)
    - with_bounding_box: whether to add the bounding box to the image
    - display_image: whether to show the generated image.
    """
    size = (400, 400)

    # Creates a grayscale image
    with Image.new('L', size, 'white') as im:
        image_drawer = ImageDraw.Draw(im)

        # actually draw the strokes
        strokes_to_image(image_drawer, sample['strokes'])

        # add bounding box if specified
        if with_bounding_box:
            image_drawer.rectangle(calculate_bounding_box(sample['strokes']), outline='red')

        if display_image:
            im.show()

        if image_path:
            im.save(image_path)

def create_dir(path):
    # Create target Directory if don't exist
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == "__main__":
    # start postgre client
    conn = pg.connect("dbname=kevinbaichoo user=kevinbaichoo")

    # open a cursor to perform database ops
    cur = conn.cursor()

    all_symbols = []
    with open(SYMBOLS_FILE, 'r') as f:
        all_symbols = json.loads(f.read())

    with open(BOUNDING_BOXS, 'w+') as f:
        for symbol in all_symbols:
            # Create an image (or multiple using the cursor)
            symbol_id = symbol['id']
            command = symbol['command']

            # File system can't grok slashes (messes with path)
            if command == '/':
                command = 'slash'
            
            print('Generating images for command: {}'.format(command))

            query = "select * from samples where key ='{}'".format(symbol_id)
            cur.execute(query)
            results = cur.fetchall()

            # generate images from strokes
            for result in results:
                id, _, strokes = result
                id = str(id)
                sample = {'strokes': strokes[0]}
                dir_name = 'symbol_images/{}'.format(command)
                create_dir(dir_name)
                image_file = "{}/{}.png".format(dir_name, id)

                # Output to the csv
                f.write('{}, {}\n'.format(image_file, calculate_bounding_box(strokes[0])))
                convert_to_image(sample, image_file) # see if bounding box generated correctly.
