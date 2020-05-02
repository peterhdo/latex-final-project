#! /usr/local/bin/python3
"""
Helps generate the corresponding symbols given the stroke dataset from detexify.
"""

from PIL import Image, ImageDraw
from numpy import array
import json
import os
import psycopg2 as pg

SYMBOLS_FILE = 'symbols.json'

# width of the rendered stroke (pixels)
STROKE_SAMPLE_WIDTH = 6

def strokes_to_image(image_drawer, strokes):
    # draw all the strokes to the image
    cleaned_strokes = [(int(x[0]), int(x[1])) for x in strokes]
    image_drawer.line(cleaned_strokes, fill="black", width=STROKE_SAMPLE_WIDTH)

def convert_to_image(sample):
    size = (400, 400)

    # Creates a grayscale image
    with Image.new('L', size, "white") as im:
        image_drawer = ImageDraw.Draw(im)

        # actually draw the strokes
        strokes_to_image(image_drawer, sample['strokes'])
        fname = "images/%s.png" % (sample['_id'],)
        im.save(fname)

if __name__ == "__main__":
    # start postgre client
    conn = pg.connect("dbname=kevinbaichoo user=kevinbaichoo")

    # open a cursor to perform database ops
    cur = conn.cursor()

    all_symbols = []
    with open(SYMBOLS_FILE, 'r') as f:
        all_symbols = json.loads(f.read())
    
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
            sample = {'_id': command + ';' + str(id), 'strokes': strokes[0]}
            convert_to_image(sample)
