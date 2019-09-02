# -*- coding: utf-8 -*-

from core import parse
import argparse
import os
import pandas as pd

# Set up the argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-a", "--annotation", required=True)
ap.add_argument("-i", "--images", required=True)
ap.add_argument("-o", "--output", required=True)

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
ann_path = args['annotation']
images_path = args['images']
output_path = args['output']

# Set up a dictionary to hold the data
data = {}

# Walk through the directory containing AI2D annotation
for (ann_root, ann_dirs, ann_files) in os.walk(ann_path):

    # Loop over the files; use enumerate to get keys for dictionary
    for i, f in enumerate(ann_files):

        # Process JSON files only
        if f.split('.')[-1] == 'json':

            # Retrieve the unique identifier from the filename (position -3) and
            # reconstruct the image file name.
            image_fn = f.split('.')[-3] + '.png'

            # Define paths to images and annotation for reading the JSON file
            json_path = os.path.join(ann_root, f)
            img_path = os.path.join(images_path, image_fn)

            # Check that both exist:
            if not os.path.exists(img_path) or not os.path.exists(json_path):

                print("[ERROR] {} or {} is missing. Skipping diagram.".format(
                    img_path, json_path
                ))

                continue

            # Set dictionary item
            data[i] = {'image_name': image_fn,
                       'annotation': parse.load_annotation(json_path)}

# Create a pandas DataFrame from the dictionary; use the dict keys as index
df = pd.DataFrame.from_dict(data=data, orient='index')

# Pickle the resulting DataFrame and save to the output file
df.to_pickle(output_path)
