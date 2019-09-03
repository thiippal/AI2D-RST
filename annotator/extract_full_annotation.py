# -*- coding: utf-8 -*-

"""
This script allows the user to extract the original AI2D annotation from JSON files and
store it into a pandas DataFrame for processing using the AI2D-RST annotation tool.

Usage:
    python extract_full_annotation.py -a annotation/ -i images/ -o output.pkl
    
Arguments:
    -a/--annotation: Path to the directory with the AI2D JSON files.
    -i/--images: Path to the directory with the AI2D diagram images.
    -o/--output: Path to the output file, in which the resulting annotation is
                 stored.

Returns:
    A pandas DataFrame containing the AI2D annotation.
"""

from core import parse
import argparse
import os
import pandas as pd

# Set up the argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-a", "--annotation", required=True,
                help="Path to the directory with AI2D JSON files.")
ap.add_argument("-i", "--images", required=True,
                help="Path to the directory with AI2D images.")
ap.add_argument("-o", "--output", required=True,
                help="Path to the file in which the extracted annotation is "
                     "stored.")

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
