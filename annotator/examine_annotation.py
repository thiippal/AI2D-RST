# -*- coding: utf-8 -*-

"""
This script loads a pandas DataFrame containing annotation and prints the
contents.

Usage:
    python examine_annotation.py -a annotation.pkl
    
Arguments:
    -a/--annotation: Path to the pandas DataFrame containing the annotation.
    
Returns:
    Prints the contents of the DataFrame on the standard output.
"""

# Import packages
import argparse
import pandas as pd


# Set up the argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-a", "--annotation", required=True)

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
ann_path = args['annotation']

# Read the DataFrame
annotation_df = pd.read_pickle(ann_path)

# Print out the dataframe content
print(annotation_df)
