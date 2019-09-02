# -*- coding: utf-8 -*-

"""
This script loads examples from the macro-group annotation in the AI2D-RST
dataset and presents them to the annotator. The resulting description is used
for measuring agreement between the annotators.

To continue annotation from a previous session, give the path to the existing
DataFrame to the -o/--output argument.

Usage:
    python evaluate_agreement_macro.py -a annotation.pkl -o output.pkl
    -s sample.pkl -i path_to_ai2d_images/

Arguments:
    -a/--annotation: Path to the pandas DataFrame containing AI2D-RST Diagram
                     objects.
    -s/--sample: Path to the file containing data sampled from the AI2D-RST
                 Diagram objects.
    -i/--images: Path to the directory containing the AI2D diagram images.
    -o/--output: Path to the output file, in which the resulting annotation is
                 stored.

Returns:
    A pandas DataFrame containing the annotation stored in the input DataFrame
    and the annotation created using this script.
"""

# Import packages
from colorama import Fore, Style, init
from core.draw import *
from core.interface import macro_groups
from core.parse import *
from pathlib import Path
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd


# Initialize colorama
init()

# Set up the argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-s", "--sample", required=True)
ap.add_argument("-a", "--annotation", required=True)
ap.add_argument("-i", "--images", required=True)
ap.add_argument("-o", "--output", required=True)

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
ann_path = args['annotation']
sample_path = args['sample']
images_path = args['images']
output_path = args['output']

# Verify the input paths, print error and exit if not found
if not Path(ann_path).exists():

    exit("[ERROR] Cannot find {}. Check the input to -a!".format(ann_path))

if not Path(images_path).exists():

    exit("[ERROR] Cannot find {}. Check the input to -i!".format(images_path))

# Check if the output file exists already, or whether to continue with previous
# annotation.
if os.path.isfile(output_path):

    # Read existing files
    annotation_df = pd.read_pickle(ann_path)
    sample = pd.read_pickle(output_path)

    # Print status message
    print("[INFO] Continuing existing annotation in {}.".format(output_path))

# Otherwise, read the annotation from the input DataFrame
if not os.path.isfile(output_path):

    # Make a copy of the input DataFrame
    annotation_df = pd.read_pickle(ann_path).copy()

    # Load DataFrame containing sample
    sample = pd.read_pickle(sample_path)

# Define a list of annotator commands
commands = ['help', 'exit', 'hide', 'show']

# Begin looping over the sample and presenting the relations to the user
for i, (ix, row) in enumerate(sample.iterrows(), start=1):

    # Check if annotation has already been performed
    try:
        if row['annotation'] in [v for k, v in macro_groups.items()]:

            continue

    except KeyError:
        pass

    # Begin the annotation by clearing the screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Set annotation completeness initially to False
    annotation_complete = False

    # Fetch the filename of current diagram image
    image_name = row['image_name']

    # Join with path to image directory with current filename
    image_path = os.path.join(images_path, row['image_name'])

    # Print status message
    print(Fore.YELLOW + "[INFO] Now processing macro-group {}/{} from {}."
          .format(i, len(sample), image_name) + Style.RESET_ALL)

    # Fetch the layout annotation from the original DataFrame
    original_row = annotation_df.loc[annotation_df['image_name'] == image_name]

    # Get the annotation dictionary for the original AI2D annotation
    annotation = original_row['annotation'].item()

    # Get the AI2D Diagram object
    diagram = original_row['diagram'].item()

    # Check if the macro-group refers to a group
    if row['node_type'] == 'group':

        # Generate a dictionary mapping group aliases to IDs
        group_dict = replace_aliases(diagram.layout_graph, 'group')

        # Get the relation shown to the user
        for k, v in group_dict.items():

            # Find the item corresponding to the row identifier for RST relation
            if v == row['id']:

                # Assign the abbreviated ID used in the visualisation into
                # variable
                abbrv_id = k.upper()

    # Otherwise, simply assign the node_id to abbrv_id in 'mg_prompt' below
    else:

        abbrv_id = row['id']

    # Draw the annotation and highlight the source and the target
    segmentation = draw_layout(image_path, annotation, height=480)

    # Draw the graph
    layout_graph = draw_graph(diagram.layout_graph, dpi=100, mode='layout')

    # Define prompt for user input
    mg_prompt = Fore.RED + "[GROUPING] Which macro-group would you assign to " \
                           "{}? ".format(abbrv_id) + Style.RESET_ALL

    # Set up flag a for tracking whether annotation is hidden
    hide = False

    # Enter a while loop for performing annotation
    while not annotation_complete:

        # Join the graph and the layout structure horizontally
        preview = np.hstack((segmentation, layout_graph))

        # Show the annotation
        cv2.imshow("Annotation", preview)

        # Set command to None
        command = None

        # Ask for initial input
        is_mg = input(mg_prompt)

        # Split the input to check for commands
        try:
            is_command = is_mg.split()[0]

        # Catch error from pressing enter with no input
        except IndexError:

            continue

        # Check if the input string is a command
        if is_command in commands:

            # Set command variable
            command = is_mg

        # If the input string is a valid shorthand code for a macro-group
        if is_mg in macro_groups.keys():

            # Fetch the macro-group from the dictionary
            macro_group = macro_groups[is_mg]

            # Save the input to DataFrame
            sample.at[ix, 'annotation'] = str(macro_group)

            # Set annotation to complete
            annotation_complete = True

        # Check if user requests help
        if command == 'help':

            print(Fore.YELLOW + "[INFO] VALID MACRO-GROUPS INCLUDE: " +
                  Style.RESET_ALL)

            # Loop over the macro-groups and print information
            for k, v in macro_groups.items():

                print(Fore.YELLOW + "[INFO] {} ({})".format(k, v)
                      + Style.RESET_ALL)

            # Reset command variable
            command = None

        # Check if the user wants to exit the annotator
        if command == 'exit':

            # Print status
            print(Fore.YELLOW + "[INFO] Saving annotation and "
                                "exiting the tool." + Style.RESET_ALL)

            # Save annotation file
            sample.to_pickle(output_path)

            # Exit the annotator
            exit()

        # Hide layout segmentation if requested
        if command == 'hide':

            # Re-draw the layout
            segmentation = draw_layout(image_path, annotation, height=480,
                                       hide=True)

            # Flag the annotation as hidden
            hide = True

            continue

        # Show layout segmentation if requested
        if command == 'show':

            # Re-draw the layout
            segmentation = draw_layout(image_path, annotation, height=480,
                                       hide=False)

            # Flag the annotation as visible
            hide = False

            continue

        # Check if some elements should be highlighted in the segmentation
        if command and command.split()[0] == 'show':

            # Prepare user input
            user_input = prepare_input(command, from_item=1)

            # Validate input against current graph
            valid = validate_input(user_input, diagram.layout_graph)

            # Proceed if the input is valid
            if valid:

                # Convert input to uppercase
                user_input = [u.upper() for u in user_input]

                # Re-draw the layout
                segmentation = draw_layout(image_path, annotation, height=480,
                                           hide=False, point=user_input)

                continue

            # If the user input is not valid, continue
            if not valid:

                continue

    # Close plot
    plt.close()

# Save the output DataFrame
sample.to_pickle(output_path)

# Print status
print(Fore.RED + "[INFO] Annotation completed!" + Style.RESET_ALL)
