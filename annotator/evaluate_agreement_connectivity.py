# -*- coding: utf-8 -*-

"""
This script loads examples from the connectivity annotation from the AI2D-RST
corpus and presents them to the annotator. The resulting description is used for
measuring agreement between the annotators.

To continue annotation from a previous session, give the path to the existing
DataFrame to the -o/--output argument.

Usage:
    python evaluate_agreement_connectivity.py -a annotation.pkl -o output.pkl
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
from pathlib import Path
from colorama import Fore, Style, init
from core.draw import *
import argparse
import cv2
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

# Define prompts for user input
conn_prompt = Fore.RED + "[CONNECTIVITY] What kind of connection holds " \
                         "between the source (red) and the target (blue)? " \
              + Style.RESET_ALL

# Define a dictionary of connection types with descriptions
connections = {'u': {'cat': 'undirected', 'desc': 'The connection is '
                                                  'undirected.'},
               'd': {'cat': 'directed', 'desc': 'The connection is directed.'},
               'b': {'cat': 'bidirectional', 'desc': 'The connection is '
                                                     'bidirectional.'},
               'n': {'cat': 'no-connection', 'desc': 'No valid connection holds'
                                                     ' between the source and '
                                                     'the target.'}
              }

# Define a list of annotator commands
commands = ['help', 'exit']

# Begin looping over the sample and presenting the relations to the user
for i, (ix, row) in enumerate(sample.iterrows(), start=1):

    # Check if annotation has already been performed
    try:
        if row['annotation'] in [v['cat'] for k, v in connections.items()]:

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
    print(Fore.YELLOW + "[INFO] Now processing connection {}/{} from {}."
          .format(i, len(sample), image_name) + Style.RESET_ALL)

    # Fetch the layout annotation from the original DataFrame
    original_row = annotation_df.loc[annotation_df['image_name'] == image_name]

    # Get the annotation dictionary for the original AI2D annotation
    annotation = original_row['annotation'].item()

    # Get the AI2D Diagram object
    diagram = original_row['diagram'].item()

    # Assign source and target information to variables
    source = row['source']
    target = row['target']

    # Check the source for group identifiers
    if len(source) == 6:

        # Convert undirected graph to directed graph for access to successors
        layout_digraph = diagram.layout_graph.copy().to_directed()

        # Get successors of the group and cast into list. The node connected
        # to the outbound edge is the last item in the list.
        successors = list(layout_digraph.successors(source))

        # Get add nodes except the last node in the list
        grouped_nodes = successors[:-1]

        # Assign extracted elements to source
        source = grouped_nodes

    # Check the target for group identifiers
    if len(target) == 6:

        # Convert undirected graph to directed graph for access to predecessors
        layout_digraph = diagram.layout_graph.copy().to_directed()

        # Get successors of the group and cast into list. The node connected
        # to the outbound edge is the last item in the list.
        successors = list(layout_digraph.successors(target))

        # Get add nodes except the last node in the list
        grouped_nodes = successors[:-1]

        # Assign extracted elements to target
        target = grouped_nodes

    # Ensure that both source and target are lists
    if type(source) is not list:

        source = source.split()

    if type(target) is not list:

        target = target.split()

    # Combine source and target for highlighting
    highlight = {'red': source, 'blue': target}

    # Add an exception for self-loops in connectivity
    if source == target:

        # Highlight self-loops in yellow
        highlight = {'yellow': source + target}

    # Draw the annotation and highlight the source and the target
    segmentation = draw_layout(image_path, annotation, height=480,
                               highlight=highlight)

    # Show the annotation
    cv2.imshow("Annotation", segmentation)

    # Enter a while loop for performing annotation
    while not annotation_complete:

        # Set command to None
        command = None

        # Ask for initial input
        is_connection = input(conn_prompt)

        # Check if the input string is a command
        if is_connection in commands:

            # Set command variable
            command = is_connection

        # If the input string is a valid description of connection
        if is_connection in connections.keys():

            # Fetch the connection from the list
            connection = connections[is_connection]['cat']

            # Save the input to DataFrame
            sample.at[ix, 'annotation'] = str(connection)

            # Set annotation to complete
            annotation_complete = True

        # Check if user requests help
        if command == 'help':

            print(Fore.YELLOW + "[INFO] VALID OPTIONS INCLUDE: " +
                  Style.RESET_ALL)

            # Loop over the Gestalt principles and print information
            for k, v in connections.items():
                print(Fore.YELLOW + "[INFO] "
                                    "{} ({}): {}"
                      .format(k, v['cat'], v['desc'])
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

# Save the output DataFrame
sample.to_pickle(output_path)

# Print status
print(Fore.RED + "[INFO] Annotation completed!" + Style.RESET_ALL)
