# -*- coding: utf-8 -*-

"""
This script allows the user to annotate a diagram.

Usage:
    python annotate.py -a annotation.pkl -i images/ -o output.pkl
    
Arguments:
    -a/--annotation: Path to a pandas DataFrame with the original annotation
                     extracted from the AI2D dataset.
    -i/--images: Path to the directory with the AI2D diagram images.
    -o/--output: Path to the output file, in which the resulting annotation is
                 stored.
    -r/--review: Optional argument that activates review mode. This mode opens
                 each Diagram object marked as complete for editing.
    -dr/--disable_rst: Optional argument for disabling RST annotation.
    -e/--edit: Optional argument that activates editing mode. This mode opens a
               single Diagram object for editing. Provide the diagram identifier
               using this flag.

Returns:
    A pandas DataFrame containing a Diagram object for each diagram.
"""

# Import packages
from core.interface import *
from core import Diagram
from pathlib import Path
import argparse
import os
import pandas as pd

# Set up the argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-a", "--annotation", required=True,
                help="Path to the pandas DataFrame with AI2D annotation.")
ap.add_argument("-i", "--images", required=True,
                help="Path to the directory with AI2D images.")
ap.add_argument("-o", "--output", required=True,
                help="Path to the file in which the annotation is stored.")
ap.add_argument("-r", "--review", required=False, action='store_true',
                help="Activates review mode, which opens each diagram marked as"
                     " complete for inspection.")
ap.add_argument("-dr", "--disable_rst", required=False, action='store_true',
                help="Disables RST annotation.")
ap.add_argument("-e", "--edit", required=False, type=int,
                help="Activates editing mode for modifying a single diagram.")

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
ann_path = args['annotation']
images_path = args['images']
output_path = args['output']

# Verify the input paths, print error and exit if not found
if not Path(ann_path).exists():

    exit("[ERROR] Cannot find {}. Check the input to -a!".format(ann_path))

if not Path(images_path).exists():

    exit("[ERROR] Cannot find {}. Check the input to -i!".format(images_path))

# Set review mode initially to false
review = False

# Activate review mode if requested using the -r/--review flag
if args['review']:

    review = True

if not args['review']:

    review = False

# Disable RST annotation if requested using the -dr/--disable_rst flag
if args['disable_rst']:

    disable_rst = True

if not args['disable_rst']:

    disable_rst = False

# Activate editing mode if requested using the -e/--edit flag
if args['edit']:

    # Check that the user is not running review mode in parallel
    if review:

        # Print error message and exit
        exit("[ERROR] Cannot run in edit and review mode at the same time.")

    # Set editing mode to True
    edit = True

    # Assign the identifier of the diagram to be edited to a variable
    edit_id = args['edit']

if not args['edit']:

    edit = False

# Check if the output file exists already, or whether to continue with previous
# annotation.
if os.path.isfile(output_path):

    # Read existing file
    annotation_df = pd.read_pickle(output_path)

    # Print status message
    print("[INFO] Continuing existing annotation in {}.".format(output_path))

# Otherwise, read the annotation from the input DataFrame
if not os.path.isfile(output_path):

    # Make a copy of the input DataFrame
    annotation_df = pd.read_pickle(ann_path).copy()

    # If the annotator is not running in editing mode, initiate an empty column
    # to hold the diagram
    if not edit:

        annotation_df['diagram'] = None

# Begin looping over the rows of the input DataFrame. Enumerate the result to
# show annotation progress to the user.
for i, (ix, row) in enumerate(annotation_df.iterrows(), start=1):

    # Begin the annotation by clearing the screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # Fetch the filename of current diagram image
    image_fname = row['image_name']

    # Join with path to image directory with current filename
    image_path = os.path.join(images_path, row['image_name'])

    # Print status message
    print("[INFO] Now processing row {}/{} ({}) ...".format(i,
                                                            len(annotation_df),
                                                            image_fname))

    # Fetch the annotation dictionary from the DataFrame
    annotation = row['annotation']

    # Assign diagram to variable
    diagram = row['diagram']

    # Check if a Diagram object has been initialized
    if diagram is not None:

        # Check if edit mode is active
        if edit:

            # Get the identifier of current image and cast into integer
            current_id = int(image_fname.split('.')[0])

            # Check the current identifier against the requested identifier
            if current_id != edit_id:

                # In case a match is not found, move on to next diagram
                continue

            # If a match can be found, open the requested diagram for editing
            if current_id == edit_id:

                # Set review mode to True to allow editing
                review = True

                # Set the methods tracking completeness to False
                diagram.group_complete = False
                diagram.connectivity_complete = False
                diagram.rst_complete = False
                diagram.complete = False

        # If the annotator runs in a review open the diagram for revision
        if review:

            # Set the methods tracking completeness to False
            diagram.group_complete = False
            diagram.connectivity_complete = False
            diagram.rst_complete = False
            diagram.complete = False

    # If a Diagram object has not been initialized, create new
    elif diagram is None:

        # Initialise a Diagram class and assign to variable
        diagram = Diagram(annotation, image_path)

    # Set grouping as initial annotation task
    task = 'group'

    # If the diagram has not been marked as complete, annotate
    while not diagram.complete:

        # If the user has requested next diagram, break from while loop
        if task == 'next':

            break

        # If the user has requested to exit the annotator, break from the loop
        if task == 'exit':

            # Store the diagram into the column 'diagram'
            annotation_df.at[ix, 'diagram'] = diagram

            # Write the DataFrame to disk at each step
            annotation_df.to_pickle(output_path)

            # Print status message
            exit("[INFO] Saving current graph and quitting.")

        # Evaluate the completion of different annotation tasks
        while not diagram.group_complete and task == 'group':

            # Annotate layout, use variable 'task' to track switches
            task = diagram.annotate_layout(review)

            # If grouping is marked as complete, annotate connectivity
            if diagram.group_complete:

                task = 'conn'

                break

        while not diagram.connectivity_complete and task == 'conn':

            # Annotate connectivity, use variable 'task' to track switches
            task = diagram.annotate_connectivity(review)

            # If connectivity is marked as complete, annotate RST
            if diagram.connectivity_complete:

                task = 'rst'

                break

        while not diagram.rst_complete and task == 'rst':

            # Annotate RST, use variable 'task' to track switches
            task = diagram.annotate_rst(review)

            # If RST is marked as complete, break from the loop
            if diagram.rst_complete:

                if not diagram.group_complete:

                    task = 'group'

                    break

                if not diagram.connectivity_complete:

                    task = 'conn'

                    break

                else:

                    break

        # Mark diagram complete if all annotation layers have been completed
        if diagram.group_complete and diagram.connectivity_complete \
                and diagram.rst_complete:

            diagram.complete = True

            continue

        # Otherwise, mark diagram as incomplete
        else:

            diagram.complete = False

        # Make sure switches to layers marked as complete are handled
        if task == 'group' and diagram.group_complete and not diagram.complete:

            # Print error message
            print(messages['layout_complete'])

            if not diagram.connectivity_complete:

                task = 'conn'

                continue

            if not diagram.rst_complete:

                task = 'rst'

                continue

        if task == 'conn' and diagram.connectivity_complete and \
                not diagram.complete:

            # Print error message
            print(messages['conn_complete'])

            if not diagram.group_complete:

                task = 'group'

                continue

            if not diagram.rst_complete:

                task = 'rst'

                continue

        if task == 'rst' and diagram.rst_complete and not diagram.complete:

            # Print error message
            print(messages['rst_complete'])

            if not diagram.group_complete:

                task = 'group'

                continue

            if not diagram.connectivity_complete:

                task = 'conn'

                continue

        # Otherwise continue
        continue

    # Store the diagram into the column 'diagram'
    annotation_df.at[ix, 'diagram'] = diagram

    # Write the DataFrame to disk at each step
    annotation_df.to_pickle(output_path)
