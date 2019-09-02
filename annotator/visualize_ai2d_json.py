# -*- coding: utf-8 -*-

from core import Diagram
from core.annotate import *
from core.draw import *
from core.interface import *
from core.parse import *
import argparse
import cv2
import numpy as np
import os
import pandas as pd

# Set up the argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-a", "--annotation", required=True, help="Path to the "
                                                          "directory or file "
                                                          "with AI2D "
                                                          "annotation.")
ap.add_argument("-i", "--images", required=True, help="Path to directory with "
                                                      "AI2D image files.")
ap.add_argument("-s", "--save", required=False, action='store_true',
                help="Save screenshots for each diagram.")

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
ann_path = args['annotation']
img_path = args['images']

# Check the type of input, beginning with files
if os.path.isfile(ann_path) and os.path.isfile(img_path):

    # Initialise diagram
    diagram = Diagram(ann_path, img_path)

    # Create a diagram with original annotation
    diagram.graph = create_graph(diagram.annotation,
                                 edges=True, arrowheads=True)

    # Draw layout segmentation
    layout = draw_layout(img_path, diagram.annotation,
                         height=720, dpi=200, arrowheads=True)

    # Visualize the graph
    graph_viz = draw_graph(diagram.graph, dpi=200)

    # Join the graph and layout visualizations
    preview = np.hstack((graph_viz, layout))

    # Write image on disk if requested
    if args['save']:
        cv2.imwrite("segmentation.png", layout)
        cv2.imwrite("graph.png", graph_viz)

    # Show the visualization
    cv2.imshow("{}".format(img_path.split('/')[-1]), preview)

    # If the user presses 'q', then exit, otherwise show next diagram.
    if cv2.waitKey() == ord('q'):
        quit()

    # Destroy all windows
    cv2.destroyAllWindows()

# If both variables are directories, loop through their contents
if os.path.isdir(ann_path) and os.path.isdir(img_path):

    # Walk through the directory containing the annotations
    for (ann_root, ann_dirs, ann_files) in os.walk(ann_path):

        # Loop over the files
        for f in ann_files:

            # Process JSON files only
            if f.split('.')[-1] == 'json':

                # Retrieve the unique identifier from the filename (position -3)
                # and reconstruct the image file name.
                image_fn = f.split('.')[-3] + '.png'

                # Get paths to annotation and image
                json_path = os.path.join(ann_root, f)
                image_path = os.path.join(img_path, image_fn)

                # Initialise diagram
                diagram = Diagram(json_path, image_path)

                # Draw layout segmentation
                layout = draw_layout(image_path, diagram.annotation,
                                     height=720, dpi=200, arrowheads=True)

                # Create a diagram with original annotation
                diagram.graph = create_graph(diagram.annotation,
                                             edges=True,
                                             arrowheads=True)

                # Visualize the graph
                graph_viz = draw_graph(diagram.graph, dpi=200)

                # Join the graph and layout visualizations
                preview = np.hstack((graph_viz, layout))

                # Write image on disk if requested
                if args['save']:
                    cv2.imwrite("segmentation_{}".format(image_fn), layout)
                    cv2.imwrite("graph_{}".format(image_fn), graph_viz)

                # Show the visualization
                cv2.imshow("{}".format(image_fn), preview)

                # If user presses 'q', then exit, otherwise show next diagram
                if cv2.waitKey() == ord('q'):
                    quit()

                # Destroy all windows
                cv2.destroyAllWindows()

# Check if the annotation is a file (pickled DataFrame) and images are in a dir
if os.path.isfile(ann_path) and os.path.isdir(img_path):

    # Attempt to load the pickled DataFrame
    try:
        annotation_df = pd.read_pickle(ann_path)

    except KeyError:
        exit("Sorry, {} is not a valid pandas DataFrame.".format(ann_path))

    # Begin looping over the rows of the input DataFrame. Enumerate the result
    # to show annotation progress to the user.
    for i, (ix, row) in enumerate(annotation_df.iterrows(), start=1):

        # Fetch the filename of current diagram image
        image_fname = row['image_name']

        # Join with path to image directory with current filename
        image_path = os.path.join(img_path, row['image_name'])

        # Load the annotation
        annotation = row['annotation']

        # Draw segmentation
        layout = draw_layout(image_path, annotation, height=720, dpi=200)

        # Initialise diagram
        diagram = Diagram(annotation, image_path)

        # Create a diagram with original annotation
        diagram.graph = diagram.create_graph(diagram.annotation,
                                             edges=True,
                                             arrowheads=True)

        # Visualize the graph
        graph_viz = draw_graph(diagram.graph, dpi=200)

        # Join the graph and layout visualizations
        preview = np.hstack((graph_viz, layout))

        # Write image on disk if requested
        if args['save']:
            cv2.imwrite("segmentation_{}.png".format(image_fn), layout)
            cv2.imwrite("graph_{}.png".format(image_fn), graph_viz)

        # Show the visualization
        cv2.imshow("{}".format(image_fname), preview)

        # If the user presses 'q', then exit, otherwise show next diagram.
        if cv2.waitKey() == ord('q'):
            quit()

        # Destroy all windows
        cv2.destroyAllWindows()
