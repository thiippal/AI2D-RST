# -*- coding: utf-8 -*-

"""
This script loads a pandas DataFrame containing AI2D-RST annotation and
prints out the status of each diagram in the file.

Usage:
    python examine_annotation.py -a annotation.pkl
    
Arguments:
    -a/--annotation: Path to the pandas DataFrame containing the annotation.
    
Returns:
    Prints the contents of the DataFrame on the standard output.
"""

# Import packages
from colorama import Fore, Style, init
import argparse
import pandas as pd

# Initialize colorama
init()

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

# Begin looping over the rows of the input DataFrame. Enumerate the result to
# show annotation progress to the user.
for i, (ix, row) in enumerate(annotation_df.iterrows(), start=1):

    # Fetch the filename of current diagram image
    image_fname = row['image_name']

    # Print status message
    print("[INFO] Now processing row {}/{} ({}) ...".format(i,
                                                            len(annotation_df),
                                                            image_fname))

    # Assign the Diagram object to a variable
    try:
        diagram = row['diagram']

        # Check if diagram is marked as complete
        try:
            complete = diagram.complete

            if complete:

                print(Fore.GREEN + "- Diagram annotation is marked as complete."
                      + Style.RESET_ALL)

            if not complete:

                print(Fore.RED + "- Diagram annotation is marked as incomplete."
                      + Style.RESET_ALL)

                try:

                    group_complete = diagram.group_complete

                    if group_complete:

                        print(Fore.GREEN + " * Grouping annotation is marked"
                                           " as complete." + Style.RESET_ALL)

                    if not group_complete:

                        print(Fore.RED + " * Grouping annotation is marked"
                                         " as incomplete." + Style.RESET_ALL)

                except AttributeError:

                    pass

                try:

                    conn_complete = diagram.connectivity_complete

                    if conn_complete:

                        print(Fore.GREEN + " * Connectivity annotation is "
                                           "marked as complete." +
                              Style.RESET_ALL)

                    if not conn_complete:

                        print(Fore.RED + " * Connectivity annotation is "
                                         "marked as incomplete." +
                              Style.RESET_ALL)

                except AttributeError:

                    pass

                try:

                    rst_complete = diagram.rst_complete

                    if rst_complete:

                        print(Fore.GREEN + " * RST annotation is marked "
                                           "as complete." +
                              Style.RESET_ALL)

                    if not rst_complete:

                        print(Fore.RED + " * RST annotation is marked as "
                                         "incomplete." +
                              Style.RESET_ALL)

                except AttributeError:

                    pass

                try:
                
                    if len(diagram.comments) > 0:
                
                        for comment in diagram.comments:
                
                            print(Fore.YELLOW + " - {}".format(comment) + Style.RESET_ALL)
                        
                except AttributeError:
                
                    pass

        except AttributeError:

            print("[ERROR] Diagram object does not exist.")

    except KeyError:

        print("[ERROR] Diagram does not exist.")
