# -*- coding: utf-8 -*-

from .annotate import *
from .draw import *
from .interface import *
from .parse import *

import cv2
import numpy as np
import os


class Diagram:
    """
    This class holds the annotation for a single AI2D-RST diagram.
    """
    def __init__(self, ai2d_ann, image):
        """
        This function initializes the Diagram class.
        
        Parameters:
            ai2d_ann: Path to the JSON file containing the original AI2D
                      annotation or a dictionary containing parsed annotation.
            image: Path to the image file containing the diagram.

        Returns:
            An AI2D Diagram object with various methods and attributes.
        """
        # Mark all annotation layers initially as incomplete
        self.complete = False
        self.group_complete = False  # grouping (hierarchy + macro)
        self.connectivity_complete = False  # connectivity
        self.rst_complete = False  # rst

        # Set image filename
        self.image_filename = image

        # Continue by checking the annotation type. If the input is a dictionary
        # assign the dictionary to the variable 'annotation'.
        if type(ai2d_ann) == dict:

            self.annotation = ai2d_ann

        else:

            # Read the JSON annotation into a dictionary
            self.annotation = load_annotation(ai2d_ann)

        # Create a graph for layout annotation (hierarchy and macro grouping)
        self.layout_graph = create_graph(self.annotation,
                                         edges=False,
                                         arrowheads=False,
                                         mode='layout'
                                         )

        # Set up placeholders for connectivity and RST layers
        self.connectivity_graph = None
        self.rst_graph = None

        # Set up a placeholder for comments
        self.comments = []

        # Set up a flag for tracking updates to the graph (for drawing)
        self.update = False

    def annotate_layout(self, review):
        """
        A function for annotating the logical / layout structure (DPG-L) of a
        diagram. This function covers both content hierarchy and macro-grouping.
        
        Parameters:
            review: A Boolean defining whether review mode is active or not.
        
        Returns:
            Updates the graph contained in the Diagram object
            (self.layout_graph) according to the user input.
        """

        # If review mode is active, unfreeze the layout graph
        if review:

            # Unfreeze the layout graph by making a copy
            self.layout_graph = self.layout_graph.copy()

        # Freeze and save current graph for resetting annotation if required
        self.reset = nx.freeze(self.layout_graph.copy())

        # Visualize the layout segmentation
        segmentation = draw_layout(self.image_filename, self.annotation, 480)

        # Draw the graph
        diagram = draw_graph(self.layout_graph, dpi=100, mode='layout')

        # Set up flag a for tracking whether annotation is hidden
        hide = False

        # Enter a while loop for the annotation procedure
        while not self.group_complete:

            # Check if the graph needs to be updated
            if self.update:

                # Close previous plot
                plt.close()

                # Re-draw the graph
                diagram = draw_graph(self.layout_graph, dpi=100, mode='layout')

                # Mark update complete
                self.update = False

            # Join the graph and the layout structure horizontally
            preview = np.hstack((diagram, segmentation))

            # Show the resulting visualization
            cv2.imshow("Annotation", preview)

            # Prompt user for input
            user_input = input(prompts['layout_default'])

            # Escape accidental / purposeful carrier returns without input
            if len(user_input.split()) == 0:

                continue

            # Check if the input is a generic command
            if user_input.split()[0] in commands['generic']:

                # Send the command to the interface along with current graph
                process_command(user_input,
                                mode='layout',
                                diagram=self,
                                current_graph=self.layout_graph)

                # If the user wants to move on the next diagram without marking
                # the annotation as done or exit altogether, break from the
                # loop.
                if user_input.split()[0] == 'next':

                    return user_input.split()[0]

                if user_input.split()[0] == 'exit':

                    return user_input.split()[0]

                continue

            # Check if the user has requested to switch annotation task
            if user_input.split()[0] in commands['tasks']:

                return user_input

            # Hide layout segmentation if requested
            if user_input == 'hide':

                # Re-draw the layout
                segmentation = draw_layout(self.image_filename,
                                           self.annotation,
                                           480, hide=True)

                # Flag the annotation as hidden
                hide = True

                continue

            # Show layout segmentation if requested
            if user_input == 'show':

                # Re-draw the layout
                segmentation = draw_layout(self.image_filename,
                                           self.annotation,
                                           480, hide=False)

                # Flag the annotation as visible
                hide = False

                continue

            # Check if some elements should be highlighted in the segmentation
            if user_input.split()[0] == 'show':

                # Prepare user input
                user_input = prepare_input(user_input, from_item=1)

                # Validate input against current graph
                valid = validate_input(user_input, self.layout_graph)

                # Proceed if the input is valid
                if valid:

                    # Convert input to uppercase
                    user_input = [u.upper() for u in user_input]

                    # Re-draw the layout
                    segmentation = draw_layout(self.image_filename,
                                               self.annotation,
                                               480, hide=False,
                                               point=user_input)

                    continue

                # If the user input is not valid, continue
                if not valid:

                    continue

            # Check if the user has requested to describe a macro-group
            if 'macro' == user_input.split()[0]:

                # Check the length of the input after the command [1:]
                if len(user_input.split()[1:]) < 1:

                    # Print error message
                    print("[ERROR] You must input at least one identifier in "
                          "addition to the command 'macro'.")

                    continue

                # Prepare input for validation
                user_input = prepare_input(user_input, 1)

                # Check the input against the current graph
                valid = validate_input(user_input, self.layout_graph,
                                       groups=True)

                # If the input is not valid, continue
                if not valid:

                    continue

                # Proceed if the user input is valid
                if valid:

                    # Generate a dictionary mapping group aliases to IDs
                    group_dict = replace_aliases(self.layout_graph, 'group')

                    # Replace aliases with valid identifiers, if used
                    user_input = [group_dict[u] if u in group_dict.keys()
                                  else u for u in user_input]

                    # Assign macro groups to nodes
                    macro_group(self.layout_graph, user_input)

                    continue

            # If user input does not include a valid command, assume the input
            # is a string containing a list of diagram elements.
            elif user_input.split()[0] not in commands['generic']:

                # Prepare input for validation
                user_input = prepare_input(user_input, 0)

                # Check the input against the current graph
                valid = validate_input(user_input, self.layout_graph,
                                       groups=True)

                # If the input is not valid, continue
                if not valid:

                    continue

                # Proceed if the user input is valid
                if valid:

                    # Check input length
                    if len(user_input) == 1:

                        # Print error message
                        print("Sorry, you must enter more than one identifier "
                              "to form a group.")

                        continue

                    # Proceed if aufficient number of valid elements is provided
                    elif len(user_input) > 1:

                        # Generate a dictionary mapping group aliases to IDs
                        group_dict = replace_aliases(self.layout_graph, 'group')

                        # Replace aliases with valid identifiers, if used
                        user_input = [group_dict[u]
                                      if u.lower() in group_dict.keys()
                                      else u for u in user_input]

                        # Update the graph according to user input
                        group_nodes(self.layout_graph, user_input)

                        # Flag the graph for re-drawing
                        self.update = True

                # Continue until the annotation process is complete
                continue

    def annotate_connectivity(self, review):
        """
        A function for annotating a diagram for its connectivity.

        Parameters:
            review: A Boolean defining whether review mode is active or not.

        Returns:
            Updated the graph contained in the Diagram object
            (self.connectivity_graph) according to the user input.
        """
        # If review mode is active, unfreeze the layout graph
        if review:

            try:
                # Unfreeze the layout graph by making a copy
                self.connectivity_graph = self.connectivity_graph.copy()

            # If a connectivity graph has never been annotated, catch the error
            except AttributeError:

                pass

        # Visualize the layout segmentation
        segmentation = draw_layout(self.image_filename, self.annotation, 480)

        # If the connectivity graph does not exist, create graph
        if self.connectivity_graph is None:

            # Create an empty MultiDiGraph
            self.connectivity_graph = nx.MultiDiGraph()

        # Update grouping information using the grouping layer
        update_grouping(self, self.connectivity_graph)

        # Draw the graph using the connectivity mode
        diagram = draw_graph(self.connectivity_graph, dpi=100,
                             mode='connectivity')

        # Set up flag a for tracking whether annotation is hidden
        hide = False

        # Enter a while loop for the annotation procedure
        while not self.connectivity_complete:

            # Check if the graph needs to be updated
            if self.update:

                # Close previous plot
                plt.close()

                # Re-draw the graph using the layout mode
                diagram = draw_graph(self.connectivity_graph, dpi=100,
                                     mode='connectivity')

                # Mark update complete
                self.update = False

            # Join the graph and the layout structure horizontally
            preview = np.hstack((diagram, segmentation))

            # Show the resulting visualization
            cv2.imshow("Annotation", preview)

            # Prompt user for input
            user_input = input(prompts['conn_default'])

            # Escape accidental / purposeful carrier returns without input
            if len(user_input.split()) == 0:

                continue

            # Check if the user input is a command
            if user_input.split()[0] in (commands['generic'] +
                                         commands['connectivity']):

                # Send the command to the interface along with current graph
                process_command(user_input,
                                mode='connectivity',
                                diagram=self,
                                current_graph=self.connectivity_graph
                                )

                # If the user wants to move on the next diagram without marking
                # the annotation as done or exit altogether, break from the
                # loop.
                if user_input.split()[0] == 'next':

                    return user_input.split()[0]

                if user_input.split()[0] == 'exit':

                    return user_input.split()[0]

                # Otherwise continue
                continue

            # Check if the user has requested to switch annotation task
            if user_input.split()[0] in commands['tasks']:

                return user_input

            # Hide layout segmentation if requested
            if user_input == 'hide':

                # Re-draw the layout
                segmentation = draw_layout(self.image_filename,
                                           self.annotation,
                                           480, hide=True)

                # Flag the annotation as hidden
                hide = True

                continue

            # Show layout segmentation if requested
            if user_input == 'show':

                # Re-draw the layout
                segmentation = draw_layout(self.image_filename,
                                           self.annotation,
                                           480, hide=False)

                # Flag the annotation as visible
                hide = False

                continue

            # Check if some elements should be highlighted in the segmentation
            if user_input.split()[0] == 'show':

                # Prepare user input
                user_input = prepare_input(user_input, from_item=1)

                # Validate input against current graph
                valid = validate_input(user_input, self.layout_graph)

                # Proceed if the input is valid
                if valid:

                    # Convert input to uppercase
                    user_input = [u.upper() for u in user_input]

                    # Re-draw the layout
                    segmentation = draw_layout(self.image_filename,
                                               self.annotation,
                                               480, hide=False,
                                               point=user_input)

                    continue

                # If the user input is not valid, continue
                if not valid:

                    continue

            # If user input does not include a valid command, assume the input
            # is a string defining a connectivity relation.
            elif user_input.split()[0] not in (commands['generic'] +
                                               commands['connectivity']):

                # Set a flag for tracking connections
                connection_found = False

                # Define connection type aliases and their names
                connection_types = {'-': 'undirectional',
                                    '>': 'directional',
                                    '<>': 'bidirectional'}

                # Split the input into a list
                user_input = user_input.split(' ')

                # Strip extra whitespace
                user_input = [u.strip() for u in user_input]

                # Loop over connection types and check them against the input
                for alias in connection_types.keys():

                    # If a match is found, record its index in user input
                    if alias in user_input:

                        # Get connection index and type; assign to variable
                        connection_ix = user_input.index(alias)
                        connection_type = connection_types[alias]

                        # Use connection index to get source and target sets
                        source = user_input[:connection_ix]
                        target = user_input[connection_ix + 1:]

                        # Strip possible extra commas from sources and targets
                        source = [x.strip(',') for x in source]
                        target = [x.strip(',') for x in target]

                        # Prepare input for validation
                        source = prepare_input(' '.join(source), from_item=0)
                        target = prepare_input(' '.join(target), from_item=0)

                        # Check the input against the current graph
                        valid = validate_input(source + target,
                                               self.connectivity_graph,
                                               groups=True)

                        # If the user input is not valid, continue
                        if not valid:

                            continue

                        # If the user input is valid, proceed
                        if valid:

                            # Set connection tracking flag to True
                            connection_found = True

                            continue

                # If a valid connection type is found, create a new connection
                if connection_found:

                    # Initialize a list for edge tuples
                    edge_bunch = []

                    # Generate a dictionary mapping group aliases to IDs
                    group_dict = replace_aliases(self.connectivity_graph,
                                                 'group')

                    # Update the group identifiers in sources and targets to use
                    # valid identifiers, not the G-prefixed aliases
                    source = [group_dict[s] if s in group_dict.keys() else s
                              for s in source]
                    target = [group_dict[t] if t in group_dict.keys() else t
                              for t in target]

                    # Loop over sources
                    for s in source:

                        # Loop over targets
                        for t in target:

                            # Convert identifiers to uppercase and add an edge
                            # tuple to the list of edges
                            edge_bunch.append((s.upper(), t.upper()))

                    # If the connection type is bidirectional, add arrows also
                    # from target to source.
                    if connection_type == 'bidirectional':

                        # Loop over targets
                        for t in target:

                            # Loop over sources
                            for s in source:

                                # Convert identifiers to uppercase as above and
                                # add an edge tupleto the list of edges
                                edge_bunch.append((t.upper(), s.upper()))

                    # When edges have been added for all connections, add edges
                    # from the edge list
                    self.connectivity_graph.add_edges_from(edge_bunch,
                                                           kind=connection_type)

                    # Flag the graph for re-drawing
                    self.update = True

            # Continue until the annotation process is complete
            continue

    def annotate_rst(self, review):
        """
        A function for annotating the rhetorical structure (DPG-R) of a diagram.
        
        Parameters:
            review: A Boolean defining whether review mode is active or not.
        
        Returns:
            Updates the RST graph in the Diagram object (self.rst_graph).
        """
        # If review mode is active, unfreeze the RST graph
        if review:

            try:
                # Unfreeze the RST graph by making a copy
                self.rst_graph = self.rst_graph.copy()

            # If RST graph has never been annotated, catch the error
            except AttributeError:

                pass

        # Visualize the layout segmentation
        segmentation = draw_layout(self.image_filename, self.annotation, 480)

        # If the RST graph does not exist, populate graph
        if self.rst_graph is None:

            # Create an empty DiGraph
            self.rst_graph = nx.DiGraph()

        # Update grouping information using the grouping layer
        update_grouping(self, self.rst_graph)

        # Draw the graph using RST mode
        diagram = draw_graph(self.rst_graph, dpi=100, mode='rst')

        # Set up flag a for tracking whether annotation is hidden
        hide = False

        # Enter a while loop for the annotation procedure
        while not self.rst_complete:

            # Check if the graph needs to be updated
            if self.update:

                # Close previous plot
                plt.close()

                # Re-draw the graph
                diagram = draw_graph(self.rst_graph, dpi=100, mode='rst')

                # Mark update complete
                self.update = False

            # Join the graph and the layout structure horizontally
            preview = np.hstack((diagram, segmentation))

            # Show the resulting visualization
            cv2.imshow("Annotation", preview)

            # Prompt user for input
            user_input = input(prompts['rst_default'])

            # Escape accidental / purposeful carrier returns without input
            if len(user_input.split()) == 0:

                continue

            # Check the input
            if user_input.split()[0] in (commands['generic'] + commands['rst']):

                # Send the command to the interface along with current graph
                process_command(user_input,
                                mode='rst',
                                diagram=self,
                                current_graph=self.rst_graph)

                # If the user wants to move on the next diagram without marking
                # the annotation as done or exit altogether, break from the
                # loop.
                if user_input.split()[0] == 'next':

                    return user_input.split()[0]

                if user_input.split()[0] == 'exit':

                    return user_input.split()[0]

                # Otherwise continue
                continue

            # Check if the user has requested to switch annotation task
            if user_input.split()[0] in commands['tasks']:

                return user_input

            # Hide layout segmentation if requested
            if user_input == 'hide':

                # Re-draw the layout
                segmentation = draw_layout(self.image_filename,
                                           self.annotation,
                                           480, hide=True)

                # Flag the annotation as hidden
                hide = True

                continue

            # Show layout segmentation if requested
            if user_input == 'show':

                # Re-draw the layout
                segmentation = draw_layout(self.image_filename,
                                           self.annotation,
                                           480, hide=False)

                # Flag the annotation as visible
                hide = False

                continue

            # Check if some elements should be highlighted in the segmentation
            if user_input.split()[0] == 'show':

                # Prepare user input
                user_input = prepare_input(user_input, from_item=1)

                # Validate input against current graph
                valid = validate_input(user_input, self.layout_graph)

                # Proceed if the input is valid
                if valid:

                    # Convert input to uppercase
                    user_input = [u.upper() for u in user_input]

                    # Re-draw the layout
                    segmentation = draw_layout(self.image_filename,
                                               self.annotation,
                                               480, hide=False,
                                               point=user_input)

                    continue

                # If the user input is not valid, continue
                if not valid:

                    continue

            # If the user input is a new relation, request additional input
            if user_input == 'new':

                # Request relation name
                relation = input(prompts['rel_prompt'])

                # Strip extra whitespace and convert the input to lowercase
                relation = relation.strip().lower()

                # Check that the input is a valid relation
                if relation in rst_relations.keys():

                    # Create a rhetorical relation and add to graph
                    create_relation(self.rst_graph, relation)

                    # Flag the graph for re-drawing
                    self.update = True

                else:
                    print("[ERROR] Sorry, {} is not a valid relation."
                          .format(relation))

            else:

                # Print error message
                print("[ERROR] Sorry, {} is not a valid command."
                      .format(user_input))

                continue

            # Continue until the annotation process is complete
            continue
