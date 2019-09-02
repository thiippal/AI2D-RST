# -*- coding: utf-8 -*-

from .parse import *

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import networkx as nx
import os


def draw_graph(graph, dpi=100, mode='layout', **kwargs):
    """
    Draws an image of a NetworkX Graph for visual inspection.
    
    Parameters:
        graph: A NetworkX Graph.
        dpi: The resolution of the image as dots per inch.
        mode: String indicating the diagram structure to be drawn, valid options
              include 'layout' (default), 'connectivity' and 'rst'.

    Optional parameters:
        highlight: A dictionary of identifier/colour pairs to emphasise.
        
    Returns:
         An image showing the NetworkX Graph.
    """

    # Set up the matplotlib Figure, its resolution and Axis
    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)

    # Initialize a neato layout for the graph
    pos = nx.nx_pydot.graphviz_layout(graph, prog='neato')

    # Generate a dictionary with nodes and their kind
    node_types = nx.get_node_attributes(graph, 'kind')

    # Create a label dictionary for nodes
    node_dict = get_node_dict(graph, kind='node')

    # Create a label dictionary for grouping nodes
    group_dict = get_node_dict(graph, kind='group')

    # Enumerate groups and use their numbers as labels for clarity
    group_dict = {k: "G{}".format(i) for i, (k, v) in
                  enumerate(group_dict.items(), start=1)}

    # If annotating RST structure, draw relations
    if mode == 'rst':

        # Create a label dictionary for RST relations
        rel_dict = get_node_dict(graph, kind='relation')

        # Enumerate relations and use their numbers as labels for clarity
        rel_dict = {k: "R{}".format(i) for i, (k, v) in
                    enumerate(rel_dict.items(), start=1)}

        # Get a dictionary of edge labels
        edge_dict = nx.get_edge_attributes(graph, 'kind')

        # Set up a temporary edge dict
        temp_edge_dict = edge_dict.copy()

        # Replace edge labels for clarity by looping over the temporary edge
        # dict. Note that modifications are made on the actual edge dict.
        for k, v in temp_edge_dict.items():

            # If the edge type is group, pop
            if v == 'grouping':

                edge_dict.pop(k)

            # If the edge type is satellite, replace with 's'
            if v == 'satellite':

                edge_dict[k] = 's'

            # If the edge type is nucleus, replace with 'n'
            if v == 'nucleus':

                edge_dict[k] = 'n'

    # Check if some nodes need to be highlighted
    if kwargs and 'highlight' in kwargs:

        # Draw nodes and highlight some nodes
        draw_nodes(graph, pos=pos, ax=ax, node_types=node_types, mode=mode,
                   highlight=kwargs['highlight'])

    # Otherwise draw nodes present in the graph as usual
    else:

        draw_nodes(graph, pos=pos, ax=ax, node_types=node_types, mode=mode)

    # Draw labels for each node in the graph
    nx.draw_networkx_labels(graph, pos, font_size=10, labels=node_dict)

    # Draw labels for groups
    nx.draw_networkx_labels(graph, pos, font_size=10, labels=group_dict)

    if mode == 'rst':

        # Draw labels for nodes representing for RST relations
        nx.draw_networkx_labels(graph, pos, font_size=10, labels=rel_dict)

        # Draw edge labels for nuclei and satellites
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_dict)

    # Remove margins from the graph and axes from the plot
    fig.tight_layout(pad=0)
    plt.axis('off')

    # Save figure to file, read the file using OpenCV and remove the file
    plt.savefig('temp.png')
    img = cv2.imread('temp.png')
    os.remove('temp.png')

    # Close matplotlib figure
    plt.close()

    return img


def draw_layout(path_to_image, annotation, height, hide=False, **kwargs):
    """
    Visualizes the AI2D layout annotation on the original input image.

    Parameters:
        path_to_image: Path to the original AI2D diagram image.
        annotation: A dictionary containing AI2D annotation.
        height: Target height of the image.
        hide: A Boolean indicating whether to draw annotation or not.

    Optional parameters:
        dpi: An integer indicating the resolution to use.
        point: A list of layout elements to draw.

    Returns:
        An image with the AI2D annotation overlaid.
    """

    # Load the diagram image and make a copy
    img, r = resize_img(path_to_image, height)

    # Change from BGR to RGB colourspace
    img = img[:, :, ::-1]

    # Create a matplotlib Figure
    fig, ax = plt.subplots(1)
    plt.tight_layout(pad=0)

    # Add the image to the axis
    ax.imshow(img)

    # Hide grid and axes
    plt.axis('off')

    # Check if the annotation should be hidden
    if hide:

        # Save figure to file, read the file using OpenCV and remove the file
        plt.savefig('temp.png')
        img = cv2.imread('temp.png')
        os.remove('temp.png')

        return img

    # Draw blobs
    try:
        for b in annotation['blobs']:

            # Define default colour for blobs
            blob_color = 'orangered'

            # Check if some annotation should be highlighted
            if kwargs and 'point' in kwargs:

                # Continue if the blob is not in the list of elements to draw
                if b not in kwargs['point']:

                    continue

            # Check if some annotation should be highlighted in different colors
            if kwargs and 'highlight' in kwargs:

                # Check that kwargs['highlight'] is a dictionary
                assert type(kwargs['highlight']) == dict

                # Assign highlights to a dictionary
                highlights = kwargs['highlight']

                # Loop over colours and elements
                for color, elements in highlights.items():

                    # If match is found, set colour and highlight to True
                    if b in elements:

                        blob_color = color
                        highlight = True

                        break

                    else:
                        highlight = False

                # If no element is to be highlighted, continue to next item
                if not highlight:

                    continue

            # Get blob ID
            blob_id = annotation['blobs'][b]['id']

            # Assign the blob points into a variable and convert into numpy
            # array
            points = np.array(annotation['blobs'][b]['polygon'], np.int32)

            # Scale the coordinates according to the ratio; convert to int
            points = np.round(points * r, decimals=0).astype('int')

            # Creat arrow polygon
            blob = patches.Polygon(points,
                                   closed=True,
                                   fill=False,
                                   alpha=1,
                                   color=blob_color)

            # Add arrow to the image
            ax.add_patch(blob)

            # Add artist for patch
            ax.add_artist(blob)

            # Get centroid
            cx, cy = np.round(points.mean(axis=0), decimals=0).astype('int')[:2]

            # If highlights have been requested, skip annotations and continue
            if 'highlight' in kwargs:

                continue

            # Annotate the blob
            ann = ax.annotate(blob_id, (cx, cy), color='white',
                              fontsize=10, ha='center', va='center')

            # Add a box around the annotation
            ann.set_bbox(dict(alpha=1, color=blob_color, pad=0))

    # Skip if there are no blobs to draw
    except KeyError:
        pass

    # Draw arrows
    try:
        for a in annotation['arrows']:

            # Define default colour for arrows
            arrow_color = 'mediumseagreen'

            # Check if some annotation should be highlighted
            if kwargs and 'point' in kwargs:

                # Continue if the blob is not in the list of elements to draw
                if a not in kwargs['point']:

                    continue

            # Check if some annotation should be highlighted in different colors
            if kwargs and 'highlight' in kwargs:

                # Check that kwargs['highlight'] is a dictionary
                assert type(kwargs['highlight']) == dict

                # Assign highlights to a dictionary
                highlights = kwargs['highlight']

                # Loop over colours and elements
                for color, elements in highlights.items():

                    # If match is found, set colour and highlight to True
                    if a in elements:

                        arrow_color = color
                        highlight = True

                        break

                    else:
                        highlight = False

                # If no element is to be highlighted, continue to next item
                if not highlight:

                    continue

            # Get arrow ID
            arrow_id = annotation['arrows'][a]['id']

            # Assign the points into a variable
            points = np.array(annotation['arrows'][a]['polygon'], np.int32)

            # Scale the coordinates according to the ratio; convert to int
            points = np.round(points * r, decimals=0).astype('int')

            # Create an arrow polygon
            arrow = patches.Polygon(points,
                                    closed=True,
                                    fill=False,
                                    alpha=1,
                                    color=arrow_color)

            # Add arrow to the image
            ax.add_patch(arrow)

            # Add artist for patch
            ax.add_artist(arrow)

            # Get centroid
            cx, cy = np.round(points.mean(axis=0), decimals=0).astype('int')[:2]

            # If highlights have been requested, skip annotations and continue
            if 'highlight' in kwargs:

                continue

            # Annotate the arrow
            ann = ax.annotate(arrow_id, (cx, cy), color='white',
                              fontsize=10, ha='center', va='center')

            # Add a box around the annotation
            ann.set_bbox(dict(alpha=1, color=arrow_color, pad=0))

    # Skip if there are no arrows to draw
    except KeyError:
        pass

    # Draw text blocks
    try:
        for t in annotation['text']:

            # Define default colour for text blocks
            text_color = 'dodgerblue'

            # Check if some annotation should be highlighted
            if kwargs and 'point' in kwargs:

                # Continue if the blob is not in the list of elements to draw
                if t not in kwargs['point']:

                    continue

            # Check if some annotation should be highlighted in different colors
            if kwargs and 'highlight' in kwargs:

                # Check that kwargs['highlight'] is a dictionary
                assert type(kwargs['highlight']) == dict

                # Assign highlights to a dictionary
                highlights = kwargs['highlight']

                # Loop over colours and elements
                for color, elements in highlights.items():

                    # If match is found, set colour and highlight to True
                    if t in elements:

                        text_color = color
                        highlight = True

                        break

                    else:
                        highlight = False

                # If no element is to be highlighted, continue to next item
                if not highlight:

                    continue

            # Get text ID
            text_id = annotation['text'][t]['id']

            # Get the start and end points of the rectangle and cast
            # them into tuples for drawing.
            rect = np.array(annotation['text'][t]['rectangle'], np.int32)

            # Get start and end coordinates, convert to int and cast into tuple
            startx, starty = np.round(rect[0] * r, decimals=0).astype('int')
            endx, endy = np.round(rect[1] * r, decimals=0).astype('int')

            # Calculate bounding box width and height
            width = endx - startx
            height = endy - starty

            # Define a rectangle and add to batch
            rectangle = patches.Rectangle((startx, starty),
                                          width, height,
                                          fill=False,
                                          alpha=1,
                                          color=text_color,
                                          edgecolor=None)

            # Add patch to the image
            ax.add_patch(rectangle)

            # Add artist object for rectangle
            ax.add_artist(rectangle)

            # Get starting coordinates
            x, y = rectangle.get_xy()

            # Get coordinates for the centre; adjust positioning
            cx = (x + rectangle.get_width() / 2.0)
            cy = (y + rectangle.get_height() / 2.0)

            # If highlights have been requested, skip annotations and continue
            if 'highlight' in kwargs:

                continue

            # Add annotation to the text box
            ann = ax.annotate(text_id, (cx, cy), color='white',
                              fontsize=10, ha='center', va='center')

            # Add a box around the annotation
            ann.set_bbox(dict(alpha=1, color=text_color, pad=0))

    # Skip if there are no text boxes to draw
    except KeyError:
        pass

    # If requested, draw arrowheads
    if kwargs and 'arrowheads' in kwargs:

        # Define colour for arrowheads
        arrowhead_color = 'darkorange'

        try:
            for ah in annotation['arrowHeads']:

                # Get arrowhead ID
                arrowhead_id = annotation['arrowHeads'][ah]['id']

                # Get the start and end points of the rectangle and cast them
                # into tuples for drawing.
                rect = np.array(annotation['arrowHeads'][ah]['rectangle'],
                                np.int32)

                # Get start and end coordinates, convert to int and cast into
                # tuple
                startx, starty = np.round(rect[0] * r, decimals=0).astype('int')
                endx, endy = np.round(rect[1] * r, decimals=0).astype('int')

                # Calculate bounding box width and height
                width = endx - startx
                height = endy - starty

                # Define a rectangle and add to batch
                rectangle = patches.Rectangle((startx, starty),
                                              width, height,
                                              fill=False,
                                              alpha=1,
                                              color=arrowhead_color,
                                              edgecolor=None)

                # Add patch to the image
                ax.add_patch(rectangle)

                # Add artist object for rectangle
                ax.add_artist(rectangle)

                # Get starting coordinates
                x, y = rectangle.get_xy()

                # Get coordinates for the centre; adjust positioning
                cx = (x + rectangle.get_width() / 2.0)
                cy = (y + rectangle.get_height() / 2.0)

                # If highlights have been requested, skip annotations and
                # continue
                if 'highlight' in kwargs:

                    continue

                # Add annotation to the text box
                ann = ax.annotate(arrowhead_id, (cx, cy), color='white',
                                  fontsize=10, ha='center', va='center')

                # Add a box around the annotation
                ann.set_bbox(dict(alpha=1, color=arrowhead_color, pad=0))

        # Skip if there are no arrowheads to draw
        except KeyError:
            pass

    # Check if a high-resolution image has been requested
    if kwargs and 'dpi' in kwargs:

        # Save figure to file in the requested resolution
        plt.savefig('temp.png', dpi=kwargs['dpi'])

        img = cv2.imread('temp.png')
        os.remove('temp.png')

        return img

    # Save figure to file, read the file using OpenCV and remove the file
    plt.savefig('temp.png')
    img = cv2.imread('temp.png')
    os.remove('temp.png')

    # Close the plot
    plt.close()

    # Return the annotated image
    return img


def draw_nodes(graph, pos, ax, node_types, draw_edges=True, mode='layout',
               **kwargs):
    """
    A generic function for visualising the nodes in a graph.

    Parameters:
        graph: A NetworkX Graph.
        pos: Positions for the NetworkX Graph.
        ax: Matplotlib Figure Axis on which to draw.
        node_types: A dictionary of node types extracted from the Graph.
        draw_edges: A boolean indicating whether edges should be drawn.
        mode: A string indicating the selected drawing mode. Valid options are
             'layout' (default), 'connectivity' and 'rst'.

    Optional parameters:
        highlight: A dictionary of identifier/colour pairs to emphasise.

    Returns:
         None
    """

    # Draw nodes for text elements
    try:
        # Retrieve text nodes for the list of nodes
        texts = [k for k, v in node_types.items() if v == 'text']

        # Add the list of nodes to the graph
        nx.draw_networkx_nodes(graph,
                               pos,
                               nodelist=texts,
                               alpha=1,
                               node_color='dodgerblue',
                               ax=ax
                               )

    # Skip if there are no text nodes to draw
    except KeyError:
        pass

    # Draw nodes for blobs
    try:
        # Retrieve blob nodes for the list of nodes
        blobs = [k for k, v in node_types.items() if v == 'blobs']

        # Add the list of nodes to the graph
        nx.draw_networkx_nodes(graph,
                               pos,
                               nodelist=blobs,
                               alpha=1,
                               node_color='orangered',
                               ax=ax
                               )

    # Skip if there are no blob nodes to draw
    except KeyError:
        pass

    # Draw nodes for arrowheads
    try:
        # Retrieve arrowhead nodes for the list of nodes
        arrowhs = [k for k, v in node_types.items() if v == 'arrowHeads']

        # Add the list of arrowheads to the graph
        nx.draw_networkx_nodes(graph,
                               pos,
                               nodelist=arrowhs,
                               alpha=1,
                               node_color='darkorange',
                               ax=ax
                               )

    # Skip if there are no arrowheads to draw
    except KeyError:
        pass

    # Draw nodes for arrows
    try:
        # Retrieve arrow nodes for the list of nodes
        arrows = [k for k, v in node_types.items() if v == 'arrows']

        # Add the list of arrows to the graph
        nx.draw_networkx_nodes(graph,
                               pos,
                               nodelist=arrows,
                               alpha=1,
                               node_color='mediumseagreen',
                               ax=ax
                               )

    # Skip if there are no arrows to draw
    except KeyError:
        pass

    # Draw nodes for image constants
    try:
        # Retrieve image constants (in most cases, only one per diagram)
        constants = [k for k, v in node_types.items() if
                     v == 'imageConsts']

        # Add the image constants to the graph
        nx.draw_networkx_nodes(graph,
                               pos,
                               nodelist=constants,
                               alpha=1,
                               node_color='palegoldenrod',
                               ax=ax
                               )

    # Skip if there are no image constants to draw
    except KeyError:
        pass

    # Draw nodes for groups
    try:
        # Retrieve group nodes from the list of nodes
        groups = [k for k, v in node_types.items() if v == 'group']

        # Add the group nodes to the graph
        nx.draw_networkx_nodes(graph,
                               pos,
                               nodelist=groups,
                               alpha=1,
                               node_color='navajowhite',
                               ax=ax
                               )

    # Skip if there are no group nodes to draw
    except KeyError:
        pass

    # Check drawing mode, continue with RST
    if mode == 'rst' and draw_edges:

        # Draw nodes for RST relations
        try:

            # Retrieve relations from the list of nodes
            relations = [k for k, v in node_types.items() if v == 'relation']

            # Check if some relations are to be highlighted
            if kwargs and 'highlight' in kwargs:

                # Assert highlight is a dict
                assert type(kwargs['highlight']) == dict

                # Draw the highlighted relations
                for relation, colour in kwargs['highlight'].items():

                    # Get the index of the relation and pop it from the list
                    relation_ix = relations.index(relation)
                    relations.pop(relation_ix)

                    # Draw the nodes
                    nx.draw_networkx_nodes(graph,
                                           pos,
                                           nodelist=relation.split(),
                                           alpha=1,
                                           linewidths=4,
                                           node_color=colour,
                                           node_shape='s',
                                           ax=ax
                                           )

            # Otherwise add the relations to the graph as usual
            nx.draw_networkx_nodes(graph,
                                   pos,
                                   nodelist=relations,
                                   alpha=1,
                                   linewidths=4,
                                   node_color='peru',
                                   node_shape='s',
                                   ax=ax
                                   )

        # Skip if there are no relations to draw
        except KeyError:
            pass

        # Get edge list
        edge_list = graph.edges(data=True)

        # Filter the edge list for satellite edges
        satellites = [(u, v, d) for (u, v, d) in edge_list
                      if d['kind'] == 'satellite']

        # Draw edges for satellites without arrows
        nx.draw_networkx_edges(graph,
                               pos,
                               satellites,
                               alpha=0.75,
                               arrows=False,
                               ax=ax)

        # Filter the edge list for nuclei edges
        nuclei = [(u, v, d) for (u, v, d) in edge_list
                  if d['kind'] == 'nucleus']

        # Draw edges for nuclei with arrows
        nx.draw_networkx_edges(graph,
                               pos,
                               nuclei,
                               alpha=0.75,
                               arrows=True,
                               ax=ax)

        # Draw grouping edges
        try:
            # Fetch a list of grouping edges (which do not have any attributes!)
            grouping_edges = [(u, v, d) for (u, v, d) in edge_list if
                              d['kind'] == 'grouping']

            # Draw edges between elements and their grouping node
            nx.draw_networkx_edges(graph,
                                   pos,
                                   grouping_edges,
                                   alpha=0.5,
                                   style='dotted',
                                   arrows=False,
                                   ax=ax)

        # Skip if no grouping edges are found
        except KeyError:

            pass

    # Check drawing mode, finish with connectivity
    if mode == 'connectivity' and draw_edges:

        # Get a list of all edges
        all_edges = list(graph.edges(data=True))

        # Draw undirectional edges
        try:

            # Filter the edges, retaining only undirectional edges
            undirectional = [(u, v, d) for u, v, d in all_edges
                             if d['kind'] == 'undirectional']

            # Draw edges without arrows
            nx.draw_networkx_edges(graph,
                                   pos,
                                   undirectional,
                                   alpha=0.75,
                                   arrows=False,
                                   ax=ax
                                   )

        # Skip if no undirectional arrows are found
        except KeyError:

            pass

        # Draw other edges
        try:

            # Filter the edges, retaining only directional/bidirectional edges
            directional = [(u, v, d) for (u, v, d) in all_edges
                           if d['kind'] in ['directional', 'bidirectional']]

            # Draw edges with arrows
            nx.draw_networkx_edges(graph,
                                   pos,
                                   directional,
                                   alpha=0.75,
                                   arrows=True,
                                   ax=ax
                                   )

        # Skip if no directional/bidirectional edges are found
        except KeyError:

            pass

        # Draw grouping edges
        try:
            # Fetch a list of grouping edges (which do not have any attributes!)
            grouping_edges = [(u, v, d) for (u, v, d) in all_edges if
                              d['kind'] == 'grouping']

            # Draw edges between elements and their grouping nodes
            nx.draw_networkx_edges(graph,
                                   pos,
                                   grouping_edges,
                                   alpha=0.5,
                                   style='dotted',
                                   arrows=False,
                                   ax=ax)

        # Skip if no grouping edges are found
        except KeyError:

            pass

    # Otherwise, draw standard edges if requested
    if draw_edges and mode == 'layout':

        # Draw edges between nodes
        nx.draw_networkx_edges(graph,
                               pos,
                               alpha=0.75,
                               ax=ax
                               )


def resize_img(path_to_image, height):
    """
    Resizes an image.

    Parameters:
        path_to_image: Path to the image to resize.
        height: Requested height of the resized image.

    Returns:
        The resized image and the ratio used for resizing.
    """

    # Load the diagram image and make a copy
    img = cv2.imread(path_to_image).copy()

    # Calculate aspect ratio (target width / current width) and new
    # width of the preview image.
    (h, w) = img.shape[:2]

    # Calculate ratio based on image height
    r = height / h
    dim = (int(w * r), height)

    # Resize the preview image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Return image
    return img, r


def highlight(element, highlight):

    # Check that highlight is a dictionary
    assert type(highlight) == dict

    # Loop over colours and elements
    for color, elements in highlight.items():

        # If match is found, set colour and highlight to True
        if element in elements:

            return color, True

        else:

            return None, False
