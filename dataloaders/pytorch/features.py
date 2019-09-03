# -*- coding: utf-8 -*-

"""
This file contains functions for loading the AI2D and AI2D-RST annotations and for
extracting simple layout features from the layout segmentation using OpenCV.
"""

# Import packages
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
import cv2
import networkx as nx
import numpy as np
import json

# Define one-hot vectors for diagram elements and relations manually
node_types = {'imageConsts': np.array([1, 0, 0, 0, 0, 0], dtype=np.float32),
              'arrows': np.array([0, 1, 0, 0, 0, 0], dtype=np.float32),
              'group': np.array([0, 0, 1, 0, 0, 0], dtype=np.float32),
              'text': np.array([0, 0, 0, 1, 0, 0], dtype=np.float32),
              'blobs': np.array([0, 0, 0, 0, 1, 0], dtype=np.float32),
              'relation': np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
              }

# Do the same for connections and nuclearity in RST relations
conn_types = {'directional': np.array([1, 0, 0], dtype=np.float32),
              'undirectional': np.array([0, 1, 0], dtype=np.float32),
              'bidirectional': np.array([0, 0, 1], dtype=np.float32)
              }
nuc_types = {'nucleus': np.array([1, 0], dtype=np.float32),
             'satellite': np.array([0, 1], dtype=np.float32)}

# Define a list of RST relations used in AI2D-RST
rst_list = ['identification', 'elaboration', 'joint', 'property-ascription',
            'preparation', 'cyclic sequence', 'connected', 'circumstance',
            'class-ascription', 'contrast', 'sequence', 'list',
            'nonvolitional-result', 'nonvolitional-cause', 'restatement',
            'background', 'conjunction', 'means', 'disjunction',
            'condition', 'enablement', 'volitional-result',
            'solutionhood', 'unless', 'justify']

# Initialize label encoder and fit to the list of RST relations
label_enc = LabelEncoder()
label_enc.fit_transform(rst_list)

# Transform the original list of relations into integer labels
int_labels = label_enc.transform(rst_list)

# Initialize binary label encoder and fit to the array of integer labels
bin_enc = LabelBinarizer()
bin_enc.fit(int_labels)

# Create a dictionary with RST relation names and their one-hot labels by
# zipping the label strings and their binary representations
rst_enc = dict(zip(label_enc.classes_, bin_enc.transform(int_labels)))


def create_graph(annotation, image, json, layers):
    """
    Creates a graph using the original AI2D annotation and AI2D-RST.

    Parameters:
        annotation: A dictionary containing parsed AI2D annotation.
        image: The diagram's image for reference.
        json: AI2D-RST JSON data.
        layers: A list of annotation layers to be returned as graphs.

    Returns:
        A dictionary of NetworkX graphs for AI2D-RST annotation.
    """

    # Set up a placeholder dictionary for graphs
    graphs = {}

    # Check which annotation layers are to be converted into graphs
    if 'grouping' in layers:

        # Load grouping annotation
        graphs['grouping'] = load_ai2d_rst(json)[0]

    if 'connectivity' in layers:

        # Load connectivity annotation
        graphs['connectivity'] = load_ai2d_rst(json)[1]

    if 'discourse' in layers:

        # Load discourse annotation
        graphs['discourse'] = load_ai2d_rst(json)[2]

    # Extract elements and their features from the original AI2D annotation
    graphs = {k: extract_features(v, annotation, image, layers)
              for k, v in graphs.items()}

    # Return graphs
    return graphs


def extract_features(graph, annotation, image, layers):
    """
    Extracts features from the original AI2D annotation and adds them to the
    AI2D-RST graphs.

    Parameters:
        graph: An AI2D-RST graph (grouping, connectivity or RST)
        annotation: A dictionary containing the original AI2D annotation.
        image: An image of the image from the original AI2D dataset.
        layers: A list of annotation layers to create.

    Returns:
        An AI2D-RST graph with updated features.
    """
    # Fetch the nodes from the graph
    nodes = graph.nodes(data=True)

    # Get height and width of the diagram image
    h, w = image.shape[:2]

    # Get the number of pixels in the image
    n_pix = h * w

    # Set up a placeholder dictionaries to hold the updated node and edge
    # features
    node_features = {}
    edge_features = {}

    # Loop over the nodes and their features
    for node, features in nodes:

        # Assign the node type to variable 'kind'
        kind = features['kind']

        # Check for split nodes in RST graph - these occur if the same element
        # participates in multiple RST relations
        if 'copy_of' in features.keys():

            # Fetch the parent node identifier from the copy_of attribute
            node_id = features['copy_of']

        # If the node has not been split, use the original node identifier
        else:
            node_id = node

        # If the node is a blob, text block or arrow, fetch the node features
        # from the original AI2D annotation
        if kind in annotation.keys():

            # Get element coordinates for different element types
            if kind in ['text']:

                # Assign coordinates to a variable
                coords = np.array(annotation[kind][node_id]['rectangle'],
                                  np.int32)

                # Calculate element area via the cv2.boundingRect, as the
                # cv2.contourArea does not work with xy-coordinates
                (rect_x, rect_y, rect_w, rect_h) = cv2.boundingRect(coords)

                # Calculate element area
                area = (rect_w * rect_h) / float(n_pix)

                # Calculate solidity, or simply assign 1, as rectangles are
                # always solid
                solidity = 1.0

                # Get centre point
                centre = (int(np.round(rect_x + (rect_w / 2), decimals=0)),
                          int(np.round(rect_y + (rect_h / 2), decimals=0)))

            if kind in ['arrows', 'blobs']:

                # Fetch coordinates and convert to numpy array
                coords = np.array(annotation[kind][node_id]['polygon'],
                                  np.int32)

                # Calculate element area
                area = cv2.contourArea(coords)

                # Calculate convex hull and its area
                hull = cv2.convexHull(coords)
                hull_area = cv2.contourArea(hull)

                # Calculate solidity (area divided by convex hull area)
                solidity = area / hull_area

                # Normalize element area by dividing by total pixel count
                area = area / float(n_pix)

                # Calculate moments for finding the centroid
                moments = cv2.moments(coords)

                # Calculate centroid
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])

                # Assign centre point to a variable
                centre = [centroid_x, centroid_y]

            # Normalize coordinates to [0..1] by dividing by width and height
            norm_coords = np.array([centre[0] / w, centre[1] / h], np.float32)

        # If the node is an abstraction, such as a group or an image constant,
        # set the features to zero.
        if kind in ['group', 'imageConsts', 'relation']:

            # Assign empty coordinates to a variable
            norm_coords = [0, 0]
            area = 0
            solidity = 0

        # Concatenate all features
        features = np.concatenate((node_types[kind], norm_coords,
                                   [area], [solidity]), axis=0)

        # Add features to the updated dictionary
        node_features[node] = {'features': features}

    # Finally, set the updated features as the graph attributes
    nx.set_node_attributes(graph, node_features)

    # Begin processing edges in the connectivity layer (a nx.MultiDiGraph)
    if 'connectivity' in layers and type(graph) == nx.MultiDiGraph:

        # Fetch edges from the graph – keys are required for MultiDiGraph
        edges = graph.edges(data=True, keys=True)

        # Loop over the edges
        for src, dst, key, features in edges:

            # Update the edge features list by retrieving the one-hot encoding
            edge_features[src, dst, key] = {'kind':
                                            conn_types[features['kind']]}

        # Set updated edge features
        nx.set_edge_attributes(graph, edge_features)

    # Continue to process edges in the RST layer (a nx.DiGraph)
    if 'discourse' in layers and type(graph) == nx.DiGraph:

        # Fetch edges from the graph
        edges = graph.edges(data=True)

        # Loop over the edges
        for src, dst, features in edges:

            # Update the edge features list by retrieving the one-hot encoding
            edge_features[src, dst] = {'kind': nuc_types[features['kind']]}

        # Set updated edge features
        nx.set_edge_attributes(graph, edge_features)

    # Return the updated graph
    return graph


def load_ai2d_rst(data):
    """"
    Creates NetworkX graphs from provided JSON data.

    Parameters:
        data: A dictionary loaded from AI2D-RST JSON.

    Returns:
        Grouping, connectivity and RST graphs as NetworkX graphs
    """
    # Separate dictionaries for each layer from the JSON dictionary
    grouping_dict_from_json = data['grouping']
    conn_dict_from_json = data['connectivity']
    rst_dict_from_json = data['rst']

    # Create the grouping graph using the nx.jit_graph function
    grouping_graph = nx.jit_graph(grouping_dict_from_json,
                                  create_using=nx.Graph())

    # Create connectivity graph manually
    connectivity_graph = nx.MultiDiGraph()

    # Load nodes and edges
    nodes = conn_dict_from_json['nodes']
    edges = conn_dict_from_json['edges']

    # Add nodes manually to the connectivity graph
    for node in nodes:

        connectivity_graph.add_node(node[0], kind=node[1]['kind'])

    # Add edges manually to the connectivity graph
    for edge in edges:

        connectivity_graph.add_edge(edge[0], edge[1], kind=edge[2]['kind'])

    # Create the RST graph using the nx.jit_graph function
    rst_graph = nx.jit_graph(rst_dict_from_json, create_using=nx.DiGraph())

    # Return all three graphs
    return grouping_graph, connectivity_graph, rst_graph


def load_annotation(json_path):
    """
    Loads AI2D annotation from a JSON file and returns the annotation as a
    dictionary.

    Parameters:
         json_path: A string containing the filepath to annotation.

    Returns:
         A dictionary containing AI2D annotation.
    """
    # Open the file containing the annotation
    with open(json_path) as annotation_file:

        # Parse the AI2D annotation from the JSON file into a dictionary
        annotation = json.load(annotation_file)

    # Return the annotation
    return annotation
