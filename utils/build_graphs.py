import networkx as nx
import argparse
import json
import glob
import os


def json_to_nx(json_f):
    """"
    A convenience function for creating NetworkX graphs from AI2D-RST JSON
    annotation.

    Parameters:
        json_f: A JSON file containing AI2D-RST annotation. 

    Returns:
        Three NetworkX graphs, one for each annotation layer.
    """

    # Read the JSON file
    with open(json_f) as json_file:

        # Assign the result into a dictionary
        data = json.load(json_file)

    # Fetch the annotation from the dictionary and assign to variables
    json_grouping = data['grouping']
    json_conn = data['connectivity']
    json_rst = data['rst']

    # Use the NetworkX function for JIT JSON to create the grouping graph
    grouping_layer = nx.jit_graph(json_grouping, create_using=nx.Graph())

    # Check if connectivity annotation exists
    if json_conn is not None:

        # Create the connectivity layer manually
        connectivity_layer = nx.MultiDiGraph()

        nodes = json_conn['nodes']
        edges = json_conn['edges']

        for node in nodes:
            connectivity_layer.add_node(node[0], kind=node[1]['kind'])

        for edge in edges:
            connectivity_layer.add_edge(edge[0], edge[1], kind=edge[2]['kind'])

    else:
        connectivity_layer = None

    # Use the NetworkX function for JIT JSON to create the RST graph
    rst_layer = nx.readwrite.jit_graph(json_rst, create_using=nx.DiGraph())

    return grouping_layer, connectivity_layer, rst_layer


# Test the json_to_nx() function
if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", required=True,
                    help="Path to directory containing AI2D-RST JSON files.")

    args = vars(ap.parse_args())
    input_dir = args['input']

    for f in glob.glob(os.path.join(input_dir, '*.json')):

        print("[INFO] Now processing file {} ...".format(f))

        grouping_layer, connectivity_layer, rst_layer = json_to_nx(f)
