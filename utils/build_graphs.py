from networkx.readwrite import json_graph
from pathlib import Path
import networkx as nx
import argparse
import json
import os


def json_to_nx(json_f):
    """"
    A convenience function for creating NetworkX graphs from AI2D-RST JSON
    annotation.

    Parameters:
        json_f: A JSON file containing AI2D-RST annotations as node-link data.

    Returns:
        Three NetworkX graphs, one for each annotation layer.
    """

    # Read the JSON file
    with open(json_f) as json_file:

        # Assign the result into a dictionary
        data = json.load(json_file)

    # Fetch the annotation from the dictionary and assign to variables
    grouping = json_graph.node_link_graph(data['grouping'])
    connectivity = json_graph.node_link_graph(data['connectivity']) if data['connectivity'] is not None else None
    rst = json_graph.node_link_graph(data['rst'])

    return grouping, connectivity, rst


# Test the json_to_nx() function
if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--input", required=True,
                    help="Path to directory containing AI2D-RST JSON files as node-link data.")

    args = vars(ap.parse_args())
    input_dir = Path(args['input']).glob('*.json')

    for f in input_dir:

        print("[INFO] Now processing file {} ...".format(f))

        grouping_layer, connectivity_layer, rst_layer = json_to_nx(f)
