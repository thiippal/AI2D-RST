from pathlib import Path
import argparse
import json
import networkx as nx


def convert_jit_to_node_link(json_file):
    """
    This function converts AI2D-RST annotations stored as JIT JSON into a node-link JSON format.
    This conversion is necessary, as JIT JSON support was dropped in NetworkX 3.0.

    Args:
        json_file: Path to a JSON file containing AI2D-RST annotations.

    Returns:
        Writes a file to disk containing the annotations in a node-link JSON format.
    """
    # Open the file for reading
    with open(json_file) as json_f:

        # Read the JSON string from the file
        data = json.load(json_f)

        # Load the grouping graph
        grouping = nx.jit_graph(data['grouping'], create_using=nx.Graph())

        # Check if connectivity annotation exists
        if data['connectivity'] is not None:

            # Create the connectivity graph manually
            connectivity = nx.MultiDiGraph()

            # Get nodes and edges
            nodes = data['connectivity']['nodes']
            edges = data['connectivity']['edges']

            # Add nodes manually to the graph
            for node in nodes:
                connectivity.add_node(node[0], kind=node[1]['kind'])

            # Add edges manually to the graph
            for edge in edges:
                connectivity.add_edge(edge[0], edge[1], kind=edge[2]['kind'])

        else:

            connectivity = None

        # Create the discourse structure (RST) graph
        rst = nx.jit_graph(data['rst'], create_using=nx.DiGraph())

        # Enhance grouping graph with segmentation information from AI2D
        with open(Path(ai2d_dir) / json_file.name) as ai2d_f:

            # Load AI2D annotations
            ai2d_data = json.load(ai2d_f)

            # Get and filter node types
            node_types = {k: v for k, v in nx.get_node_attributes(grouping, 'kind').items()
                          if v in ['text', 'blobs', 'arrows']}

            # Unpack and merge the dictionaries with bounding boxes for arrows, text and blobs
            bboxes = {**ai2d_data['arrows'], **ai2d_data['text'], **ai2d_data['blobs']}

            # Get bounding box for each node
            for n in node_types:

                try:
                    bbox = bboxes[n]['rectangle']

                except KeyError:
                    bbox = bboxes[n]['polygon']

                # Store the bounding box under the attribute 'bbox' in the grouping graph
                grouping.nodes[n]['bbox'] = bbox

        # Add information about AI2D and AI2D-RST categories to the graph
        try:
            with open(cats_dir / 'categories_ai2d-rst.json') as ai2d_rst_cat_f:

                ai2d_rst_cats = json.load(ai2d_rst_cat_f)
                grouping.graph['ai2d-rst_category'] = ai2d_rst_cats[json_file.stem]

        except FileNotFoundError:

            print("Error: AI2D-RST categories file not found. The file must be named 'categories_ai2d-rst.json'.")
            exit()

        try:
            with open(cats_dir / 'categories.json') as ai2d_cat_f:

                ai2d_cats = json.load(ai2d_cat_f)
                grouping.graph['ai2d_category'] = ai2d_cats[json_file.stem]

        except FileNotFoundError:

            print("Error: AI2D categories file not found. The file must be named 'categories.json'.")
            exit()

        # Convert all graphs to node-link format
        grouping = nx.node_link_data(grouping)
        connectivity = nx.node_link_data(connectivity) if connectivity is not None else None
        rst = nx.node_link_data(rst)

        # Create a dictionary to be converted into JSON
        to_json = {'id': data['id'],
                   'grouping': grouping,
                   'connectivity': connectivity,
                   'rst': rst}

        return to_json


if __name__ == '__main__':

    ap = argparse.ArgumentParser()

    ap.add_argument("-a", "--ai2d", required=True,
                    help="Path to directory containing AI2D JSON files.")
    ap.add_argument("-r", "--ai2d_rst", required=True,
                    help="Path to the directory containing AI2D-RST JSON files.")
    ap.add_argument("-c", "--cats", required=True,
                    help="Path to the directory containing AI2D and AI2D-RST category JSON files.")

    args = vars(ap.parse_args())

    ai2d_dir = Path(args['ai2d'])
    ai2d_rst_dir = Path(args['ai2d_rst'])
    cats_dir = Path(args['cats'])
    output_dir = Path('ai2d-rst_nld')

    if not output_dir.exists():

        output_dir.mkdir()

    counter = 0

    for json_in in ai2d_rst_dir.glob('*.json'):

        print(f"Now converting {json_in.name} ...")

        json_out = convert_jit_to_node_link(json_in)

        with open(output_dir / json_in.name, mode='w') as file_out:

            json.dump(json_out, file_out, indent=4)

            counter += 1

    print(f"Converted a total of {counter} files.")
