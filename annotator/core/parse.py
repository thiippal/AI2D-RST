# -*- coding: utf-8 -*-

import networkx as nx
import json


def create_graph(annotation, edges=False, arrowheads=False, mode='layout'):
    """
    Draws an initial graph of diagram elements parsed from AI2D annotation.

    Parameters:
        annotation: A dictionary containing parsed AI2D annotation.
        edges: A boolean defining whether edges are to be drawn.
        arrowheads: A boolean defining whether arrowheads are drawn.
        mode: A string indicating the diagram structure to be drawn, valid 
              options include 'layout', 'connect' and 'rst'. Default mode is
              layout.

    Returns:
        A networkx graph with diagram elements.
    """
    # Check the mode attribute to determine the correct Graph type
    if mode == 'layout':

        # Create an undirected graph for layout annotation
        graph = nx.Graph()

    if mode == 'connectivity':

        # Create a directed graph with multiple edges for connectivity
        graph = nx.MultiDiGraph()

    if mode == 'rst':

        # Create a directed graph for RST annotation
        graph = nx.DiGraph()

    # Check the input type
    if type(annotation) == list:

        # Populate the graph with the node list
        graph.add_nodes_from(annotation)

        return graph

    # If the input is a dictionary, assume the input is parsed AI2D annotation
    if type(annotation) == dict:

        # Parse the annotation from the dictionary
        diagram_elements, relations = parse_annotation(annotation, mode=mode)

        # Extract element types
        element_types = extract_types(diagram_elements, annotation)

        # Check if arrowheads should be excluded
        if not arrowheads:

            # Remove arrowheads from the dictionary
            element_types = {k: v for k, v in element_types.items()
                             if v != 'arrowHeads'}

        # Set up a dictionary to track arrows and arrowheads
        arrowmap = {}

        # Add diagram elements to the graph and record their type (kind)
        for element, kind in element_types.items():

            # Add node to graph
            graph.add_node(element, kind=kind)

        # Draw edges between nodes if requested
        if edges:

            # Loop over individual relations
            for relation, attributes in relations.items():

                # If the relation is 'arrowHeadTail', draw an edge between the
                # arrow and its head
                if attributes['category'] == 'arrowHeadTail':

                    # Add edge to graph
                    graph.add_edge(attributes['origin'],
                                   attributes['destination'])

                    # Add arrowhead information to the dict for tracking arrows
                    arrowmap[attributes['origin']] = attributes['destination']

                # Next, check if the relation includes a connector
                try:
                    if attributes['connector']:

                        # Check if the connector (arrow) has an arrowhead
                        if attributes['connector'] in arrowmap.keys():

                            # First, draw an edge between origin and connector
                            graph.add_edge(attributes['origin'],
                                           attributes['connector'])

                            # Then draw an edge between arrowhead and
                            # destination, fetching the arrowhead identifier
                            # from the dictionary
                            graph.add_edge(arrowmap[attributes['connector']],
                                           attributes['destination'])

                        else:
                            # If the connector does not have an arrowhead, draw
                            # edge from origin to destination via the connector
                            graph.add_edge(attributes['origin'],
                                           attributes['connector'])

                            graph.add_edge(attributes['connector'],
                                           attributes['destination'])

                # If connector does not exist, draw a normal relation between
                # the origin and the destination
                except KeyError:
                    graph.add_edge(attributes['origin'],
                                   attributes['destination'])

        # Return graph
        return graph


def extract_types(elements, annotation):
    """
    Extracts the types of the identified diagram elements.

    Parameters:
        elements: A list of diagram elements.
        annotation: A dictionary of AI2D annotation.

    Returns:
         A dictionary with element types as keys and identifiers as values.
    """
    # Check for correct input type
    assert isinstance(elements, list)
    assert isinstance(annotation, dict)

    # Define the target categories for various diagram elements
    targets = ['arrowHeads', 'arrows', 'blobs', 'text', 'containers',
               'imageConsts']

    # Create a dictionary for holding element types
    element_types = {}

    # Loop over the diagram elements
    for e in elements:

        try:
            # Search for matches in the target categories
            for t in targets:

                # Get the identifiers for each element category
                ids = [i for i in annotation[t].keys()]

                # If the element is found among the identifiers, add the
                # type to the dictionary
                if e in ids:
                    element_types[e] = t

        # Skip if the category is not found
        except KeyError:
            continue

    # Return the element type dictionary
    return element_types


def get_node_dict(graph, kind=None):
    """
    A function for creating a dictionary of nodes and their kind.

    Parameters:
        graph: A NetworkX Graph.
        kind: A string defining what to include in the dictionary. 'node'
              returns only nodes and 'group' returns only groups. By
              default, the function returns all nodes defined in the graph.

    Returns:
        A dictionary with node names as keys and kind as values.
    """

    # Generate a dictionary with nodes and their kind
    node_types = nx.get_node_attributes(graph, 'kind')

    # If the requested output consists of node groups, return group dict
    if kind == 'group':

        # Generate a dictionary of groups
        group_dict = {k: k for k, v in node_types.items() if
                      v == 'group'}

        # Return dictionary
        return group_dict

    # If the requested output consists of nodes, return node dict
    if kind == 'node':

        # Generate a dictionary of nodes
        node_dict = {k: k for k, v in node_types.items() if v not in
                     ['group', 'relation']}

        # Return dictionary
        return node_dict

    # If the requested output consists of relations, return relation dict
    if kind == 'relation':

        # Generate a dictionary of RST relations
        rel_dict = {k: k for k, v in node_types.items() if v ==
                    'relation'}

        # Return dictionary
        return rel_dict

    # Otherwise return all node types
    else:
        return node_types


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


def parse_annotation(annotation, mode='layout'):
    """
    Parses AI2D annotation stored in a dictionary and prepares the annotation 
    for drawing a graph.

    Parameters:
        annotation: A dictionary containing AI2D annotation.
        mode: A string indicating the diagram structure to be drawn, valid 
              options include 'layout', 'connect' and 'rst'. Default mode is
              layout.

    Returns:
        A dictionary for drawing a graph of the annotation.
    """
    # Define target diagram elements to be added to the graph according to the
    # selected mode of processing (grouping/connectivity/rst).
    grouping_targets = ['blobs', 'arrows', 'text', 'arrowHeads', 'containers',
                        'imageConsts']
    conn_targets = ['blobs', 'arrows', 'text', 'containers', 'imageConsts']
    rst_targets = ['blobs', 'arrows', 'text']

    # Check the processing mode
    if mode == 'layout':

        # Target the list of layout elements
        targets = grouping_targets

    if mode == 'connect':

        # Target the list of connectivity elements
        targets = conn_targets

    if mode == 'rst':

        # Target the list of RST elements
        targets = rst_targets

    try:
        # Parse the diagram elements defined in the annotation, cast into list
        diagram_elements = [list(annotation[t].keys()) for t in targets]

        # Filter empty diagram types
        diagram_elements = list(filter(None, diagram_elements))

        # Flatten the resulting list
        diagram_elements = [i for sublist in diagram_elements
                            for i in sublist]

    except KeyError:
        pass

    # Parse the semantic relations defined in the annotation into a dict
    try:
        relations = annotation['relationships']

    except KeyError:
        pass

    return diagram_elements, relations


def prepare_input(input_str, from_item):
    """
    A function for preparing input for validation against a graph.

    Parameters:
        input_str: A string containing node or group identifiers.
        from_item: Start processing only after this item.

    Returns:
        A list of node identifiers.
    """

    # Get the list of nodes provided by the user
    input_list = input_str.lower().split()[from_item:]

    # Strip commas
    input_list = [u.strip(',') for u in input_list]

    # Strip extra whitespace
    input_list = [u.strip() for u in input_list]

    # Create a placeholder for the final list of identifiers
    final_list = []

    # Check if the input contains group aliases
    for i in input_list:

        # If the input contains a range of identifiers, unpack
        if ':' in i:

            # Get the prefix of the alias (I, B, T, or G)
            prefix = i[0]

            # Get numbers and cast: ignore the first character of the first id
            try:
                start, end = int(i.split(':')[0][1:]), int(i.split(':')[1])

            except ValueError:

                # Print error message
                print("[ERROR] Check syntax for identifier range: do not add "
                      "identifier prefix to the second part, i.e. g1:g5.")

                # Append erroneous identifier to the list to catch the error
                final_list.append(i)

                continue

            # Create a list of unpacked identifiers in the range
            unpacked = [prefix + str(x) for x in range(start, end + 1)]

            # Extend the list of identifiers
            final_list.extend(unpacked)

        # Otherwise, append identifier to the final list
        if ':' not in i:
            final_list.append(i)

    return final_list


def validate_input(user_input, current_graph, **kwargs):
    """
    A function for validating user input against the nodes of a NetworkX graph.

    Parameters:
        user_input: A list of nodes provided by the user.
        current_graph: The current NetworkX graph.

    Returns:
        True or False depending on whether the input is valid.
    """

    # Generate a list of valid elements present in the graph
    valid_nodes = [n.lower() for n in current_graph.nodes]

    # Forms the initial set of nodes, which may be updated using optional flags
    valid_elems = set(valid_nodes)

    # Check for optional keywords and arguments, begin by checking if groups
    # need to be validated as well
    try:

        if kwargs['groups']:

            # Generate a dictionary of groups present in the graph
            group_dict = get_node_dict(current_graph, kind='group')

            # Count the current groups and enumerate for convenience. This
            # allows the user to refer to group number instead of complex
            # identifier.
            group_dict = {"g{}".format(i): k for i, (k, v) in
                          enumerate(group_dict.items(), start=1)}

            # Create a list of valid identifiers based on group dictionary keys
            valid_groups = [g.lower() for g in group_dict.keys()]

            # Add valid groups to the set of valid elements
            valid_elems.update(valid_groups)

    except KeyError:

        pass

    # Check if RST relations need to be validated as well
    try:

        if kwargs['rst']:

            # Generate a dictionary of RST relations present in the graph
            relation_ix = get_node_dict(current_graph, kind='relation')

            # Loop through current RST relations and rename for convenience.
            # This allows the user to refer to the relation identifier (e.g. r1)
            # instead of complex relation ID (e.g. B0-T1+B9) during annotation.
            relation_ix = {"R{}".format(i): k for i, (k, v) in
                           enumerate(relation_ix.items(), start=1)}

            # Create a list of valid relation identifiers based on the dict keys
            valid_rels = [r.lower() for r in relation_ix.keys()]

            # Add valid relations to the set of valid elements
            valid_elems.update(valid_rels)

    except KeyError:

        pass

    # Check for invalid input by comparing the user input and the valid elements
    if not set(user_input).issubset(valid_elems):

        # Get the difference in sets
        diff = set(user_input).difference(valid_elems)

        # Print an error message with difference in sets
        print("[ERROR] Sorry, {} is not a valid diagram element or command."
              " Please try again.".format(' '.join(diff)))

        # Return validation flag
        return False

    # If the input is valid
    if set(user_input).issubset(valid_elems):

        # Return validation flag
        return True


def replace_aliases(current_graph, kind='group'):
    """
    A function for replacing aliases used for identifiers in the tool.

    Parameters:
        current_graph: A NetworkX graph.
        kind: A string indicating the type of alias used ('group' or 'relation')

    Returns:
         A dictionary mapping group aliases to actual group identifiers.
    """

    # Generate a dictionary of nodes present in the graph
    gd = get_node_dict(current_graph, kind=kind)

    # Count the current nodes and enumerate for convenience
    gd = {"{}{}".format(kind[0], i):
          k for i, (k, v) in enumerate(gd.items(), start=1)}

    # Return the group dictionary
    return gd


def update_grouping(diagram, graph):
    """
    Updates a graph after switches between annotation tasks. This means removing
    obsolete grouping nodes and edges, which have been removed from the grouping
    annotation.

    Parameters:
        diagram: A Diagram object.
        graph: A graph of the Diagram object that is currently being annotated.

    Returns:
        A NetworkX graph with updated grouping edges.
    """

    # Get a lists of existing nodes and edge tuples
    edge_bunch = list(graph.edges(data=True))
    nodes = dict(graph.nodes(data=True))

    # Collect grouping edges
    edge_bunch = [(u, v) for (u, v, d) in edge_bunch if d['kind'] == 'grouping']

    # Remove grouping edges from current graph
    graph.remove_edges_from(edge_bunch)

    # Use the isolates function to locate obsolete grouping nodes
    isolates = list(nx.isolates(graph))

    # Remove isolated grouping nodes
    isolates = [i for i in isolates if nodes[i]['kind'] == 'group']

    # Remove isolated nodes from the graph
    graph.remove_nodes_from(isolates)

    # Create a temporary copy of the layout graph for filtering content
    temp_graph = diagram.layout_graph.copy()

    # Get a dictionary of nodes and a list of edges
    nodes = dict(temp_graph.nodes(data=True))
    edges = list(temp_graph.edges())

    # Fetch a list of edges to/from imageConsts
    iconst_edges = [(s, t) for (s, t) in edges
                    if nodes[s]['kind'] == 'imageConsts'
                    or nodes[t]['kind'] == 'imageConsts']

    # Remove grouping edges using the list
    temp_graph.remove_edges_from(iconst_edges)

    # Use the isolates function to locate grouping nodes for groups
    isolates = list(nx.isolates(temp_graph))

    # Remove isolated grouping nodes
    isolates = [i for i in isolates if nodes[i]['kind']
                in ['group', 'imageConsts']]

    # Remove isolated nodes from the graph
    temp_graph.remove_nodes_from(isolates)

    # Add attributes to the remaining edges
    nx.set_edge_attributes(temp_graph, 'grouping', 'kind')

    # Add the filtered nodes and edges to the graph
    graph.add_nodes_from(temp_graph.nodes(data=True))
    graph.add_edges_from(temp_graph.edges(data=True))


