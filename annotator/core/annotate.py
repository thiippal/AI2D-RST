# -*- coding: utf-8 -*-

import string
import random
from .interface import *
from .parse import *


def create_id(length=6, chars=string.ascii_uppercase+string.digits):
    """
    A function for creating random identifiers for grouping identifiers, which
    can become big enough to slowing down the drawing of graphs.

    Parameters:
        length: The requested length of the identifier.
        chars: The characters used to generate the random identifier.

    Returns:
        A random identifier of requested length.
    """
    return ''.join(random.choice(chars) for x in range(length))


def create_relation(rst_graph, user_input):
    """
    A function for drawing an RST relation between several diagram elements.

    Parameters:
        rst_graph: A NetworkX Graph.
        user_input: A string containing the name of a valid RST relation.

    Returns:
         An updated NetworkX Graph.
    """
    # Retrieve the name and kind of the RST relation
    relation_name = rst_relations[user_input]['name']
    relation_kind = rst_relations[user_input]['kind']

    # Generate a dictionary mapping RST relation aliases to IDs
    rel_dict = replace_aliases(rst_graph, 'relation')

    # Generate a dictionary mapping group aliases to IDs
    group_dict = replace_aliases(rst_graph, 'group')

    # Check whether the RST relation is mono- or multinuclear. Start with
    # mononuclear relations.
    if relation_kind == 'mono':

        # Request the identifier of the nucleus in the RST relation
        nucleus = input(prompts['nucleus_id'])

        # Prepare and validate input
        nucleus = prepare_input(nucleus, 0)
        valid = validate_input(nucleus, rst_graph, rst=True, groups=True)

        # Check the total number of inputs in the list
        if len(nucleus) != 1:

            # Print error message and return
            print(messages['nucleus_err'])

            return

        # The input is invalid, return: otherwise continue to process satellites
        if not valid:

            return

        # Request the identifier(s) of the satellite(s) in the RST relation
        satellites = input(prompts['satellite_id'])

        # Prepare and validate input
        satellites = prepare_input(satellites, 0)
        valid = validate_input(satellites, rst_graph, rst=True, groups=True)

        # Proceed if the input is valid
        if valid:

            # Replace group aliases with valid IDs for nuclei and satellites
            nucleus = [group_dict[n] if n in group_dict.keys() else n for n
                       in nucleus]

            satellites = [group_dict[s] if s in group_dict.keys() else s for s
                          in satellites]

            # Replace relation aliases with valid IDs for nuclei and satellites
            nucleus = [rel_dict[n] if n in rel_dict.keys() else n for n in
                       nucleus]

            satellites = [rel_dict[s] if s in rel_dict.keys() else s for s in
                          satellites]

            # Generate an ID for the new relation
            new_rel_id = create_id()

            # Add a new node to the graph to represent the RST relation
            rst_graph.add_node(new_rel_id,
                               kind='relation',
                               nucleus=' '.join(nucleus).upper(),
                               satellites=' '.join(satellites).upper(),
                               rel_name=relation_name,
                               id=new_rel_id
                               )

            # Draw edges from satellite(s) to the current RST relation
            for s in satellites:

                # Check if the satellites include another RST relation
                if s in rel_dict.keys():

                    # Fetch the origin node from the dictionary of relations
                    satellite_rel = rel_dict[s]

                    # Add edge from satellite relation to the new relation
                    rst_graph.add_edge(satellite_rel, new_rel_id,
                                       kind='satellite')

                # If the satellite is not a relation, draw edge from node
                else:

                    # Add edge to graph
                    rst_graph.add_edge(s.upper(), new_rel_id,
                                       kind='satellite')

            # Draw edges from nucleus to relation
            for n in nucleus:

                # Check if the nucleus is a RST relation
                if n in rel_dict.keys():

                    # Fetch the origin node from the dictionary of relations
                    nuclei_rel = rel_dict[n]

                    # Add edge from satellite relation to the new relation
                    rst_graph.add_edge(new_rel_id, nuclei_rel,
                                       kind='nucleus')

                # If the nucleus is not a relation, draw edge from node
                else:

                    # Add edge to graph
                    rst_graph.add_edge(new_rel_id, n.upper(),
                                       kind='nucleus')

    # Continue by checking if the relation is multinuclear
    if relation_kind == 'multi':

        # Request the identifiers of the nuclei in the RST relation
        nuclei = input(prompts['nuclei_id'])

        # Prepare and validate input
        nuclei = prepare_input(nuclei, 0)
        valid = validate_input(nuclei, rst_graph, rst=True, groups=True)

        # Check the total number of inputs in the list
        if len(nuclei) <= 1:

            # Print error message and return
            print(messages['nuclei_err'])

            return

        # The input is invalid, return: otherwise continue to process nuclei
        if not valid:

            return

        # If the input is valid, continue to draw the relations
        if valid:

            # Replace aliases with valid identifiers
            nuclei = [group_dict[n] if n in group_dict.keys() else n for n
                      in nuclei]

            nuclei = [rel_dict[n] if n in rel_dict.keys() else n for n in
                      nuclei]

            # Generate an ID for the new relation
            new_rel_id = create_id()

            # Add a new node to the graph to represent the RST relation
            rst_graph.add_node(new_rel_id,
                               kind='relation',
                               nuclei=' '.join(nuclei).upper(),
                               rel_name=relation_name,
                               id=new_rel_id
                               )

            # Draw edges from nuclei to the current RST relation
            for n in nuclei:

                # Check if the nuclei include another RST relation
                if n in rel_dict.keys():

                    # Fetch the origin node from the relation index
                    origin = rel_dict[n]

                    # Add edge from the RST relation to the nuclei
                    rst_graph.add_edge(new_rel_id, origin,
                                       kind='nucleus')

                # If all nuclei are nodes, draw edges from relation to
                # nuclei
                else:

                    # Add edge to graph
                    rst_graph.add_edge(new_rel_id, n.upper(),
                                       kind='nucleus')


def group_nodes(graph, user_input):
    """
    A function for grouping together nodes of a graph, which are included in
    the accompanying list.

    Parameters:
        graph: A NetworkX Graph.
        user_input: A list of valid nodes in the graph.

    Returns:
        An updated NetworkX Graph.
    """
    # Create a dictionary of nodes in the graph
    node_dict = get_node_dict(graph)

    # Check user input against the node dictionary for input types
    input_node_types = [node_dict[u.upper()] for u in user_input]

    # If the user input contains an imageConsts, do not add a node
    if 'imageConsts' in input_node_types:

        # Loop over key/value pairs
        for k, v in node_dict.items():

            # If the node is an image image constants
            if v == 'imageConsts':

                # Loop over image constants
                for valid_elem in user_input:

                    # Add edge from image constant to the element
                    graph.add_edge(valid_elem.upper(), k.upper())

    else:
        # Generate a name for the new node that joins together the elements
        # provided by the user
        new_node = create_id()

        # Add the new node to the graph
        graph.add_node(new_node, kind='group')

        # Add edges from nodes in the user input to the new node
        for valid_elem in user_input:

            graph.add_edge(valid_elem.upper(), new_node)


def macro_group(graph, user_input):
    """
    A function for assigning macro-grouping information to nodes in the
    graph.

    Parameters:
        graph: A NetworkX Graph.
        user_input: A list of valid nodes in the graph.

    Returns:
        An updated NetworkX graph.
    """
    # Request macro grouping type:
    macro_group_type = input(prompts['macro_group'])

    # Flatten a dictionary of valid macro groups and their abbreviations
    valid_macro_groups = list(macro_groups.keys()) +\
                         list(macro_groups.values()) +\
                         ['none']

    # Check for valid macro group type
    if macro_group_type not in valid_macro_groups:

        # Print error message
        print("Sorry, {} is not a valid macro group."
              " Please try again.".format(macro_group_type))

        return

    # If the input is valid, add macro grouping information to the nodes
    if macro_group_type in valid_macro_groups:

        # Check if formatting is needed in case the input is an abbreviation
        if macro_group_type in list(macro_groups.keys()):

            # Update the macro group name to the full name
            macro_group_type = macro_groups[macro_group_type]

        # Generate a matching list of grouping information
        group_list = [macro_group_type for x in range(0, len(user_input))]

        # Convert user input to uppercase
        user_input = [x.upper() for x in user_input]

        # Create a dictionary from user input; convert to uppercase
        macro_grouping = dict(zip(user_input, group_list))

        # Loop over macro-groups to check whether they include tables
        for node, macro_group in macro_grouping.copy().items():

            if macro_group == 'none':

                # Delete node attribute and item from macro_grouping dictionary
                try:
                    del graph.node[node]['macro_group']
                    del macro_grouping[node]

                    # Print status
                    print("[INFO] Deleted macro-group information from"
                          " {}.".format(node))

                # Catch error from missing macro-group information
                except KeyError:

                    # Print error
                    print("[INFO] Sorry, node {} has not been assigned with a "
                          "macro-group. Please check your input.".format(node))

                    return

            # If macro-group is a table, request description
            if macro_group == 'table':

                # Create a dictionary mapping group aliases to valid IDs. This
                # is needed for converting possible aliases in table cells.
                group_dict = replace_aliases(graph, 'group')

                # Prompt user for table properties and cast into integers
                try:
                    table_rows = int(input(prompts['table_rows']))

                # Catch error from invalid input type
                except ValueError:

                    # Print error message and return
                    print("[ERROR] Sorry, {} is not an integer."
                          .format(table_rows))

                    return

                try:
                    table_cols = int(input(prompts['table_cols']))

                # Catch error from invalid input type
                except ValueError:

                    # Print error message and return
                    print("[ERROR] Sorry, {} is not an integer."
                          .format(table_cols))

                    return

                try:
                    table_axes = int(input(prompts['table_axes']))

                # Catch error from invalid input type
                except ValueError:

                    # Print error message and return
                    print("[ERROR] Sorry, {} is not an integer."
                          .format(table_axes))

                    return

                # Store table shape into a dict with a string (rows, columns)
                shape_dict = {node: ', '.join([str(table_rows),
                                               str(table_cols)])}

                # Store row information into a dictionary
                row_dict = {}

                # Add table shape attribute to the node
                nx.set_node_attributes(graph, shape_dict, 'table_shape')

                # Prompt user to fill the table rows
                for x in range(1, table_rows + 1):

                    # Set up a flag to track row completion
                    row_complete = False

                    # Keep requesting input for each row until complete
                    while row_complete is False:

                        # Get identifiers for each row
                        row = input("[GROUPING] Please enter identifiers for "
                                    "elements on row {} (use ; to separate "
                                    "identifiers for each row): ".format(x))

                        # Compare the number of identifiers and columns
                        if len(row.split(';')) == table_cols:

                            # Split the input into entries for each column
                            row_entries = row.split(';')

                            # Strip whitespace
                            row_entries = [r.strip() for r in row_entries]

                            # Set up a placeholder for validation results
                            validation = set()

                            # Prepare and validate each entry
                            for i, entry in enumerate(row_entries):

                                # Prepare the input for validation
                                entry = prepare_input(entry, from_item=0)

                                # Validate the input for both nodes and groups
                                valid = validate_input(entry, graph,
                                                       groups=True)

                                # Add validation result to the set
                                validation.add(valid)

                            # If there are no false entries, mark row complete
                            if False not in validation:

                                # Replace aliases with valid identifiers
                                row_entries = [group_dict[e]
                                               if e.lower() in group_dict.keys()
                                               else e for e in row_entries]

                                # Make sure all row entries are uppercase
                                row_entries = [r.upper() for r in row_entries]

                                # Add row entries to row dict
                                row_dict['row_{}'.format(x)] = ' '\
                                    .join(row_entries)

                                # Mark row complete
                                row_complete = True

                # Format table data by unpacking the dictionary into a list
                row_dict = [str(k) + ': ' + str(v) for k, v in row_dict.items()]

                # Join the list into a single string separated by commas
                row_dict = ', '.join(row_dict)

                # Assign table data to a dictionary for setting node properties
                table_data = {node: row_dict}

                # Add table information to node
                nx.set_node_attributes(graph, table_data, 'table_data')

                # Check if table has axis labels
                if table_axes > 0:

                    # Set a placeholder dict for axis labels
                    axis_labels = {}

                    # Prompt user for axis labels
                    for x in range(1, table_axes + 1):

                        # Get axis labels
                        axis_label = input("[GROUPING] Please enter the "
                                           "identifiers for labels on axis {}: "
                                           .format(x))

                        # Prepare the input for validation
                        axis_label = prepare_input(axis_label, from_item=0)

                        # Validate the input for both nodes and groups
                        valid = validate_input(axis_label, graph, groups=True)

                        # If the the input is valid, proceed
                        if valid:

                            # Replace aliases with valid identifiers
                            axis_label = [group_dict[l] if l.lower() in
                                          group_dict.keys()
                                          else l for l in axis_label]

                            # Convert to uppercase
                            axis_label = [l.upper() for l in axis_label]

                            # Make sure all row entries are uppercase
                            row_entries = [r.upper() for r in row_entries]

                            # Add axis label to the dictionary
                            axis_labels['axis_{}'.format(x)] = ' '.join(
                                axis_label)

                    # Format table data by unpacking the dictionary into a list
                    axis_labels = [str(k) + ': ' + str(v) for k, v in
                                   axis_labels.items()]

                    # Join the list into a single string separated by commas
                    axis_labels = ', '.join(axis_labels)

                    # Assign table data to a dictionary for setting node
                    # properties.
                    label_data = {node: axis_labels}

                    # Add table information to node
                    nx.set_node_attributes(graph, label_data, 'table_labels')

        # Add macro grouping information to the graph nodes
        nx.set_node_attributes(graph, macro_grouping, 'macro_group')

        # Print status message
        print("[INFO] Updated macro-group information for {}.".format(
            ', '.join(user_input)))
