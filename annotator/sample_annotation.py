# Import packages
import argparse
import pandas as pd


def resolve_id(identifier, relations):
    """
    This function resolves the identifiers for recursive RST relations through
    the entire parse tree.

    Parameters:
        identifier: An identifier with with six characters.
        relations: A dictionary of RST relations in the AI2D Diagram object.

    Returns: The identifiers of elements parsed from the RST graph.
    """

    # Set up a placeholder list for all elements participating in recursive
    # relations
    elements = []

    # Check relation type: if the relation has nuclei, it is symmetrical
    try:

        # Append nuclei to the list of elements
        elements.extend(relations[identifier]['nuclei'].split())

    # Otherwise, the relation must be asymmetrical and have nuclei and
    # satellites
    except KeyError:

        # Append nuclei and satellites to the list of elements
        elements.extend(relations[identifier]['nucleus'].split())
        elements.extend(relations[identifier]['satellites'].split())

    # Start a second pass over the list of elements for deeper recursion
    while not all(len(e) < 6 for e in elements):

        # Set up a temp list to avoid modifying list looped over
        temp_elements = elements.copy()

        # Loop over the temporary list and enumerate
        for i, e in enumerate(temp_elements):

            # If the identifier is for RST
            if e in rst_dict.keys():

                # Pop item from the temporary list
                temp_elements.pop(i)

                # Extend the list by retrieving the relation referred to, start
                # with symmetrical relations
                try:

                    temp_elements.extend(relations[e]['nuclei'].split())

                # In case a KeyError is raised, the relation is asymmetrical
                except KeyError:

                    temp_elements.extend(relations[e]['nucleus'].split())

                    temp_elements.extend(relations[e]['satellites'].split())

            # If the element is a group identifier, pop the item from list
            if len(e) == 6 and e not in rst_dict.keys():

                temp_elements.pop(i)

        # Assign the updated list to the variable elements
        elements = temp_elements

    # Return the list of elements
    return ' '.join(elements)


# Set up the argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-a", "--annotation", required=True)
ap.add_argument("-o", "--output", required=True)
ap.add_argument("-m", "--mode", required=True)
ap.add_argument("-p", "--p_sample", required=True, type=float)

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
ann_path = args['annotation']
output_path = args['output']
mode = args['mode']
p_sample = args['p_sample']

# Make a copy of the input DataFrame
annotation_df = pd.read_pickle(ann_path).copy()

# If requested, sample annotation for groups
if mode == 'grouping':

    # Create a placeholder dictionary
    groups = {}

    # Loop over the annotation DataFrame
    for ix, row in annotation_df.iterrows():

        # Assign the AI2D-RST diagram object into a variable
        diagram = row['diagram']

        # Assign the grouping graph into a variable
        grouping_graph = diagram.layout_graph

        # Get all nodes
        group_nodes = grouping_graph.nodes(data=True)

        # Create a dictionary of all nodes in the graph
        group_nodes = {node_id: node_data for node_id, node_data in group_nodes}

        # Create separate a dictionary of groups
        group_dict = {k: v for k, v in group_nodes.items()
                      if v['kind'] == 'group'}

        # Set up a placeholder dictionary for valid groups
        valid_groups = {}

        # Add Diagram ID to the dictionary for convenience
        for k, v in group_dict.items():

            # Set up a list for children of the grouping node
            children = []

            # Get a list of edges for each group
            edge_list = grouping_graph.edges(k)

            # Loop over edge list
            for source, target in edge_list:

                # Perform crude filtering by excluding I0 (image constant) and
                # group identifiers (six characters)
                if target == 'I0' or len(target) == 6:

                    continue

                # Append element to list of children
                else:

                    children.append(target)

            # If the grouping node has elements as children, update values
            if len(children) > 1:

                # Update the dictionary with image name information
                v.update({'image_name': row['image_name'],
                          'elements': children})

                # Add the group to valid groups
                valid_groups[k] = v

            # Otherwise continue
            else:

                continue

        # Add relation to the relation dictionary
        groups.update(valid_groups)

    # Create a DataFrame from the grouping dictionary
    group_df = pd.DataFrame.from_dict(groups, orient='index')

    # Print status
    print("[INFO] Collected a total of {} groups for "
          "sampling.".format(len(group_df)))

    # Sample the DataFrame for the fraction defined in p_sample
    sample = group_df.sample(frac=p_sample, random_state=42)

    # Print status
    print("[INFO] Done – sampled {} percentage of {} ({} examples) for {}."
          .format(p_sample, ann_path, len(sample), mode))

    # Save the DataFrame to disk
    sample.to_pickle(output_path)

# If requested, sample the annotation for connectivity
if mode == 'connectivity':

    # Create a placeholder dictionary
    connections = {}

    # Set up a counter for connections
    connection_counter = 0

    # Loop over the annotation DataFrame
    for ix, row in annotation_df.iterrows():

        # Assign the AI2D-RST diagram object into a variable
        diagram = row['diagram']

        # Assign the connectivity and grouping graphs into variables
        connectivity_graph = diagram.connectivity_graph
        grouping_graph = diagram.layout_graph

        # Get nodes for connectivity and grouping and edges for connectivity
        conn_nodes = connectivity_graph.nodes(data=True)
        group_nodes = grouping_graph.nodes(data=True)
        conn_edges = connectivity_graph.edges(data=True)

        # Create a dictionary of all conn_nodes in the graph
        conn_nodes = {node_id: node_data for node_id, node_data in conn_nodes}
        group_nodes = {node_id: node_data for node_id, node_data in group_nodes}

        # Set up a placeholder dictionary for edges
        valid_edges = []

        # Loop over the edge three-tuples (source, target, kind)
        for e in conn_edges:

            # Define variables related to the connection
            source = e[0]
            target = e[1]
            kind = e[2]['kind']
            image_name = row['image_name']

            # Check sources and targets against the grouping graph nodes
            if e[0] not in group_nodes.keys():

                # Print status
                print("[ERROR] Source {} not in grouping graph for {}. "
                      "Skipping ...".format(e[0], row['image_name']))

                continue

            if e[1] not in group_nodes.keys():

                # Print status
                print("[ERROR] Target {} not in grouping graph for {}. "
                      "Skipping ...".format(e[1], row['image_name']))

                continue

            # Unpack the edge three tuple and add image name to a dict
            data_dict = {'source': source,
                         'target': target,
                         'kind': kind,
                         'image_name': image_name}

            # Update dictionary
            connections[connection_counter] = data_dict

            # Update counter
            connection_counter += 1

    # Create a DataFrame from the connection list
    connection_df = pd.DataFrame.from_dict(connections, orient='index')

    # Print status
    print("[INFO] Collected a total of {} connections for sampling.".format(
        len(connection_df)))

    # Sample the DataFrame for the fraction defined in p_sample
    sample = connection_df.sample(frac=p_sample, random_state=42)

    # Print status
    print("[INFO] Done – sampled {} percentage of {} ({} examples) for {}."
          .format(p_sample, ann_path, len(sample), mode))

    # Save the DataFrame to disk
    sample.to_pickle(output_path)

# If requested, sample the annotation for RST relations
if mode == 'rst':

    # Create a placeholder dictionary
    relations = {}

    # Set up a counter for RST relations
    relation_counter = 0

    # Loop over the annotation DataFrame
    for ix, row in annotation_df.iterrows():

        # Assign the AI2D-RST diagram object into a variable
        diagram = row['diagram']

        # Assign the RST and grouping graphs into variables
        rst_graph = diagram.rst_graph
        grouping_graph = diagram.layout_graph

        # Get all nodes in the RST graph and cast into a dictionary
        rst_nodes = dict(rst_graph.nodes(data=True))

        # Filter the nodes for RST relations
        rst_dict = {k: v for k, v in rst_nodes.items()
                    if v['kind'] == 'relation'}

        # Get all nodes in the grouping graph
        group_nodes = dict(grouping_graph.nodes(data=True))

        # Create separate a dictionary of groups
        group_dict = {k: v for k, v in group_nodes.items()
                      if v['kind'] == 'group'}

        # Set up a placeholder dictionary for relations
        for k, v in rst_dict.items():

            # Define variables for the relation
            relation_id = v['id']
            relation_name = v['rel_name']
            image_name = row['image_name']

            # Check nuclei and satellites for recursive relations
            try:

                # Assign nuclei to variable
                nuclei = v['nuclei']

                # Check nuclei for IDs to replace
                for n in nuclei.split():

                    # Check and replace IDs for RST relations
                    if len(n) == 6 and n in rst_dict.keys():

                        # Get replacement IDs
                        replacement = resolve_id(n, rst_dict)

                        # Perform string replacement
                        nuclei = nuclei.replace(n, replacement)

                    # Check and replace IDs for groups
                    if len(n) == 6 and n in group_dict.keys():

                        # Convert undirected graph to directed graph for access
                        # to the successors method, which is needed below.
                        layout_digraph = grouping_graph.copy().to_directed()

                        # Get successors of the group and cast into list. The
                        # node connected to the outbound edge is the last item
                        # in the list. Remove this from the list.
                        successors = list(layout_digraph.successors(n))[:-1]

                        # Get all nodes except the last node in the list and use
                        # these to replace the identifier.
                        replacement = ' '.join(successors)

                        # Perform string replacement
                        nuclei = nuclei.replace(n, replacement)

                # Set nucleus and satellites to None
                nucleus = None
                satellites = None

            # A key error indicates that the relation is asymmetric
            except KeyError:

                # Assign nucleus and satellites to variables
                nucleus = v['nucleus']
                satellites = v['satellites']

                # Cleck nucleus for IDs to replace
                for n in nucleus.split():

                    # Check and replace IDs for RST relations
                    if len(n) == 6 and n in rst_dict.keys():

                        # Get replacement IDs
                        replacement = resolve_id(n, rst_dict)

                        # Perform string replacement
                        nucleus = nucleus.replace(n, replacement)

                    # Check and replace IDs for groups
                    if len(n) == 6 and n in group_dict.keys():

                        # Convert undirected graph to directed graph for access
                        # to the successors method, which is needed below.
                        layout_digraph = grouping_graph.copy().to_directed()

                        # Get successors of the group and cast into list. The
                        # node connected to the outbound edge is the last item
                        # in the list. Remove this from the list.
                        successors = list(layout_digraph.successors(n))[:-1]

                        # Get all nodes except the last node in the list and use
                        # these to replace the identifier.
                        replacement = ' '.join(successors)

                        # Perform string replacement
                        nucleus = nucleus.replace(n, replacement)

                # Check satellites for IDs to replace
                for s in satellites.split():

                    # Get identifiers and limit search to RST relations for now
                    if len(s) == 6 and s in rst_dict.keys():

                        # Get replacement IDs
                        replacement = resolve_id(s, rst_dict)

                        # Perform string replacement
                        satellites = satellites.replace(s, replacement)

                    # Check and replace IDs for groups
                    if len(s) == 6 and s in group_dict.keys():

                        # Convert undirected graph to directed graph for access
                        # to the successors method, which is needed below.
                        layout_digraph = grouping_graph.copy().to_directed()

                        # Get successors of the group and cast into list. The
                        # node connected to the outbound edge is the last item
                        # in the list. Remove this from the list.
                        successors = list(layout_digraph.successors(s))[:-1]

                        # Get all nodes except the last node in the list and use
                        # these to replace the identifier.
                        replacement = ' '.join(successors)

                        # Perform string replacement
                        satellites = satellites.replace(s, replacement)

                # Set nuclei to None
                nuclei = None

            # Replace split nodes with valid identifiers
            if nuclei is not None:

                # Perform replacement using a list comprehension
                nuclei = ' '.join([n.split('.')[0] if '.' in n
                                   else n for n in nuclei.split()])

            # If the relation is asymmetric, perform replacement for n + s
            if nuclei is None:

                # Perform replacement using a list comprehension
                nucleus = ' '.join([n.split('.')[0] if '.' in n
                                    else n for n in nucleus.split()])

                satellites = ' '.join([s.split('.')[0] if '.' in s
                                       else s for s in satellites.split()])

            # Create data dictionary for the relation
            data_dict = {'id': relation_id,
                         'relation_name': relation_name,
                         'nuclei': nuclei,
                         'nucleus': nucleus,
                         'satellites': satellites,
                         'image_name': image_name}

            # Update relation dictionary
            relations[relation_counter] = data_dict

            # Update counter
            relation_counter += 1

    # Create a DataFrame from the dict of relations
    relation_df = pd.DataFrame.from_dict(relations, orient='index')

    # Sample the DataFrame for the fraction defined in p_sample
    sample = relation_df.sample(frac=p_sample, random_state=32)

    # Print status
    print("[INFO] Done – sampled {} percentage of {} ({} examples) for {}."
          .format(p_sample, ann_path, len(sample), mode))

    # Save the DataFrame to disk
    sample.to_pickle(output_path)

# If requested, sample the annotation for macro-groups
if mode == 'macro':

    # Create a placeholder dictionary
    macrogs = {}

    # Set up a counter for macro-groups
    macrog_counter = 0

    # Loop over the DataFrame
    for ix, row in annotation_df.iterrows():

        # Assign the AI2D-RST diagram object into a variable
        diagram = row['diagram']

        # Assign the grouping graph into a variable
        grouping_graph = diagram.layout_graph

        # Get all nodes in the graph and cast into a dictionary
        group_nodes = dict(grouping_graph.nodes(data=True))

        # Create a separate dictionary for macro-groups. Include only groups and
        # image constants.
        mgroup_dict = {k: v for k, v in group_nodes.items() if 'macro_group'
                       in v.keys() and v['kind'] in ['imageConsts', 'group']}

        # Loop over the dictionary of macro-groups
        for k, v in mgroup_dict.items():

            # Define variables for the macro-group
            node_id = k
            node_type = v['kind']
            macro_group = v['macro_group']
            image_name = row['image_name']

            # Create a data dictionary for the macro group
            data_dict = {'id': node_id,
                         'node_type': node_type,
                         'macro_group': macro_group,
                         'image_name': image_name}

            # Update macro-group dictionary
            macrogs[macrog_counter] = data_dict

            # Update counter
            macrog_counter += 1

    # Create a Dataframe from the dict of macro-groups
    macrogroup_df = pd.DataFrame.from_dict(macrogs, orient='index')

    # Sample the DataFrame for the fraction defined in p_sample
    sample = macrogroup_df.sample(frac=p_sample, random_state=42)

    # Print status
    print("[INFO] Done – sampled {} percentage of {} ({} examples) for {}."
          .format(p_sample, ann_path, len(sample), mode))

    # Save the DataFrame to disk
    sample.to_pickle(output_path)
