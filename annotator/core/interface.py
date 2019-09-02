# -*- coding: utf-8 -*-

# Import modules
from .draw import *


def process_command(user_input, mode, diagram, current_graph):
    """
    A function for handling generic commands coming in from multiple annotation
    tasks.

    Parameters:
        user_input: A string containing the command entered by the user.
        mode: A string defining the current annotation task, either 'layout',
              'connectivity' or 'rst'.
        diagram: A Diagram class object that is currently being annotated.
        current_graph: The graph of a Diagram currently being annotated.

    Returns:
        Performs the requested action.
    """

    # Extract command from the user input
    command = user_input.split()[0]

    # Save a screenshot of all annotations if requested
    if command == 'acap':

        # Get filename of current image (without extension)
        fname = os.path.basename(diagram.image_filename).split('.')[0]

        # Join filename to get a string
        fname = ''.join(fname)

        # Draw segmentation
        segmentation = draw_layout(diagram.image_filename,
                                   diagram.annotation,
                                   height=720,
                                   dpi=100)

        # Draw grouping graph
        try:
            grouping = draw_graph(diagram.layout_graph, dpi=100, mode='layout')

        except AttributeError:

            # Print error message
            print("[ERROR] Sorry, you have not annotated the {} graph yet."
                  .format(mode))

            return

        # Draw connectivity graph
        try:
            connectivity = draw_graph(diagram.connectivity_graph, dpi=100,
                                      mode='connectivity')

        except AttributeError:

            # Print error message
            print("[ERROR] Sorry, you have not annotated the {} graph yet."
                  .format(mode))

            return

        # Draw RST graph
        try:
            rst = draw_graph(diagram.rst_graph, dpi=100, mode='rst')

        except AttributeError:

            # Print error message
            print("[ERROR] Sorry, you have not annotated the {} graph yet."
                  .format(mode))

            return

        # Stack images of all graphs side by side and on top of each other
        seg_group = np.hstack([segmentation, grouping])
        rst_group = np.hstack([connectivity, rst])
        all_graphs = np.vstack([seg_group, rst_group])

        # Write image on disk
        cv2.imwrite("all_graphs_{}.png".format(fname), all_graphs)

        # Print status message
        print("[INFO] Saved screenshots for all graphs on disk for {}.png"
              .format(fname))

        return

    # Save a screenshot if requested
    if command == 'cap':

        # Get filename of current image (without extension)
        fname = os.path.basename(diagram.image_filename).split('.')[0]

        # Join filename to get a string
        fname = ''.join(fname)

        # Render high-resolution versions of graph and segmentation
        layout_hires = draw_layout(diagram.image_filename,
                                   diagram.annotation,
                                   height=720,
                                   dpi=200)

        diag_hires = draw_graph(current_graph, dpi=200,
                                mode=mode)

        # Write image on disk
        cv2.imwrite("segmentation_{}.png".format(fname), layout_hires)
        cv2.imwrite("{}_{}.png".format(mode, fname), diag_hires)

        # Print status message
        print("[INFO] Saved separate screenshots on disk for {}.png".format(
            fname
        ))

        return

    # Store a comment if requested
    if command == 'comment':

        # Show a prompt for comment
        comment = input(prompts['comment'])

        # Return the comment
        diagram.comments.append(comment)

        return

    # If requested, mark the annotation as complete and remove isolates from the
    # graph.
    if command == 'done':

        # Check the annotation task and mark complete as appropriate
        if mode == 'layout':

            # Set status to complete
            diagram.group_complete = True

            # Print status message
            print("[INFO] Marking grouping as complete.")

        if mode == 'connectivity':

            # Set status to complete
            diagram.connectivity_complete = True

            print("[INFO] Marking connectivity as complete.")

        if mode == 'rst':

            # Set status to complete
            diagram.rst_complete = True

            print("[INFO] Marking rhetorical structure as complete.")

        # Check if the current graph is frozen
        if nx.is_frozen(current_graph):

            # If the graph is frozen, unfreeze by making a copy
            current_graph = current_graph.copy()

        # Remove grouping edges from RST and connectivity annotation
        if mode == 'rst' or mode == 'connectivity':

            # Retrieve a list of edges in the graph
            edge_bunch = list(current_graph.edges(data=True))

            # Collect grouping edges from the edge list
            try:
                edge_bunch = [(u, v) for (u, v, d) in edge_bunch
                              if d['kind'] == 'grouping']

            except KeyError:
                pass

            # Remove grouping edges from current graph
            current_graph.remove_edges_from(edge_bunch)

        # Find nodes without edges (isolates)
        isolates = list(nx.isolates(current_graph))

        # Remove isolates
        current_graph.remove_nodes_from(isolates)

        # Freeze the graph
        nx.freeze(current_graph)

        # Destroy any remaining windows
        cv2.destroyAllWindows()

        return

    # If requested, exit the annotator immediately
    if command == 'exit':

        # Destroy any remaining windows
        cv2.destroyAllWindows()

        return

    # Export a graphviz DOT graph if requested
    if command == 'export':

        # Get filename of current image (without extension)
        fname = os.path.basename(diagram.image_filename).split('.')[0]

        # Join filename to get a string
        fname = ''.join(fname)

        # Remove grouping edges from RST and connectivity annotation
        if mode == 'rst' or mode == 'connectivity':

            # Retrieve a list of edges in the graph
            edge_bunch = list(current_graph.edges(data=True))

            # Collect grouping edges from the edge list
            try:
                edge_bunch = [(u, v) for (u, v, d) in edge_bunch
                              if d['kind'] == 'grouping']

            except KeyError:
                pass

            # Remove grouping edges from current graph
            current_graph.remove_edges_from(edge_bunch)

        # Find nodes without edges (isolates)
        isolates = list(nx.isolates(current_graph))

        # Remove isolates
        current_graph.remove_nodes_from(isolates)

        # Write DOT graph to disk
        nx.nx_pydot.write_dot(current_graph,
                              '{}_{}.dot'.format(fname, mode))

        # Print status message
        print("[INFO] Saved a DOT graph for {}.png on disk.".format(fname))

        return

    # If requested, release all connections leading to a node
    if command == 'free':

        # Prepare input for validation
        user_input = prepare_input(user_input, 1)

        # Check input against current graph
        valid = validate_input(user_input, current_graph)

        # If the input is not valid, return
        if not valid:

            return

        # If the input is valid, proceed
        if valid:

            # Convert user input to uppercase
            user_input = [i.upper() for i in user_input]

            # Retrieve the list of edges to delete
            edge_bunch = list(current_graph.edges(user_input))

            # Remove designated edges
            current_graph.remove_edges_from(edge_bunch)

            # Flag the graph for re-drawing
            diagram.update = True

    # If requested, print info on current annotation task
    if command == 'info':

        # Clear screen first
        os.system('cls' if os.name == 'nt' else 'clear')

        # Print information on layout commands
        print(info[mode])
        print(info['generic'])

    # If requested, remove isolates from the current graph
    if command == 'isolate':

        # Find nodes without edges (isolates)
        isolates = list(nx.isolates(current_graph))

        # Remove isolates
        current_graph.remove_nodes_from(isolates)

        # Print status message
        print("[INFO] Removing isolates from the graph as requested.")

        # Flag the graph for re-drawing
        diagram.update = True

        return

    # If requested, print macro-groups
    if command == 'macrogroups':

        # Print header for available macro-groups
        print("---\nAvailable macro-groups and their aliases\n---")

        # Print the available macro-groups and their aliases
        for k, v in macro_groups.items():
            print("{} (alias: {})".format(v, k))

        # Print closing line
        print("---")

        # Get current macro-groups from the layout graph
        mgroups = dict(nx.get_node_attributes(diagram.layout_graph,
                                              'macro_group'))

        # If more than one macro-group has been defined, print groups
        if len(mgroups) > 0:

            # Print header for current macro-groups
            print("\nCurrent macro-groups \n---")

            # Print the currently defined macro-groups
            for k, v in mgroups.items():
                print("{}: {}".format(k, v))

            # Print closing line
            print("---\n")

        return

    # If requested, move to the next graph
    if command == 'next':

        # Destroy any remaining windows
        cv2.destroyAllWindows()

        return

    # If requested, removing grouping nodes
    if command == 'ungroup':

        # Retrieve a list of edges in the graph
        edge_bunch = list(current_graph.edges(data=True))

        # Collect grouping edges from the edge list
        edge_bunch = [(u, v) for (u, v, d) in edge_bunch
                      if d['kind'] == 'grouping']

        # Remove grouping edges from current graph
        current_graph.remove_edges_from(edge_bunch)

        # Flag the graph for re-drawing
        diagram.update = True

        return

    # If requested, print available RST relations
    if command == 'rels':

        # Clear screen first
        os.system('cls' if os.name == 'nt' else 'clear')

        # Print header for available macro-groups
        print("---\nAvailable RST relations and their aliases\n---")

        # Loop over RST relations
        for k, v in rst_relations.items():
            # Print information on each RST relation
            print("{} (alias: {}, type: {})".format(
                v['name'],
                k,
                v['kind']))

        # Print closing line
        print("---")

        # Generate a dictionary of RST relations present in the graph
        relation_ix = get_node_dict(current_graph, kind='relation')

        # Loop through current RST relations and rename for convenience.
        relation_ix = {"R{}".format(i): k for i, (k, v) in
                       enumerate(relation_ix.items(), start=1)}

        # If more than one macro-group has been defined, print groups
        if len(relation_ix) > 0:

            # Print header for current macro-groups
            print("\nCurrent RST relations \n---")

            # Print relations currently defined in the graph
            for k, v in relation_ix.items():

                print("{}: {}".format(k,
                                      diagram.rst_graph.nodes[v]['rel_name']))

            # Print closing line
            print("---\n")

        return

    # If requested, reset the annotation
    if command == 'reset':

        # Reset layout graph if requested
        if mode == 'layout':

            # Unfreeze the reset graph and assign to layout_graph
            diagram.layout_graph = create_graph(diagram.annotation,
                                                edges=False,
                                                arrowheads=False,
                                                mode='layout'
                                                )

        # Reset connectivity graph if requested
        if mode == 'connectivity':

            # Create a new connectivity graph for the Diagram object
            diagram.connectivity_graph = nx.MultiDiGraph()

            # Update grouping information from the grouping graph to the new
            # connectivity graph
            update_grouping(diagram, diagram.connectivity_graph)

        # Reset RST graph if requested
        if mode == 'rst':

            # Create a new RST graph for the Diagram object
            diagram.rst_graph = nx.DiGraph()

            # Update grouping information from the grouping graph to the new RST
            # graph
            update_grouping(diagram, diagram.rst_graph)

        # Flag the graph for re-drawing
        diagram.update = True

        return

    # If requested, delete grouping nodes
    if command == 'rm':

        # Prepare input for validation
        user_input = prepare_input(user_input, 1)

        # Check if RST relations need to be included in validation
        if mode == 'rst':

            # Validate input against relations as well
            valid = validate_input(user_input, current_graph,
                                   groups=True, rst=True)

        else:
            # Check input against the current graph
            valid = validate_input(user_input, current_graph, groups=True)

        # If the input is not valid, continue
        if not valid:

            return

        # If input is valid, proceed
        if valid:

            # Generate a dictionary mapping group aliases to IDs
            group_dict = replace_aliases(current_graph, 'group')

            # Replace aliases with valid identifiers, if used
            user_input = [group_dict[u] if u in group_dict.keys()
                          else u for u in user_input]

            # If annotating RST relations, check RST relations as well
            if mode == 'rst':

                # Generate a dictionary mapping relation aliases to IDs
                rel_dict = replace_aliases(current_graph, 'relation')

                # Replace aliases with valid identifiers, if used
                user_input = [rel_dict[u] if u in rel_dict.keys()
                              else u.upper() for u in user_input]

            # Remove the designated nodes from the graph
            current_graph.remove_nodes_from(user_input)

            # Flag the graph for re-drawing
            diagram.update = True

            return

    # if requested, split a node
    if command == 'split':

        # Begin by checking the number of desired splits
        n_splits = int(user_input.split()[1])

        # Prepare input for validation
        user_input = prepare_input(user_input, 2)

        # Validate input – only diagram elements can be split
        valid = validate_input(user_input, current_graph)

        # If the input is valid, proceed
        if valid:

            # Set up a placeholder list for split nodes
            split_list = []

            # Get properties of the node to duplicate
            for n in user_input:

                # Generate new identifiers for split nodes by taking the node
                # name in uppercase and adding the number of split after stop.
                split_ids = [n.upper() + '.{}'.format(i)
                             for i in range(1, n_splits + 1)]

                # Get the attribute of the node that is being split
                attr_dict = current_graph.nodes[n.upper()]

                # Add parent node information to the dictionary
                attr_dict['copy_of'] = n.upper()

                # Loop over split ids
                for s in split_ids:

                    # Append a tuple of identifier and attributes to split_list
                    split_list.append((s, attr_dict))

                # Remove node from the RST graph
                current_graph.remove_node(n.upper())

            # Add split nodes to the graph
            current_graph.add_nodes_from(split_list)

            # Flag the graph for re-drawing
            diagram.update = True

            return


# Define a dictionary of available commands during annotation
commands = {'rst': ['rels', 'split', 'ungroup'],
            'connectivity': ['ungroup'],
            'generic': ['acap', 'cap', 'comment', 'done', 'exit', 'export',
                        'free', 'info', 'isolate', 'macrogroups', 'next',
                        'reset', 'rm'],
            'tasks': ['conn', 'group', 'rst']
            }

info = {'layout': "---\n"
                  "Enter the identifiers of diagram elements you wish to\n"
                  "group together. Separate the identifiers with a comma.\n"
                  "\n"
                  "Example of valid input: b1, a1, t1\n\n"
                  ""
                  "This command groups nodes B1, A1 and T1 together under a\n"
                  "grouping node.\n"
                  "---\n"
                  "Grouping nodes may be deleted using command rm.\n\n"
                  "Example command: rm g1\n\n"
                  "This command deletes group G1. Multiple groups can be\n"
                  "deleted by entering multiple identifiers, e.g. rm g1 g2 g3\n"
                  "---\n"
                  "To add macro-grouping information to a node, group, image\n"
                  "constant or their groups, enter the command 'macro' and \n"
                  "by the identifier or identifiers.\n\n"
                  "Example command: macro i0\n\n"
                  "A list of available macro-groups can be printed using the\n"
                  "command 'macrogroups'. This command will also print all\n"
                  "currently defined macro-groups.\n"
                  "---\n",
        'rst': "---\n"
               "Enter the command 'new' to create a new RST relation.\n"
               "The tool will then ask you to enter a valid name for the\n"
               "relation. Relations can be deleted using the command 'rm'.\n"
               "Names are entered by using abbreviations, which can be listed\n"
               "using the command 'relations'.\n\n"
               "The tool will infer the type of relation and ask you to enter\n"
               "either a nucleus and satellites or several nuclei.\n"
               "---\n"
               "If diagram elements are picked out by multiple rhetorical\n"
               "relations, you can use the command 'split' to split the node.\n"
               "This creates multiple instances of the same node, which can \n"
               "be picked out as parts of different rhetorical relations.\n\n"
               "Example command: split 2 b1\n\n"
               "This command splits node B1 into two nodes, which are given\n"
               "identifiers B1.1 and B1.2.\n"
               "---\n",
        'connectivity': "---\n"
                        "Drawing a connection between nodes requires three\n"
                        "types of information: source, connection type and\n"
                        "target.\n\n"
                        "The sources and targets must be valid identifiers,\n"
                        "elements and groups or lists of valid identifiers\n"
                        "separated using commas.\n\n"
                        "Example command: t1 > b0, b1\n\n"
                        "The connection type must be one of the following\n"
                        "shorthand aliases:\n\n"
                        "- for undirected lines\n"
                        "> for unidirectional arrow\n"
                        "<> for bidirectional arrow\n"
                        "---\n",
        'generic': "Other valid commands include:\n\n"
                   "acap: Save a screen capture for all graphs in diagram.\n"
                   "cap: Save a screen capture of the current visualisation.\n"
                   "comment: Enter a comment about current diagram.\n"
                   "free: Remove all edges leading to a node, e.g. free b0.\n"
                   "exit: Exit the annotator immediately.\n"
                   "export: Export the current graph into DOT format. \n"
                   "done: Mark the current diagram as complete and move on.\n"
                   "hide: Hide the layout segmentation.\n"
                   "info: Print this message.\n"
                   "isolate: Remove isolates from the graph.\n"
                   "next: Save current work and move on to the next diagram.\n"
                   "reset: Reset the current annotation.\n"
                   "show: Show the layout segmentation. Use e.g. show b0 to\n"
                   "      a single unit.\n"
                   "---",
        }

# Define a dictionary of various prompts presented to user during annotation
prompts = {'nucleus_id': "[RST] Enter the identifier of nucleus: ",
           'satellite_id': "[RST] Enter the identifier(s) of satellite(s): ",
           'layout_default': "[GROUPING] Please enter nodes to group or a valid"
                             " command: ",
           'comment': "Enter comment: ",
           'rst_default': "[RST] Please enter a valid command: ",
           'rel_prompt': "[RST] Please enter relation name: ",
           'nuclei_id': "[RST] Enter the identifiers of the nuclei: ",
           'macro_group': "[GROUPING] Please enter macro-group type: ",
           'conn_default': "[CONNECTIVITY] Please enter a connection or a valid"
                           " command: ",
           'table_rows': "[GROUPING] How many rows does the table have? ",
           'table_cols': "[GROUPING] How many columns does the table have? ",
           'table_axes': "[GROUPING] How many axes have labels? "
           }

# Define a dictionary of various error messages that may arise during annotation
messages = {'nucleus_err': "Sorry, a mononuclear relation cannot have more "
                           "than one nucleus. Please try again.",
            'nuclei_err': "Sorry, a multinuclear relation must have more than "
                          "one nucleus. Please try again.",
            'layout_complete': "[ERROR] Grouping annotation is marked as "
                               "complete.",
            'conn_complete': "[ERROR] Connectivity annotation is marked as "
                             "complete.",
            'rst_complete': "[ERROR] RST annotation is marked as complete. "
            }


# Define a dictionary of RST relations / types and their aliases (keys)
rst_relations = {'anti': {'name': 'antithesis', 'kind': 'mono'},
                 'back': {'name': 'background', 'kind': 'mono'},
                 'circ': {'name': 'circumstance', 'kind': 'mono'},
                 'conc': {'name': 'concession', 'kind': 'mono'},
                 'cond': {'name': 'condition', 'kind': 'mono'},
                 'elab': {'name': 'elaboration', 'kind': 'mono'},
                 'enab': {'name': 'enablement', 'kind': 'mono'},
                 'eval': {'name': 'evaluation', 'kind': 'mono'},
                 'evid': {'name': 'evidence', 'kind': 'mono'},
                 'pret': {'name': 'interpretation', 'kind': 'mono'},
                 'just': {'name': 'justify', 'kind': 'mono'},
                 'mean': {'name': 'means', 'kind': 'mono'},
                 'moti': {'name': 'motivation', 'kind': 'mono'},
                 'nvoc': {'name': 'nonvolitional-cause', 'kind': 'mono'},
                 'nvor': {'name': 'nonvolitional-result', 'kind': 'mono'},
                 'otws': {'name': 'otherwise', 'kind': 'mono'},
                 'prep': {'name': 'preparation', 'kind': 'mono'},
                 'purp': {'name': 'purpose', 'kind': 'mono'},
                 'rest': {'name': 'restatement', 'kind': 'multi'},
                 'solu': {'name': 'solutionhood', 'kind': 'mono'},
                 'summ': {'name': 'summary', 'kind': 'mono'},
                 'unls': {'name': 'unless', 'kind': 'mono'},
                 'volc': {'name': 'volitional-cause', 'kind': 'mono'},
                 'volr': {'name': 'volitional-result', 'kind': 'mono'},
                 'cont': {'name': 'contrast', 'kind': 'multi'},
                 'join': {'name': 'joint', 'kind': 'multi'},
                 'list': {'name': 'list', 'kind': 'multi'},
                 'sequ': {'name': 'sequence', 'kind': 'multi'},
                 'cseq': {'name': 'cyclic sequence', 'kind': 'multi'},  # NEW
                 'iden': {'name': 'identification', 'kind': 'mono'},
                 'casc': {'name': 'class-ascription', 'kind': 'mono'},
                 'pasc': {'name': 'property-ascription', 'kind': 'mono'},
                 'poss': {'name': 'possession', 'kind': 'mono'},
                 'proj': {'name': 'projection', 'kind': 'mono'},
                 'conn': {'name': 'connected', 'kind': 'multi'},  # NEW!
                 'titl': {'name': 'title', 'kind': 'mono'},
                 'conj': {'name': 'conjunction', 'kind': 'multi'},  # NEW!
                 'disj': {'name': 'disjunction', 'kind': 'multi'}  # NEW!
                 }

# Define a dictionary of valid macro-groups and their aliases
macro_groups = {'table': 'table',
                'hor': 'horizontal',
                'ver': 'vertical',
                'net': 'network',
                'cycle': 'cycle',
                'slice': 'slice',
                'cut': 'cut-out',
                'exp': 'exploded',
                'photo': 'photograph',
                'ill': 'illustration',
                'diag': 'diagrammatic'
                }
