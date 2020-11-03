# -*- coding: utf-8 -*-

# Import packages
from collections import namedtuple
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils import data
import cv2
import dgl
import json
import networkx as nx
import numpy as np
import torch

# Make the AI2D_RST class and the AI2D_RSTBatch named tuple available through
# import using __all__, which provides a list of public objects for a module
# when importing using import *
__all__ = ['AI2D_RST', 'AI2D_RSTBatch', 'create_batch']

# Define named tuple for batched graphs
AI2D_RSTBatch = namedtuple('AI2D_RSTBatch', ['graphs', 'labels'])

# Define node (diagram element) types and their numerical labels. Note
# that these depend on the selected annotation layers. The first one is
# a dictionary of basic nodes for grouping and connectivity layers and
# their combinations.
node_dicts = {'grouping': {'imageConsts': np.array([0], dtype=np.int32),
                           'arrows': np.array([1], dtype=np.int32),
                           'group': np.array([2], dtype=np.int32),
                           'text': np.array([3], dtype=np.int32),
                           'blobs': np.array([4], dtype=np.int32)
                           },
              'discourse': {'arrows': np.array([0], dtype=np.int32),
                            'group': np.array([1], dtype=np.int32),
                            'text': np.array([2], dtype=np.int32),
                            'blobs': np.array([3], dtype=np.int32),
                            'relation': np.array([4], dtype=np.int32)},
              'relations': {'identification': np.array([0], dtype=np.int32),
                            'cyclic sequence': np.array([1], dtype=np.int32),
                            'preparation': np.array([2], dtype=np.int32),
                            'joint': np.array([3], dtype=np.int32),
                            'elaboration': np.array([4], dtype=np.int32),
                            'property-ascription': np.array([5],
                                                            dtype=np.int32),
                            'list': np.array([6], dtype=np.int32),
                            'contrast': np.array([7], dtype=np.int32),
                            'circumstance': np.array([8], dtype=np.int32),
                            'restatement': np.array([9], dtype=np.int32),
                            'connected': np.array([10], dtype=np.int32),
                            'conjunction': np.array([11], dtype=np.int32),
                            'sequence': np.array([12], dtype=np.int32),
                            'nonvolitional-cause': np.array([13], dtype=np.int32),
                            'nonvolitional-result': np.array([14], dtype=np.int32),
                            'condition': np.array([15], dtype=np.int32),
                            'means': np.array([16], dtype=np.int32),
                            'background': np.array([17], dtype=np.int32),
                            'class-ascription': np.array([18], dtype=np.int32),
                            'disjunction': np.array([19], dtype=np.int32),
                            'justify': np.array([20], dtype=np.int32),
                            'volitional-result': np.array([21], dtype=np.int32),
                            'solutionhood': np.array([22], dtype=np.int32),
                            'enablement': np.array([23], dtype=np.int32),
                            'unless': np.array([24], dtype=np.int32)}
              }

edge_dicts = {'grouping': {'grouping': np.array([0], dtype=np.int32)},
              'connectivity':  {'grouping': np.array([0], dtype=np.int32),
                                'directional': np.array([1], dtype=np.int32),
                                'undirectional': np.array([2], dtype=np.int32),
                                'bidirectional': np.array([3], dtype=np.int32)
                                },
              'discourse': {'grouping': np.array([0], dtype=np.int32),
                            'nucleus': np.array([1], dtype=np.int32),
                            'satellite': np.array([2], dtype=np.int32)},
              'discourse+connectivity': {'grouping': np.array([0], dtype=np.int32),
                                         'nucleus': np.array([1], dtype=np.int32),
                                         'satellite': np.array([2], dtype=np.int32),
                                         'directional': np.array([3], dtype=np.int32),
                                         'undirectional': np.array([4], dtype=np.int32),
                                         'bidirectional': np.array([5], dtype=np.int32)}
              }


# Define a function for batching data and sending it to a device (CPU or GPU)
def create_batch(device):

    def batch_to_device(batch):
        # Get graphs and labels
        graph_batch = [x[0] for x in batch]
        label_batch = [x[1] for x in batch]

        # Batch diagrams
        diagram_batch = dgl.batch(graph_batch)

        # Convert the list of labels into a numpy array
        label_batch = np.concatenate(label_batch)

        # Cast label batch into a torch tensor
        label_batch = torch.from_numpy(label_batch)

        # Return a batch of graphs and diagrams wrapped into a namedtuple
        return AI2D_RSTBatch(graphs=diagram_batch,
                             labels=label_batch)

    return batch_to_device


# This defines a class for the AI2D-RST dataset based on
# torch.utils.data.Dataset class
class AI2D_RST(data.Dataset):

    # Initialize class
    def __init__(self, cat_path, img_path, orig_json_path, rst_json_path,
                 layers, **kwargs):
        """
        Initializes the AI2D-RST dataset.

        Parameters:
            cat_path: Path to the JSON file containing diagram labels.
            img_path: Path to the directory containing AI2D images.
            orig_json_path: Path to the directory containing AI2D JSON files.
            rst_json_path: Path to the directory containing AI2D-RST JSON files.
            layers: A string indicating which annotation layers to include in
                    the graph. Valid values include: 'grouping',
                    'grouping+connectivity', 'discourse' and
                    'discourse+connectivity'.
            kwargs: Optional keywords and arguments:
                    nx (Boolean): Return NetworkX graphs instead of DGL graphs.
                    node_types (Boolean): Add type information to node features.
                    smooth (Boolean): Use smoothed labels for node type info.

        Returns:
            The AI2D-RST dataset as a torch.utils.data.Dataset object.
        """

        # Initialize class attributes
        self.name = "AI2D-RST"                      # Dataset name
        self.cat_path = Path(cat_path)              # Path to categories JSON
        self.img_path = Path(img_path)              # Path to AI2D images
        self.orig_json_path = Path(orig_json_path)  # Path to AI2D JSON
        self.rst_json_path = Path(rst_json_path)    # Path to AI2D-RST JSON

        # Check input types
        assert self.cat_path.is_file()
        assert self.img_path.is_dir()
        assert self.orig_json_path.is_dir()
        assert self.rst_json_path.is_dir()
        assert layers in ['grouping', 'grouping+connectivity', 'connectivity',
                          'discourse', 'discourse+connectivity']

        # Load node and edge dictionaries
        self.node_dict = node_dicts
        self.edge_dict = edge_dicts

        # Load diagram labels from the labels JSON file
        categories = self._load_annotation(cat_path)

        # Initialize label encoder and encode integer labels
        le = LabelEncoder().fit(list(categories.values()))

        # Create a dictionary mapping encoded class integers to their names
        self.class_names = {k: v for k, v in zip(le.transform(le.classes_),
                                                 le.classes_)}

        # Create a dictionary mapping filenames to labels
        label_dict = {k: le.transform([v]) for k, v in categories.items()}

        # Convert labels into a numpy array for calculating class weights
        label_arr = np.concatenate(list(label_dict.values()))

        # Calculate class weights
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(label_arr),
                                             y=label_arr)

        # Wrap class weights into a torch Tensor and make available through
        # attribute
        self.class_weights = torch.FloatTensor(class_weights)

        # Get diagram identifiers and labels
        self.file_ids = list(label_dict.keys())
        self.labels = list(label_dict.values())

        # Return DGL graph objects by default
        self._return_nx = False

        # Check if NetworkX graphs have been requested
        if kwargs and kwargs['nx']:

            # Set the flag for returning NetworkX graphs to True
            self._return_nx = True

        # Check if node type information should be added to node features
        if kwargs and kwargs['node_types']:

            # Set add node types flag to True
            self._add_node_types = True

            # Check which node label dictionary to use: this depends on the kind
            # of annotation layers requested
            if 'discourse' in layers:

                # Get the node labels from the node dictionary & cast to array
                node_labels = list(self.node_dict['discourse'].values())
                node_labels = np.asarray(node_labels)

            else:

                # Get the node labels from the node dictionary & cast to array
                node_labels = list(self.node_dict['grouping'].values())
                node_labels = np.asarray(node_labels)

            # Initialize label binarizer and fit to node labels
            self._node_binarizer = LabelBinarizer().fit(node_labels)

        else:

            self._add_node_types = False

        # Check if smoothed labels have been requested
        if 'smooth' in kwargs and kwargs['smooth']:

            # Set the flag for smoothed labels to True
            self._smooth_labels = True

        else:

            self._smooth_labels = False

        # Initialize label binarizer for RST relations if needed
        if 'discourse' in layers:

            # Get the RST relations from the node dictionary
            rst_relations = np.asarray(list(self.node_dict['relations'].values()))

            # Initialize label binarizer and fit to node labels
            self._rst_binarizer = LabelBinarizer().fit(rst_relations)

        # Load the requested annotation and create the graphs accordingly
        self._load(layers)

        # Get number of unique diagram classes in the dataset
        self.n_classes = len(np.unique(self.labels))

        # Get the number of node and edge classes for DGL graphs (grouping +
        # connectivity)
        if 'discourse' not in layers and not self._return_nx:

            # Get unique node and edge types for graphs that don't use typed
            # nodes or edges
            node_list = [x.ndata['kind'].flatten() for x in self.diagrams]
            self.n_node_classes = len(np.unique(torch.cat(node_list).numpy()))

            edge_list = [x.edata['kind'].flatten() for x in self.diagrams]
            self.n_edge_classes = len(np.unique(torch.cat(edge_list).numpy()))

        # Do the same for DGLHeteroGraphs (discourse)
        if 'discourse' in layers and not self._return_nx:

            node_list = np.concatenate(np.asarray([x.ntypes for x in
                                                   self.diagrams]))
            self.n_node_classes = len(np.unique(node_list))

            edge_list = np.concatenate(np.asarray([x.etypes for x in
                                                   self.diagrams]))
            self.n_edge_classes = len(np.unique(edge_list))

    @staticmethod
    def _load_annotation(json_path):
        """
        Loads annotation from a JSON file and returns a dictionary.

        Parameters:
             json_path: A string containing the path to the annotation file.

        Returns:
             A dictionary containing annotation.
        """
        # Open the file containing the annotation
        with open(json_path) as annotation_file:

            # Parse the AI2D annotation from the JSON file into a dictionary
            annotation = json.load(annotation_file)

        # Return the annotation
        return annotation

    @staticmethod
    def _encode_edges(graph, e_dict):
        """
        Converts edge types (strings) into numerical labels.

        Parameters:
            graph: A NetworkX graph.
            e_dict: A dictionary mapping strings to integer labels.

        Returns:
            Updates the edge attributes for graph.
        """

        # Set up a placeholder dictionary for updated edge features
        upd_edge_feats = {}

        # Get edges in the graph
        edges = graph.edges(data=True)

        # Loop over the edges
        for src, dst, features in edges:

            # Skip edges that already have numerical labels
            if type(features['kind']) == np.ndarray:

                # Use original features and continue
                upd_edge_feats[src, dst] = features

                continue

            # Encode edge type information using numerical labels and
            # store into dictionary.
            upd_edge_feats[src, dst] = {'kind': e_dict[features['kind']]}

        # Set updated edge attributes
        nx.set_edge_attributes(graph, upd_edge_feats)

    @staticmethod
    def _resolve_grouping_node(group_node, group_tree, group_graph,
                               target_graph):
        """
        Resolves the predecessors of a grouping node and adds them to the
        target graph. This function can be used to enrich connectivity
        and discourse graphs with information from the grouping graph
        by fetching the nodes that participate in a visual group.

        Parameters:
            group_node: A string with the identifier of the grouping node.
            group_tree: A depth-first search tree for the grouping graph.
            group_graph: An AI2D-RST grouping graph.
            target_graph: A NetworkX graph which contains the grouping node
                          to resolve.

        Returns:
             An updated target graph with diagram element nodes added
             under the grouping node.
        """

        # Get the predecessors of the grouping node
        preds = nx.dfs_predecessors(group_tree, group_node)

        # Get a list of unique node identifiers among predecessors. These are
        # the nodes on which a subgraph will be induced.
        preds = list(set(list(preds.keys()) + list(preds.values())))

        # Induce a subgraph based on the nodes
        pred_group = group_graph.subgraph(preds).copy()

        # Set up edge dictionary
        edge_attrs = {}

        # Encode edge type information
        for s, t in pred_group.edges():

            # Add edge attributes to the dictionary
            edge_attrs[(s, t)] = {'kind': 'grouping'}

        # Set edge attributes
        nx.set_edge_attributes(pred_group, edge_attrs)

        # Add the nodes and edges from the subgraph to the connectivity graph
        target_graph.add_nodes_from(pred_group.nodes(data=True))
        target_graph.add_edges_from(pred_group.edges(data=True))

    # Load annotation from JSON files
    def _load(self, layers):
        """
        Loads AI2D-RST annotation from JSON files.

        Parameters:
            layers: A string defining which annotation layers are included in
                    the graph based representation.

        Returns:
            Appends the diagram to the list of diagrams.
        """

        # Initialise a list to hold the diagrams
        self.diagrams = []

        # Loop over the diagrams
        for diagram_id in self.file_ids:

            # Print status
            print(f"[INFO] Now building graph for diagram {diagram_id} ...")

            # Build graphs requested for each diagram
            try:
                diagram = self._build_graphs(diagram_id, layers)

                # If requested, add node type information to the node features.
                # This option is only available for grouping and connectivity
                # DGL graphs or their combination, which do not use typed nodes.
                if self._add_node_types and 'discourse' not in layers:

                    # Check if NetworkX graphs have been requested
                    if self._return_nx:

                        raise NotImplementedError("This option is not available"
                                                  " for NetworkX graphs. Set"
                                                  " 'nx' to False.")

                    # Get the node labels and encode using the label binarizer
                    node_labels = self._node_binarizer.transform(
                        diagram.ndata['kind'].numpy())

                    # Check if the node labels should be smoothed
                    if self._smooth_labels:

                        # Cast into float for label smoothing
                        node_labels = np.asarray(node_labels, dtype=np.float64)

                        # Smooth the labels by a factor of 0.1
                        node_labels *= (1 - 0.1)
                        node_labels += (0.1 / node_labels.shape[1])

                    # Cast the smoothed labels into a Torch tensor
                    th_labels = torch.from_numpy(node_labels)

                    # Concate the layout feature and node identity vectors
                    updated_features = torch.cat((diagram.ndata['features'],
                                                  th_labels), dim=1)

                    # Update diagram features
                    diagram.ndata['features'] = updated_features

            # This catches errors arising from AI2D diagrams *not* included in
            # AI2D-RST, allowing using the original AI2D categories.json with
            # AI2D-RST.
            except FileNotFoundError:

                continue

            # Check if a diagram exists - not all diagrams have connectivity
            # graphs, if only connectivity graphs have been requested.
            if diagram is not None:

                # Append diagram to list of diagrams
                self.diagrams.append(diagram)

    # A function for parsing annotation from AI2D-RST JSON files
    @staticmethod
    def _parse_ai2d_rst_json(data):
        """"
        Creates NetworkX graphs from dictionaries loaded from AI2D-RST JSON.

        Parameters:
            data: A dictionary loaded from AI2D-RST JSON.

        Returns:
            Grouping, connectivity and discourse graphs as NetworkX graphs
        """
        # Separate dictionaries for each layer from the JSON dictionary
        grouping_dict_from_json = data['grouping']
        conn_dict_from_json = data['connectivity']
        rst_dict_from_json = data['rst']

        # Create the grouping graph using the nx.jit_graph function
        grouping_graph = nx.jit_graph(grouping_dict_from_json,
                                      create_using=nx.DiGraph())

        # Check if connectivity annotation exists
        if conn_dict_from_json is not None:

            # Create connectivity graph manually
            connectivity_graph = nx.DiGraph()

            # Load nodes and edges
            nodes = conn_dict_from_json['nodes']
            edges = conn_dict_from_json['edges']

            # Add nodes manually to the connectivity graph
            for node in nodes:

                connectivity_graph.add_node(node[0], kind=node[1]['kind'])

            # Add edges manually to the connectivity graph
            for e in edges:

                connectivity_graph.add_edge(e[0], e[1], kind=e[2]['kind'])

        else:

            connectivity_graph = None

        # Create the RST graph using the nx.jit_graph function
        rst_graph = nx.jit_graph(rst_dict_from_json,
                                 create_using=nx.DiGraph())

        # Return all three graphs
        return grouping_graph, connectivity_graph, rst_graph

    # A function for parsing AI2D layout segmentation annotation
    def _parse_ai2d_layout(self, ai2d_ann, h, w, n_pix, node_type, node_id):
        """
        Parse the original AI2D layout segmentation annotation.

        Parameters:
            ai2d_ann: A dictionary of parsed AI2D annotation.
            h: Height of the diagram image in pixels.
            w: Width of the diagram image in pixels.
            n_pix: Number of pixels in the entire diagram image.
            node_type: Type of the diagram element.
            node_id: Unique identifier of the diagram element.

        Returns:
             A 4-dimensional array describing the position, size and solidity of
             the diagram element as NumPy array of dtype float32.
        """
        # Process elements with bounding boxes first
        if node_type in ['text', 'arrowHeads']:

            # Assign coordinates to a variable
            coords = np.array(ai2d_ann[node_type][node_id]['rectangle'],
                              np.int32)

            # Calculate element area via the cv2.boundingRect, as
            # the cv2.contourArea does not work with xy-coordinates
            (rect_x, rect_y, rect_w, rect_h) = cv2.boundingRect(coords)

            # Calculate element area
            area = (rect_w * rect_h) / float(n_pix)

            # Calculate solidity, or simply assign 1, as rectangles
            # are always solid
            solidity = 1.0

            # Get centre point
            centre = (int(np.round(rect_x + (rect_w / 2), decimals=0)),
                      int(np.round(rect_y + (rect_h / 2), decimals=0)))

        # Continue to process elements with polygons
        if node_type in ['arrows', 'blobs']:

            # Fetch coordinates and convert to numpy array
            coords = np.array(ai2d_ann[node_type][node_id]
                              ['polygon'], np.int32)

            # Calculate element area
            area = cv2.contourArea(coords)

            # Calculate convex hull and its area
            hull = cv2.convexHull(coords)
            hull_area = cv2.contourArea(hull)

            # Calculate solidity (area divided by convex hull area)
            try:

                solidity = area / hull_area

            except ZeroDivisionError:

                solidity = 0

            # Normalize element area by dividing by total pixel
            # count
            area = area / float(n_pix)

            # Calculate moments for finding the centroid
            try:
                moments = cv2.moments(coords)

                # Calculate centroid
                centroid_x = int(moments["m10"] / moments["m00"])
                centroid_y = int(moments["m01"] / moments["m00"])

                # Assign centre point to a variable
                centre = [centroid_x, centroid_y]

            except ZeroDivisionError:

                centre = [0, 0]

        # If the node is a group or an image constant, set the features to zero
        if node_type in ['group', 'imageConsts']:

            # Assign empty coordinates to a variable
            norm_coords = [0, 0]
            area = 0
            solidity = 0

        # Otherwise normalize coordinates to [0..1] by dividing by image width
        # and height. This must be done last to prevent error from imageConsts
        # and groups, which is caused by an absent centre point.
        else:

            norm_coords = np.array([centre[0] / w, centre[1] / h], np.float32)

        # Concatenate all layout features along axis 0
        layout_feats = np.concatenate((norm_coords, [area], [solidity]), axis=0)

        # Return layout feature vector
        return layout_feats

    # A function for extracting features from AI2D annotation and adding them to
    # the AI2D-RST graphs
    def _extract_features(self, graphs, ai2d_ann, image, layers):
        """
        Extracts features from the original AI2D annotation and adds them to the
        AI2D-RST graphs.

        Parameters:
            graphs: A dictionary of NetworkX graphs for AI2D-RST annotation.
            ai2d_ann: A dictionary containing the original AI2D annotation.
            image: An image of the diagram from the original AI2D dataset.
            layers: A string defining annotation layers to include in the
                    updated graphs.

        Returns:
            A dictionary of NetworkX graphs with updated features.
        """
        # To begin with, build the grouping graph, which is provides the layout
        # information on all diagram elements, which can be then picked out in
        # other graphs, if necessary.
        graph = graphs['grouping']

        # Check that a graph exists
        try:

            # Fetch nodes from the graph
            nodes = graph.nodes(data=True)

        except AttributeError:

            return None

        # Begin extracting the features by getting the diagram image shape
        h, w = image.shape[:2]

        # Get the number of pixels in the image
        n_pix = h * w

        # Set up a placeholder dictionaries to hold updated node and edge
        # features
        node_features = {}
        edge_features = {}

        # Loop over the nodes and their features
        for node, features in nodes:

            # Fetch the node type from its features under the key 'kind'
            node_type = features['kind']

            # Parse layout annotation
            layout_feats = self._parse_ai2d_layout(ai2d_ann,    # annotation
                                                   h,           # image height
                                                   w,           # image width
                                                   n_pix,       # n of pixels
                                                   node_type,   # elem type
                                                   node         # node id
                                                   )

            # Add layout features to the dictionary of updated node features
            node_features[node] = {'features': layout_feats,
                                   'kind': self.node_dict['grouping'][node_type]}

            # Updated node attributes in the grouping graph using layout
            # features
            nx.set_node_attributes(graph, node_features)

        # Calculate features for grouping nodes based on their children. This
        # requires a directed tree graph.
        group_tree = nx.dfs_tree(graph, source="I0")

        # Get a list of grouping nodes and image constants in the graph
        groups = [n for n, attr in graph.nodes(data=True) if attr['kind']
                  in [self.node_dict['grouping']['imageConsts'],
                      self.node_dict['grouping']['group']]]

        # Iterate over the nodes in the graph
        for n, attr in graph.nodes(data=True):

            # Check if the node type is a group
            if n in groups:

                # Get predecessors of the grouping node
                n_preds = nx.dfs_predecessors(group_tree, n)

                # Remove groups from the list of predecessor;
                # each group will be processed indepedently
                n_preds = [n for n in n_preds.keys() if n not in groups]

                # Create a subgraph consisting of preceding nodes
                n_subgraph = graph.subgraph(n_preds)

                # Get layout features for each node
                n_feats = [ad['features'] for n, ad in
                           n_subgraph.nodes(data=True)]

                # Cast stacked features into a 2D numpy array
                stacked_feats = np.array(n_feats)

                # Get average centre point for group by slicing the array
                x_avg = np.average(stacked_feats[:, 0])
                y_avg = np.average(stacked_feats[:, 1])

                # Add up their area
                a_sum = np.sum(stacked_feats[:, 2])

                # Average the solidity
                s_avg = np.average(stacked_feats[:, 3])

                # Concatenate the features
                layout_feats = np.concatenate([[x_avg], [y_avg],
                                               [a_sum], [s_avg]], axis=0)

                # Update group feature dictionary
                upd_group_feats = {n: {'features': layout_feats,
                                       'kind': attr['kind']}}

                # Update group features
                nx.set_node_attributes(graph, upd_group_feats)

        # Add edge types to the grouping layer, as these are not defined in the
        # JSON annotation. To do so, get the edges from the grouping graph.
        edges = graph.edges(data=True)

        # Loop over the edges in the graph
        for src, dst, features in edges:

            # Add edge type unde key 'kind' to the edge_features dictionary
            edge_features[src, dst] = {'kind': 'grouping'}

        # Update edge features in the grouping graph
        nx.set_edge_attributes(graph, edge_features)

        # Encode edge features
        self._encode_edges(graph, self.edge_dict['grouping'])

        # Update the grouping graph in the graphs dictionary
        graphs['grouping'] = graph

        # Now that the grouping layer has been created, check which other
        # annotation layers must be included in the graph-based representation.

        # The combination of grouping and connectivity layers is a relatively
        # simple case.
        if layers == "grouping+connectivity":

            # If a connectivity graph exists, merge it with the grouping graph
            if graphs['connectivity'] is not None:

                # Use nx.compose() to combine the grouping and connectivity
                # graphs
                graph = nx.compose(graphs['connectivity'], graphs['grouping'])

            # Encode edge type information using numerical labels
            self._encode_edges(graph, self.edge_dict['connectivity'])

            # Update the grouping graph
            graphs['grouping'] = graph

        # The connectivity layer alone is a bit more complex, as the children of
        # grouping nodes need to be copied over to the connectivity graph.
        if layers == 'connectivity' and graphs['connectivity'] is not None:

            # Get the grouping and connectivity graphs
            conn_graph = graphs['connectivity']
            group_graph = graphs['grouping']

            # Get a list of nodes in the connectivity graph
            conn_nodes = list(conn_graph.nodes(data=True))

            # Get a list of grouping nodes in the connectivity graph
            grouping_nodes = [n for n, attr_dict in conn_nodes
                              if attr_dict['kind'] == 'group']

            # If grouping nodes are found, get their children and add them to
            # the graph
            if len(grouping_nodes) > 0:

                # Create a directed tree graph using depth-first search,
                # starting from the image constant I0.
                group_tree = nx.dfs_tree(group_graph, source="I0")

                # Loop over each grouping node
                for gn in grouping_nodes:

                    # Resolve grouping nodes by adding their children to the
                    # connectivity graph
                    self._resolve_grouping_node(gn, group_tree,
                                                group_graph, conn_graph)

            # If the connectivity graph does not include grouping nodes, simply
            # copy the node features from the grouping graph.
            n_subgraph = group_graph.subgraph(conn_graph.nodes)

            # Add these nodes to the connectivity graph
            conn_graph.add_nodes_from(n_subgraph.nodes(data=True))

            # Encode edge type information using numerical labels
            self._encode_edges(conn_graph, self.edge_dict['connectivity'])

            # Update the connectivity graph in the graphs dictionary
            graphs['connectivity'] = conn_graph

        # Start building the discourse graph by getting node features from the
        # grouping graph.
        if layers == 'discourse':

            # Get grouping and discourse graphs
            group_graph = graphs['grouping']
            rst_graph = graphs['discourse']

            # Reverse node type dictionary for the grouping layer
            rev_group_dict = {int(v.item()): k for k, v in
                              self.node_dict['grouping'].items()}

            # Re-encode node types to ensure that node types do not clash with
            # those defined for discourse graph
            upd_node_types = {k: rev_group_dict[int(v['kind'].item())]
                              for k, v in group_graph.nodes(data=True)}

            # Update node attributes for the grouping graph
            nx.set_node_attributes(group_graph, upd_node_types, 'kind')

            # Get the nodes participating in the discourse graph from the
            # grouping graph using the .subgraph() method.
            subgraph = group_graph.subgraph(rst_graph.nodes)

            # Add these nodes back to the discourse graph with their features
            # and numerical labels. These will overwrite the original nodes.
            rst_graph.add_nodes_from(subgraph.nodes(data=True))

            # Check if discourse graph contains groups or split nodes. Split
            # nodes are used to preserve the tree structure in case a diagram
            # element participates in multiple RST relations.
            for n, attr_dict in rst_graph.copy().nodes(data=True):

                # Check if the node is a group
                if 'group' in attr_dict['kind']:

                    # Create a directed tree graph using depth-first search,
                    # starting from the image constant I0.
                    group_tree = nx.dfs_tree(group_graph, source="I0")

                    # Resolve grouping nodes by adding their children to the
                    # discourse graph.
                    self._resolve_grouping_node(n, group_tree,
                                                group_graph, rst_graph)

                # Check node for the copy_of attribute, which contains a
                # reference to the node which has been split.
                if 'copy_of' in attr_dict.keys():

                    # Get the identifier of the node in AI2D layout annotation
                    n_orig_id = attr_dict['copy_of']
                    n_orig_kind = attr_dict['kind']

                    # Fetch node data from the AI2D layout annotation
                    layout_feats = self._parse_ai2d_layout(ai2d_ann,
                                                           h,
                                                           w,
                                                           n_pix,
                                                           n_orig_kind,
                                                           n_orig_id)

                    # Add updated features to a dictionary
                    upd_node_feats = {n: {'features': layout_feats,
                                          'kind': n_orig_kind}}

                    # Update node features in the graph
                    nx.set_node_attributes(rst_graph, upd_node_feats)

                # Check if the node is a relation
                if 'relation' in attr_dict['kind']:

                    # Get integer label for RST relation
                    rst_int_label = self.node_dict['relations'][attr_dict['rel_name']]

                    # Get node labels and encode using label binarizer
                    rst_label = self._rst_binarizer.transform(rst_int_label)

                    # Check if label smoothing is requested:
                    if self._smooth_labels:

                        # Cast into float for label smoothing
                        rst_label = np.asarray(rst_label, dtype=np.float64)

                        # Smooth the labels by a factor of 0.1
                        rst_label *= (1 - 0.1)
                        rst_label += (0.1 / rst_label.shape[1])

                    # Store encoded information into the updated features dict
                    upd_node_feats = {n: {'features': rst_label.flatten()}}

                    # Set the updated features to nodes in the discourse graph
                    nx.set_node_attributes(rst_graph, upd_node_feats)

            # Check if a NetworkX graph should be returned
            if self._return_nx:

                return rst_graph

            # Convert node identifiers to integers. This needs to be performed
            # before creating a heterograph.
            rst_graph = nx.convert_node_labels_to_integers(rst_graph,
                                                           first_label=0)

            # Get nodes and convert to NumPy array; get unique nodes; get node
            # type index vector
            nodes = np.asarray([attr['kind'] for n, attr in
                                rst_graph.nodes(data=True)]).flatten()

            ntypes = np.unique(nodes)

            node_ixs = np.array([np.where(ntypes == n) for n in
                                 np.nditer(nodes)], dtype=np.int64).flatten()

            # Do the same for edges
            edges = np.asarray([attr['kind'] for s, t, attr in
                                rst_graph.edges(data=True)]).flatten()

            etypes = np.unique(edges)

            edge_ixs = np.array([np.where(etypes == e) for e in
                                 np.nditer(edges)], dtype=np.int64).flatten()

            # Create DGL graph object from the discourse graph
            g = dgl.from_networkx(rst_graph)

            # Assign node and edge types
            g.ndata[dgl.NTYPE] = torch.LongTensor(node_ixs)
            g.edata[dgl.ETYPE] = torch.LongTensor(edge_ixs)

            # Create a DGL heterograph from the DGL graph object
            hg = dgl.to_heterogeneous(g, ntypes, etypes)

            # Loop over node types in the heterograph
            for ntype in hg.ntypes:

                # Get unique node identifiers for this node type; cast to list
                rst_node_ids = hg.nodes[ntype].data[dgl.NID].tolist()

                # Loop over RST node identifiers
                features = np.vstack([rst_graph.nodes[node_id]['features']
                                      for node_id in rst_node_ids])

                # Add features to DGL heterograph
                hg.nodes[ntype].data['features'] = torch.from_numpy(features)

            # Update the RST graph
            graphs['discourse'] = hg

        # Return all graphs
        return graphs

    # Build graphs from JSON annotation
    def _build_graphs(self, img_id, layers):
        """
        Builds DGL or NetworkX graph objects from AI2D/AI2D-RST annotation.

        Params:
            img_id: A string with the unique identifier of the diagram.
            layers: A string with annotation layers to include in the graph.
            kwargs: Keywords and arguments controlling which graphs are built.

        Returns:
             A graph either as DGL graph (default) or NetworkX graph.
        """
        # Create paths to the JSON files and diagram image
        orig_json = self.orig_json_path / Path(f'{img_id}.json')
        rst_json = self.rst_json_path / Path(f'{img_id}.json')
        ai2d_img = self.img_path / Path(f'{img_id}')

        # Read AI2D annotation and diagram image. Note that OpenCV requires the
        # path as a string, hence one must use the .as_posix() method.
        orig_ann = self._load_annotation(orig_json.as_posix())
        rst_ann = self._load_annotation(rst_json.as_posix())
        diagram_img = cv2.imread(ai2d_img.as_posix())

        # Check which layers have been requested to be included in the graphs,
        # starting with the grouping layer.
        if layers == 'grouping':

            # Build grouping graph
            graphs = {'grouping': self._parse_ai2d_rst_json(rst_ann)[0]}

            # Extract features from the JSON annotation and add this to graph
            graphs = self._extract_features(graphs,
                                            orig_ann,
                                            diagram_img,
                                            layers)

            # Check if a NetworkX graph should be returned
            if self._return_nx:

                return graphs['grouping']

            # Convert the grouping graph labels into integers
            group_graph = nx.convert_node_labels_to_integers(graphs['grouping'],
                                                             first_label=0)

            # Convert NetworkX graph into a DGL graph object
            g = dgl.from_networkx(group_graph,
                                  node_attrs=['kind', 'features'],
                                  edge_attrs=['kind'])

            # Return the DGL graph object
            return g

        # If requested, build both grouping and connectivity layers
        if layers == 'grouping+connectivity':

            # Build grouping and connectivity graphs
            gc_graphs = {'grouping': self._parse_ai2d_rst_json(rst_ann)[0],
                         'connectivity': self._parse_ai2d_rst_json(rst_ann)[1]}

            # Extract features from the JSON annotation and add them to graphs
            gc_graphs = self._extract_features(gc_graphs,
                                               orig_ann,
                                               diagram_img,
                                               layers)

            # Check if a NetworkX graph should be returned
            if self._return_nx:

                return gc_graphs['grouping']

            # Convert the grouping graph labels into integers
            gc_graph = nx.convert_node_labels_to_integers(gc_graphs['grouping'],
                                                          first_label=0)

            # Convert NetworkX graph into a DGL graph object
            gc = dgl.from_networkx(gc_graph,
                                   node_attrs=['kind', 'features'],
                                   edge_attrs=['kind'])

            # Return the DGL graph object
            return gc

        # If requested, build only the connectivity graph
        if layers == 'connectivity':

            # Build grouping and connectivity graphs
            c_graph = {'grouping': self._parse_ai2d_rst_json(rst_ann)[0],
                       'connectivity': self._parse_ai2d_rst_json(rst_ann)[1]}

            # Check that connectivity annotation exists for current diagram
            if c_graph['connectivity'] is None:

                return

            # Extract features from the JSON annotation and add them to graphs
            c_graph = self._extract_features(c_graph,
                                             orig_ann,
                                             diagram_img,
                                             layers)

            # Check if a NetworkX graph should be returned
            if self._return_nx:

                return c_graph['connectivity']

            # Convert the grouping graph labels into integers
            c_graph = nx.convert_node_labels_to_integers(c_graph['connectivity'],
                                                         first_label=0)

            # Convert NetworkX graph into a DGL graph object
            cg = dgl.from_networkx(c_graph,
                                   node_attrs=['kind', 'features'],
                                   edge_attrs=['kind'])

            # Return the DGL graph object
            return cg

        # If requested, build only the discourse graph
        if layers == 'discourse':

            # Build grouping and discourse graphs
            graphs = {'grouping': self._parse_ai2d_rst_json(rst_ann)[0],
                      'discourse': self._parse_ai2d_rst_json(rst_ann)[2]}

            # Extract features from the JSON annotation and update the graphs
            # dictionary
            dg_graphs = self._extract_features(graphs,
                                               orig_ann,
                                               diagram_img,
                                               layers)

            # Check if a NetworkX graph should be returned
            if self._return_nx:

                return dg_graphs

            # Return the discourse graph, which is already a DGLHeteroGraph
            # object
            return dg_graphs['discourse']

        if layers == 'discourse+connectivity':

            raise NotImplementedError

    # Add class property for reporting dataset size
    def __len__(self):

        return len(self.diagrams)

    # Return items from the dataset
    def __getitem__(self, index):

        # Return diagram graph and label
        return self.diagrams[index], self.labels[index], self.file_ids[index]
