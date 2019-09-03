# -*- coding: utf-8 -*-

# Import packages
from collections import namedtuple
from torch.utils import data
from dgl import DGLGraph
from features import *
import cv2
import dgl
import os
import numpy as np
import networkx as nx
import torch


# Make AI2D class and AI2DBatch namedtuple available through import
# using __all__, which provides a list of public objects for a module
# when importing using import *
__all__ = ['AI2D_RST', 'AI2D_RSTBatch', 'create_batch']

# Define namedtuple for batched graphs
AI2D_RSTBatch = namedtuple('AI2D_RSTBatch', ['graphs', 'labels'])


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


# This defines the AI2D-RST dataset based on torch.utils.data.Dataset class
class AI2D_RST(data.Dataset):

    # Initialize class
    def __init__(self, ids, labels, layers, img_path, ai2d_path, ai2d_rst_path):
        """
        Initialize the AI2D-RST dataset.

        Parameters:
            ids: A list of AI2D identifiers that correspond to their filenames.
            labels: A list of labels to match with the filenames.
            layers: A list of annotation layers that defines the graphs to be
                    built. Valid values include 'grouping', 'connectivity' and
                    'discourse'.
            img_path: Path to the directory containing AI2D images.
            ai2d_path: Path to the directory containing AI2D JSON files.
            ai2d_rst_path: Path to the directory containing AI2D-RST JSON files.

        Returns:
            The AI2D-RST dataset as a torch.utils.data.Dataset object.
        """
        # Initialize class attributes.
        self.ids = ids
        self.labels = labels
        self.diagrams = []
        self.layers = layers
        self.img_path = img_path
        self.ai2d_path = ai2d_path
        self.ai2d_rst_path = ai2d_rst_path

        # Load diagrams
        self._load()

    # Add class property for reporting dataset size
    def __len__(self):
        return len(self.ids)

    # Load AI2D-RST annotation from JSON files
    def _load(self):

        # Create graph for each diagram
        for diagram_id in self.ids:
            
            # Append graph to the list in self.diagrams
            self.diagrams.append(self._build_graph(diagram_id))

    # Build graphs from JSON annotation
    def _build_graph(self, img_id):

        # Get filenames for annotation and image
        ai2d_json = os.path.join(self.ai2d_path,
                                 '{}.json'.format(img_id))
        ai2d_img = os.path.join(self.img_path,
                                '{}'.format(img_id))
        ai2d_rst_json = os.path.join(self.ai2d_rst_path,
                                     '{}.json'.format(img_id))
            
        # Load graphs from AI2D and AI2D-RST JSON and read the diagram image
        ai2d_annotation = load_annotation(ai2d_json)
        ai2d_rst_annotation = load_annotation(ai2d_rst_json)
        diagram_img = cv2.imread(ai2d_img)

        # Create the requested NetworkX graphs from AI2D-RST annotation
        graphs = create_graph(ai2d_annotation,
                              diagram_img,
                              ai2d_rst_annotation,
                              self.layers)

        # Get a single graph from the dictionary
        diagram_graph = graphs['grouping']
        
        # Convert node labels to integers for DGLGraph
        diagram_graph = nx.convert_node_labels_to_integers(diagram_graph,
                                                           first_label=0)
    
        # Initialize a DGLGraph object and the node initializer
        g = DGLGraph()

        # Load networkx graph
        g.from_networkx(diagram_graph, node_attrs=['features'])

        # Return graph
        return g

    # Return items from the dataset
    def __getitem__(self, index):

        # Return diagram graph and label
        return self.diagrams[index], self.labels[index]
