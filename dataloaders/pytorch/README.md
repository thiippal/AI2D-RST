# PyTorch DataLoader for AI2D-RST

This directory contains a [PyTorch](https://pytorch.org) DataLoader for the AI2D-RST dataset that yield [Deep Graph Library](https://www.dgl.ai) and [NetworkX](https://networkx.org/) graphs.

## Directory structure

| Directory or file | Description |
| ----------------- | ----------- |
| `ai2d_rst.py` | Defines a PyTorch DataLoader for AI2D-RST. | 
| `train_gcn.py` | Trains a simple Graph Convolutional Network to classify diagrams. |

## How to create a DataLoader for AI2D or AI2D-RST

You can create a PyTorch DataLoader by calling the `AI2D_RST` class, as exemplified below:

```
dataset = AI2D_RST(cat_path="path_to_categories.json",
                   img_path="path_to_ai2d_images/",
                   orig_json_path="path_to_ai2d_annotation/",
                   rst_json_path="path_to_ai2d_rst_annotation/",
                   layers='discourse',
                   nx=True,
                   node_types=True,
                   smooth=False
                   )
```

The first four arguments define paths to

 1. the JSON file that contains class labels for entire diagrams (`cat_path`)
 2. the directory with original AI2D diagram images (`img_path`), 
 3. the directory with original AI2D JSON annotations (`orig_json_path`),
 4. the directory with AI2D-RST JSON annotations (`rst_json_path`).

The following keywords and arguments can be used to customise the output:

 1. `layers`: Which annotation layers does the graph represent. Possible values for AI2D-RST include: `grouping`, `connectivity`, `grouping+connectivity` and `discourse`.
 2. `nx`: Return a NetworkX graph instead of a DGL graph (default: False).
 3. `node_types`: Add node type information to the feature vector for each node (True/False).
 4. `smooth`: Use smoothed labels when adding node type information to node features (True/False).

## DataLoader output

The output of the DataLoader depends on (1) the annotation layers and (2) node features requested.

In `grouping`, `connectivity` and `grouping+connectivity` graphs, each node that represents a diagram element contains a four-dimensional feature vector. 

These four dimensions represent the following information:

 1. The relative position of element in the layout along the X-axis, calculated by dividing the coordinate with the width of the image
 2. The relative position of element in the layout along the Y-axis, calculated by dividing the coordinate with the width of the image
 3. The relative area occupied by the element, calculated by dividing the number of pixels in the area by the total number of pixels in the image
 4. The solidity of the element shape, calculated by dividing the area covered by the element by its convex hull

For grouping nodes, which capture the hierarchal organisation of diagram elements, node features are based on the features of their children.

In `discourse` graph, node features are the same as those described above for diagram elements.

For nodes that stand for specific rhetorical relations, the features consist of a 25-dimensional one-hot encoded vector that determines the rhetorical relation in question.
