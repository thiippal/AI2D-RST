# PyTorch DataLoaders for AI2D and AI2D-RST

This directory contains a [PyTorch](https://pytorch.org) DataLoader for the AI2D and AI2D-RST datasets that yield [Deep Graph Library](https://www.dgl.ai) and [NetworkX](https://networkx.org/) graphs.

## Directory structure

| Directory or file | Description |
| ----------------- | ----------- |
| `ai2d_rst.py` | Defines a PyTorch DataLoader for AI2D-RST. | 
| `ai2d.py` | Defines a PyTorch DataLoader for AI2D. | 
| `train_gcn.py` | Script for training a simple Graph Convolutional Network to classify diagrams. |

## How to use the DataLoaders

You can create a PyTorch DataLoader by calling the `AI2D_RST` or `AI2D` class.

The follwowing example shows how to create a DataLoader for AI2D-RST using the `AI2D_RST` class:

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
