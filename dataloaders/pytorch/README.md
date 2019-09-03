# A data loader for PyTorch and Deep Graph Library

This directory contains a data loader for [PyTorch](https://pytorch.org) and [Deep Graph Library](https://www.dgl.ai) to facilitate the use of AI2D-RST dataset.

## Directory structure

| Directory or file | Description |
| `dataloader.py` | Defines the AI2D-RST dataloader using PyTorch's DataLoader class. | 
| `features.py` | Defines various functions for extracting features from the AI2D annotation. | 
| `train_gcn.py` | Script for training a simple Graph Convolutional Network to classify diagrams according to their macro-groups. | 
