# -*- coding: utf-8 -*-

"""
"""

# Import libraries
from dgl.nn.pytorch.conv import *
from ai2d_rst import AI2D_RST, create_batch
from torch.utils import data
from sklearn.metrics import f1_score
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Initialize the AI2D-RST dataset
dataset = AI2D_RST("../../../ai2d/categories_ai2d-rst_fine.json",
                   "../../../ai2d/images/",
                   "../../../ai2d/annotations/",
                   "../../../ai2d/ai2d-rst/",
                   layers='grouping'
                   )


# Define a Graph Convolutional Network (GCN) for graph classification
class GC_GCN(nn.Module):
    def __init__(self, in_dim, out_dim, n_classes, n_layers):
        super(GC_GCN, self).__init__()

        # Make key attributes available to the forward pass
        self.n_layers = n_layers

        # Initialize a list of layers
        self.layers = nn.ModuleList()

        # Add the first graph convolutional layer to the model
        self.layers.extend([GraphConv(in_dim,
                                      out_dim,
                                      activation=F.relu,
                                      allow_zero_in_degree=True)])

        # Update the number of layers needed
        remaining_layers = n_layers - 1

        # If more graph convolutional layers remain to be added, continue
        if remaining_layers > 0:

            # Add as many layers to the model as requested
            self.layers.extend([GraphConv(out_dim,
                                          out_dim,
                                          activation=F.relu)
                                for i in range(0, remaining_layers)])

        # Add final linear classifier
        self.layers.extend([nn.Linear(out_dim, n_classes)])

    def forward(self, g, features):

        # Assign features to variable 'x' as this is the variable name used
        # during the loop through the list of layers in self.layers.
        x = features

        # Set up a placeholder list to hold the features from the GCN layers
        gcn_out = []

        # Feed the graphs through GCN layers: the last two (-2) are the sort
        # pooling and linear classification layers.
        for i, graph_conv in enumerate(self.layers[:self.n_layers]):

            # Feed previous features to the next layer
            x = graph_conv(g, x)

            # Update node features
            g.ndata['features'] = x

            # Average node features
            avg_out = dgl.mean_nodes(g, 'features')

            # Return result from linear classifier at self.layers[-1]
            return self.layers[-1](avg_out)


# Define function for evaluating model performance
def evaluate(model, loader):

    # Do not calculate gradients
    with torch.no_grad():

        # Set the model to evaluation mode
        model.eval()

        # Set up placeholder lists for target and predicted labels
        target_labels, pred_labels = [], []

        # Loop over the loader
        for g, target_y in loader:

            # Get predicted labels
            target_y = target_y.view(-1, 1)

            # Feed data to model and get softmax probabilities
            probs_y = torch.softmax(model(g, g.ndata['features'].float()), 1)

            # Get argmax predictions
            pred_y = torch.max(probs_y, 1)[1].view(-1, 1)

            # Append true and predicted labels to lists
            target_labels.append(target_y)
            pred_labels.append(pred_y)

        # Concatenate target labels and predicted labels; convert to numpy array
        target_labels = torch.cat(target_labels).numpy()
        pred_labels = torch.cat(pred_labels).numpy()

        return target_labels, pred_labels


# Build GCN
model = GC_GCN(in_dim=4,
               out_dim=10,
               n_layers=2,
               n_classes=dataset.n_classes
               )

# Initialize weighted loss function
loss_func = nn.CrossEntropyLoss(weight=dataset.class_weights)

# Configure optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)

# Set the model to training mode
model.train()

# Define training, validation and testing splits at the beginning of each trial
train, valid = data.random_split(dataset, [800, 200])

# Initialize dataloaders for training and validation data
train_loader = data.DataLoader(dataset=train,
                               batch_size=2,
                               shuffle=True,
                               collate_fn=create_batch('cpu'),
                               num_workers=0
                               )

# Initialize dataloaders for training and validation data
valid_loader = data.DataLoader(dataset=valid,
                               batch_size=32,
                               shuffle=True,
                               collate_fn=create_batch('cpu'),
                               num_workers=0
                               )

# Begin training loop; train for 100 epochs
for i in range(0, 100):

    # Loop over batched graphs from the training data loader
    for bg, labels in train_loader:

        # Feed data to model and retrieve output
        preds = model(bg, bg.ndata['features'].float())

        # Calculate loss
        loss = loss_func(preds, labels)

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass
        loss.backward()
        optimizer.step()

    # Do not calculate gradients
    with torch.no_grad():

        # Get predictions for validation data
        valid_targets, valid_preds = evaluate(model, valid_loader)

        # Calculate macro F1 score
        v_m_f1 = f1_score(valid_targets, valid_preds, average='macro')

        # Print metric at end of epoch
        print(f"Epoch: {i + 1}, macro F1-score: {round(v_m_f1, 3)}")
