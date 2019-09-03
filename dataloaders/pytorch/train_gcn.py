# -*- coding: utf-8 -*-

"""
This script defines a small Graph Convolutional Network (GCN), which is the trained using
the AI2D/AI2D-RST annotations to classify diagrams to different macro-groups identified
in the AI2D-RST annotation.

The script also demonstrates the usage of the AI2D-RST dataloader for PyTorch.

Usage:
    python train_gcn.py -c categories.json -i images/ -a ai2d_json/ -ar ai2d_rst_json/
    
Arguments:
    -c/--categories: Path to the *AI2D-RST* categories JSON file.
    -i/--images: Path to the directory with the AI2D images.
    -a/--ai2d_json: Path to the directory with AI2D JSON files.
    -ar/--ai2d_rst_json: Path to the directory with AI2D-RST JSON files.
    
Returns:
    Trains the GCN using Deep Graph Library (DGL) and prints out a classification report.
"""

# Import packages
from dataloader import *
from features import *
from torch.utils import data
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import argparse
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Set up the argument parser
ap = argparse.ArgumentParser()

# Define arguments
ap.add_argument("-c", "--categories", required=True,
				help="Path to the AI2D-RST categories_ai2d-rst.json file.")
ap.add_argument("-i", "--images", required=True,
				help="Path to the directory containing AI2D images.")
ap.add_argument("-a", "--ai2d", required=True,
				help="Path to the directory containing AI2D JSON files.")
ap.add_argument("-ar", "--ai2d_rst", required=True,
				help="Path to the directory containing AI2D-RST JSON files.")

# Parse arguments
args = vars(ap.parse_args())

# Assign arguments to variables
cat_path = args['categories']
img_path = args['images']
ai2d_path = args['ai2d']
ai2d_rst_path = args['ai2d_rst']

# Define layers and message-passing functions for the Graph Convolutional Network

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')


def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = th.mean(nodes.mailbox['m'], 1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.gcn1 = GCN(in_dim, hidden_dim, F.relu)
        self.gcn2 = GCN(hidden_dim, hidden_dim, F.relu)
    
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)

        g.ndata['h'] = x

        hg = dgl.mean_nodes(g, 'h')
        
        return self.classify(hg)


# Set device
device = th.device('cpu')

# Load categories
categories = load_annotation(cat_path)

# Initialize label encoder and encode integer labels
label_enc = LabelEncoder()
label_enc.fit_transform(list(categories.values()))

# Create a dictionary mapping filenames to labels
labels = {k: label_enc.transform([v]) for k, v in categories.items()}

# Convert labels into a numpy array for calculating class weights
label_arr = np.concatenate(list(labels.values()))

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(label_arr),
                                     y=label_arr)

# Wrap class weights into a torch Tensor
class_weights = th.FloatTensor(class_weights)

# Get diagram identifiers and labels
ids = list(labels.keys())
labels = list(labels.values())

# Create the AI2D dataset
AI2D_RST_data = AI2D_RST(ids, labels, layers=['grouping'], img_path=img_path, 
						ai2d_path=ai2d_path, ai2d_rst_path=ai2d_rst_path)

# Define training (90%) and validation (10%) split sizes
train_len = int(0.90 * len(AI2D_RST_data))
valid_len = (len(AI2D_RST_data) - train_len)

# Print status
print("Using {} graphs for training and {} for validation ...".format(train_len,
                                                                      valid_len)
      )

# Create training and validation splits
AI2D_RST_train, AI2D_RST_valid = data.random_split(AI2D_RST_data, [train_len,
                                                                   valid_len])

# Initialize dataloaders for training and testing data
train_loader = data.DataLoader(dataset=AI2D_RST_train,
                               batch_size=32,
                               shuffle=True,
                               collate_fn=create_batch(device),
                               num_workers=0)

valid_loader = data.DataLoader(dataset=AI2D_RST_valid,
                               batch_size=len(AI2D_RST_valid),
                               shuffle=True,
                               collate_fn=create_batch(device),
                               num_workers=0)

# Initialize model with a 10-dimensional input feature vector and 64 neurons in the hidden
# layer. The number of classes is retrieved from the label encoder.
model = Classifier(10, 64, len(label_enc.classes_))

# Set up loss function with class weights and the optimizer.
loss_func = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

# Set the model to training mode.
model.train()

# A list for storing the losses
epoch_losses = []

for i, epoch in enumerate(range(100)):
    
    epoch_loss = 0
    
    # Iterate over batches of training data
    for iter, (bg, label) in enumerate(train_loader):

		# Feed data to the model and retrieve output
        prediction = model(bg, bg.ndata['features'].float())
        
        # Calculate loss
        loss = loss_func(prediction, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

# Switch to evaluation mode
model.eval()

# Loop over the validation loader
for test_X, test_Y in valid_loader:
    
    test_Y = th.tensor(test_Y).float().view(-1, 1)
    
    probs_Y = th.softmax(model(test_X, test_X.ndata['features'].float()), 1)
    
    sampled_Y = th.multinomial(probs_Y, 1)
    
    argmax_Y = th.max(probs_Y, 1)[1].view(-1, 1)

    # Get labels for predicted classes
    int_labels = np.unique(test_Y.numpy().astype(np.int32))
    str_labels = label_enc.inverse_transform(int_labels)
    
    print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
        (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
    print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
        (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))

    print(classification_report(test_Y, argmax_Y, labels=int_labels, 
    	  target_names=str_labels))
