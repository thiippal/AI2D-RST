# AI2D-RST: A multimodal corpus of 1000 primary school science diagrams

This repository contains resources related to AI2D-RST, a multimodal corpus of 1000 English-language diagrams that represent topics in primary school natural science, such as food webs, life cycles, moon phases and human physiology. The corpus is based on the [Allen Institute for Artificial Intelligence Diagrams (AI2D) dataset](https://allenai.org/plato/diagram-understanding/), a collection of diagrams with crowd-sourced descriptions. 

Building on the layout segmentation in AI2D, the AI2D-RST corpus presents a multi-layer annotation schema that provides a rich, graph-based description of their structure. Annotated by trained experts, the layers describe

| Layer | Example |
|:----- | :------- |
| **Diagram image**: The AI2D dataset contains 4907 diagrams scraped from Google Image Search by using chapter titles in primary school science textbooks (for ages 6–11) as search terms. The AI2D-RST corpus contains a subset of 1000 diagrams randomly sampled from AI2D. | <img src="examples/2185.png" width=300> | 
| **Layout segmentation** | <img src="examples/segmentation_2185.png" width=300> |
| **Grouping**: | <img src="examples/layout_2185.png" width=300> | 
| **Connectivity**: | <img src="examples/connectivity_2185.png" width=300> | 
| **Discourse structure**: | <img src="examples/rst_2185.png" width=300> | 


  1. grouping of diagram elements into perceptual units
  2. connections set up by diagrammatic elements such as arrows and lines
  3. discourse relations between diagram elements, described using [Rhetorical Structure Theory (RST)](http://sfu.ca/rst/), an established theory of text organisation and coherence

Each annotation layer in AI2D-RST is represented using a graph.



## Repository structure

`annotator`: The annotator used to create the AI2D-RST corpus.

`dataloader`: A data loader for PyTorch.

`examples`: Example visualisations used in this document.

`utils`: Various utilities and convenience functions for working with AI2D-RST.

## Download the corpus

Insert download link here.

## Installation

## Annotation schema
