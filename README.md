# AI2D-RST: A multimodal corpus of 1000 primary school science diagrams

This repository contains resources related to AI2D-RST, a multimodal corpus of 1000 English-language diagrams that represent topics in primary school natural science, such as food webs, life cycles, moon phases and human physiology. The corpus is based on the [Allen Institute for Artificial Intelligence Diagrams (AI2D) dataset](https://allenai.org/plato/diagram-understanding/), a collection of diagrams with crowd-sourced descriptions. 

Building on the layout segmentation in AI2D, the AI2D-RST corpus presents a multi-layer annotation schema that provides a rich description of their structure. Annotated by trained experts, the layers describe (1) the grouping of diagram elements into perceptual units, (2) the connections set up by diagrammatic elements such as arrows and lines, and (3) the discourse relations between diagram elements, which are described using [Rhetorical Structure Theory (RST)](http://sfu.ca/rst/), an established theory of text organisation and coherence. 

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
