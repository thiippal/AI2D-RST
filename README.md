# AI2D-RST: A multimodal corpus of 1000 primary school science diagrams

This repository contains resources related to AI2D-RST, a multimodal corpus of 1000 English-language diagrams that represent topics in primary school natural science, such as food webs, life cycles, moon phases and human physiology. The corpus is based on the [Allen Institute for Artificial Intelligence Diagrams (AI2D) dataset](https://allenai.org/plato/diagram-understanding/), a collection of diagrams with crowd-sourced descriptions. 

Building on the layout segmentation in AI2D, the AI2D-RST corpus presents a multi-layer annotation schema that provides a rich, graph-based description of their structure. The data and its description are introduced below using a single diagram (#2185 in the corpus) as an example.

1. **Diagram image**: The AI2D dataset contains 4907 diagrams scraped from Google Image Search by using chapter titles in primary school science textbooks (for ages 6–11) as search terms. The AI2D-RST corpus contains a subset of 1000 diagrams randomly sampled from AI2D. <br> <img src="examples/2185.png" width=400>
2. **Layout segmentation**: The AI2D dataset contains segmentations for four types of elements: blobs (images, line art, photographs, etc.), text blocks, arrows and arrowheads. AI2D-RST uses blobs, text blocks and arrows, which provide the foundation for the graphs representing diagram structure. <br> <img src="examples/segmentation_2185.png" width=400>
3. **Grouping**: The grouping layer captures perceptual groups of diagram elements, that is, groups of elements that are likely to be perceived as belonging together. Elements that form a group are assigned a parent node with the prefix `G`. The grouping annotation is represented using an acyclic tree graph, whose root node `IO` stands for the entire diagram. <br> <img src="examples/layout_2185.png" width=400>
4. **Macro-grouping**: The macro-grouping layer captures the generic principles of organisation governing the diagrams or its parts. If the diagram contains a single macro-group, macro-grouping information is assigned to the `I0` node that represents the entire diagram, or if the diagram features multiple macro-groups, to appropriate grouping nodes with the prefix `G`. <br>
<img src="examples/macro-groups.png" width=400>
5. **Connectivity**: The connectivity layer describes connections between diagram elements or their groups, which are signalled visually using diagrammatic elements such as arrows and lines. Connectivity is represented using a mixed cyclic graph, which may feature both undirected and directed edges. <br> <img src="examples/connectivity_2185.png" width=300>
6. **Discourse structure**: The discourse structure layer describes discourse relations between diagram elements using [Rhetorical Structure Theory (RST)](http://sfu.ca/rst/), an established theory of text organisation and coherence<br> <img src="examples/rst_2185.png" width=400> 



## Repository structure

`annotator`: The annotator used to create the AI2D-RST corpus.

`dataloader`: A data loader for PyTorch.

`examples`: Example visualisations used in this document.

`utils`: Various utilities and convenience functions for working with AI2D-RST.

## Download the corpus

Insert download link here.

## Installation

## Annotation schema
