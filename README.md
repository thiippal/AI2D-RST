# AI2D-RST: A multimodal corpus of 1000 primary school science diagrams

## Description

This repository contains resources related to AI2D-RST, a multimodal corpus of 1000 English-language diagrams that represent topics in primary school natural science, such as food webs, life cycles, moon phases and human physiology. The corpus is based on the [Allen Institute for Artificial Intelligence Diagrams (AI2D) dataset](https://allenai.org/plato/diagram-understanding/), a collection of diagrams with crowd-sourced descriptions. 

Building on the layout segmentation in AI2D, the AI2D-RST corpus presents a multi-layer annotation schema that provides a rich, graph-based description of diagram structure. The annotation was performed by trained experts. The data and its description are introduced below using a single diagram (#2185 in the corpus) as an example.

1. **Diagram image**: The AI2D dataset contains 4907 diagrams scraped from Google Image Search by using chapter titles in primary school science textbooks (for ages 6–11) as search terms. The AI2D-RST corpus contains a subset of 1000 diagrams randomly sampled from AI2D. <br> <img src="examples/2185.png" width=400/>
2. **Layout segmentation**: The AI2D dataset contains segmentations for four types of elements: blobs (images, line art, photographs, etc.), text blocks, arrows and arrowheads. AI2D-RST uses blobs, text blocks and arrows, which provide the foundation for the graphs representing diagram structure. <br> <img src="examples/segmentation_2185.png" width=400/>
3. **Grouping**: The grouping layer captures perceptual groups of diagram elements, that is, groups of elements that are likely to be perceived as belonging together. Elements that form a group are assigned a parent node with the prefix `G`. The grouping annotation is represented using an acyclic tree graph, whose root node `IO` stands for the entire diagram. The identifiers are carried over from the layout segmentation. <br> <img src="examples/layout_2185.png" width=400/>
4. **Macro-grouping**: The macro-grouping layer captures the generic principles of organisation governing the diagrams or its parts. If the diagram contains a single macro-group, macro-grouping information is assigned to the `I0` node that represents the entire diagram, or if the diagram features multiple macro-groups, to appropriate grouping nodes with the prefix `G`. The figure below shows the macro-groups identified in the AI2D-RST corpus, their counts and frequencies. <br> <img src="examples/macro-groups.png" width=400/>
5. **Connectivity**: The connectivity layer describes connections between diagram elements or their groups, which are signalled visually using diagrammatic elements such as arrows and lines. Connectivity is represented using a mixed cyclic graph, which may feature both undirected and directed edges. <br> <img src="examples/connectivity_2185.png" width=400/>
6. **Discourse structure**: The discourse structure layer describes discourse relations between diagram elements using [Rhetorical Structure Theory (RST)](http://sfu.ca/rst/), an established theory of text organisation and coherence. Relations are represented in the graph by nodes with the prefix `R`. The edges carry information about the nuclei and satellites of the RST relations. <br> <img src="examples/rst_2185.png" width=400/> 

## Repository structure

`annotator/`: The in-house annotation tool used to create the AI2D-RST corpus.

`dataloader/pytorch/`: AI2D-RST data loader for PyTorch.

`examples/`: Example visualisations used in this document.

`guide/`: The annotation manual for AI2D-RST.

`reliability/`: Scripts and data for measuring inter-annotator agreement.

`utils/`: Convenience functions for working with AI2D-RST.

`requirements.txt`: Defines the Python dependencies required by this repository for quick installation using `pip`.

## Download the corpus

Click [here](http://www.helsinki.fi/~thiippal/data/ai2d-rst_v1.zip) to download the AI2D-RST dataset.
## Install the corpus

1. Download the [Allen Institute for Artificial Intelligence Diagrams (AI2D) dataset](https://allenai.org/plato/diagram-understanding/).

2. Unzip the AI2D dataset. This will give you the following directory structure:

```
ai2d/
|--- annotations/
|- categories.json
|--- images/
|- license.txt
|- questions/
|- README.txt
```

3. Download the [AI2D-RST dataset](http://www.helsinki.fi/~thiippal/data/ai2d-rst_v1.zip).

4. Unzip the AI2D-RST dataset. Place the `categories_ai2d-rst.json` file and the `ai2d-rst` directory into the `ai2d/` directory. The directory structure for `ai2d/` should then look like this:

```
ai2d/
|--- ai2d_rst/
|--- annotations/
|- categories.json
|- categories_ai2d-rst.json
|--- images/
|- license.txt
|- questions/
|- README.txt
```

## JSON annotation schema

The JSON schema for AI2D-RST has three top-level keys, which correspond to the annotation layers. Each of the layers (`grouping`, `connectivity` and `rst`) can be found at the top level, along with the diagram identifier (`id`) defined in the original AI2D dataset:

```
{
	"id": "1189",
	"grouping": [],
	"connectivity": {},
	"rst": []
}
```
The `id` key is a string, `grouping` and `rst` are lists and `connectivity` is a dictionary.

### Grouping

The `grouping` key contains a list with each node as a dictionary, exported from [NetworkX as JIT JSON](https://networkx.github.io/documentation/stable/_modules/networkx/readwrite/json_graph/jit.html):

```
{
      "id": "B0",
      "name": "B0",
      "data": {
        "kind": "blobs"
      },
      "adjacencies": [
        {
          "nodeTo": "I0",
          "data": {}
        }
	]
}
```

### Connectivity

The `connectivity` key contains a dictionary with two keys, `nodes` for nodes and `edges` for edges:

```
{
    "nodes": [],
    "edges": []
}
```

The `nodes` list contains a list for each node. Each list has two items: the node identifier as a string and a dictionary specifying node attributes:

```
[
  "T0",
  {
    "kind": "text"
  }
]
```

The `edges` list contains a list for each edge. Each list has three items: the identifier of the source node, the identifier of the target node and a dictionary specifying edge attributes:

```
[
  "T0",
  "T2",
  {
    "kind": "directional"
  }
]
```

### RST

The `rst` key contains a list with each node as a dictionary, exported from [NetworkX as JIT JSON](https://networkx.github.io/documentation/stable/_modules/networkx/readwrite/json_graph/jit.html). 

The nodes may stand for either diagram elements or discourse relations.

For diagram elements, the dictionary is structured as follows:

```
{
      "id": "B0",
      "name": "B0",
      "data": {
        "kind": "blobs"
      }
}
```

For discourse relations, the dictionary is structured as follows:

```
{
      "id": "UTDA79",
      "name": "UTDA79",
      "data": {
        "kind": "relation",
        "nuclei": "T0 T1 T2 T3 T4",
        "rel_name": "joint",
        "id": "UTDA79"
      },
      "adjacencies": []
}
```

The keys of the `data` dictionary depend on the type of the discourse relation. Multinuclear relations contain the key `nuclei`, whose value is a list of identifiers separated by whitespace. Mononuclear relations, in turn, contain keys `nucleus` and `satellites`, as exemplified below:

```
    {
      "id": "RXG74P",
      "name": "RXG74P",
      "data": {
        "kind": "relation",
        "nucleus": "B0",
        "satellites": "UTDA79",
        "rel_name": "elaboration",
        "id": "RXG74P"
      }
```

In both cases, the `adjacencies` key holds a list of dictionaries, which define the edges:

```
{
	"nodeTo": "T0",
	"data": {
	 "kind": "nucleus"
	 }
}
```

## Contact information

Questions? Open an issue on GitHub or e-mail the maintainer at tuomo dot hiippala @ helsinki dot fi.
