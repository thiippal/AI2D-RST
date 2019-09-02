# AI2D-RST Annotator

This directory contains the in-house annotation tool used to create the AI2D-RST dataset.

See the docstrings in each file for instructions on how to use the scripts.

## Directory structure

| Directory or file | Description |
| ----------------- | ------------|
| `core/` | Contains code for the annotation tool. |
| `0.pkl` | Contains the diagram 0 from the AI2D dataset, which can be used to test the annotator. |
| `README.md` | This document. |
| `annotate.py` | Script for launching the annotator. |
| `check_status.py` | Script for checking the status of annotation in an output file. |
| `evaluate_agreement_connectivity.py` | Script for annotating connectivity as a part of inter-annotator agreement measures. |
| `evaluate_agreement_grouping.py` | Script for annotating grouping as a part of inter-annotator agreement measures. |
| `evaluate_agreement_macro.py` | Script for annotating macro-grouping as a part of inter-annotator agreement measures. |
| `evaluate_agreement_rst.py` | Script for annotating discourse structure as a part of inter-annotator agreement measures. |
| `examine_annotation.py` | Script for printing out the Pandas DataFrame containing AI2D-RST annotation. |
| `extract_full_annotation.py` | Script for extracting AI2D annotation from JSON format. |
| `sample_annotation.py`* | Script for sampling AI2D-RST annotation for inter-annotator agreement measures. |
| `visualize_ai2d_json.py`* | Script for visualizing the annotation in AI2D JSON files. |
| `visualize_annotation.py`* | Script for visualizing AI2D-RST annotation in Pandas DataFrames. |

TODO: * needs docstrings!
