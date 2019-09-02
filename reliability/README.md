# Evaluating the reliability of the AI2D-RST annotation

This directory contains scripts and data for measuring the reliability of the AI2D-RST annotation schema.

The following table summarizes the results of inter-annotator agreement measures among five annotators.

| Annotation layer | Number of samples | Fleiss' kappa |
| ---------------- | -------------: | -------------: |
| Grouping         | 256 | 0.836 |
| Macro-grouping   | 119 | 0.784 |
| Connectivity     | 239 | 0.878 |
| Discourse structure | 227  | 0.733 |

## Directory structure

| Directory or file | Description |
| ----------------- | ----------- |
| `connectivity/` | Contains a CSV file with annotations for connectivity. |
| `grouping/` | Contains a CSV file with annotations for grouping. |
| `macro-grouping/` | Contains a CSV file with annotations for macro-grouping. |
| `rst/` | Contains a CSV file with annotations for discourse structure. |
| `sampled_data` | Contains samples for connectivity, grouping and discourse structure annotation drawn from the AI2D-RST corpus as pandas DataFrames. |
| `measure_ia_connectivity.r` | Script for measuring agreement on connectivity. |
| `measure_ia_group.r` | Script for measuring agreement on grouping. |
| `measure_ia_macro.r` | Script for measuring agreement on macro-grouping. |
| `measure_ia_rst.r` | Script for measuring agreement on discourse structure. |
