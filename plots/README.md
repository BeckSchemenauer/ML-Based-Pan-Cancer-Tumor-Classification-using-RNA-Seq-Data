# Visualizations for Gene Selection & Model Performance

This directory contains scripts used to visualize the performance of various machine learning models as a function of the number of selected gene features.

## Scripts and Visuals

### 1. `line_accuracy_curves.py`
Generates a multi-line plot showing the mean accuracy and standard deviation (shaded region) for each model across a range of gene counts (e.g., $N=1$ to $40$).
* **Key Use Case**: Used to demonstrate the "elbow point" where adding more genes provides diminishing returns in accuracy.
* **Special Feature**: Includes a `mask` parameter to selectively hide models for clearer visual comparison between specific algorithms.

### 2. `grouped_bar_comparison.py`
Generates a grouped bar chart comparing all classifiers at specific, high-interest gene counts ($N = 5, 10, 15,$ and $39$).
* **Key Use Case**: Used for final performance benchmarking across models when the number of features is constrained.
* **Output**: Saves a high-resolution PNG (`AAA_grouped_bar_chart.png`) with a fixed Y-axis (90%–101%) to highlight the performance delta between top-performing models.
