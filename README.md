# ML-Based-Pan-Cancer-Tumor-Classification-using-RNA-Seq-Data

A machine learning framework for analyzing high-dimensional gene expression data (RNA-Seq). This project implements feature selection via Random Forest importance within a nested cross-validation loop to identify stable genes and evaluate various classification models.

## Project Structure

* **`Data.py`**: Core utility for data loading, standard scaling, and Exploratory Data Analysis (EDA). Includes functions for class distribution plots and PCA (2D/3D).
* **`prediction.py`**: A modular "model bank" containing scikit-learn and XGBoost pipelines. Models include Logistic Regression, SVM, KNN, MLP (Neural Network), and XGBoost.
* **`full_pipeline.py`**: The primary execution script. Implements an efficient nested CV:
    * **Inner Loop**: Ranks features using Random Forest importance.
    * **Outer Loop**: Evaluates all models on the top-N features (N=1 to 50) to determine optimal feature set size.
* **`baseline.py`**: Provides a control experiment by evaluating models using random feature selection instead of RF importance. Uses logistic regression for prediction.
* **`gene_mapper.py`**: A post-processing script that maps internal dummy names back to biological gene symbols.
* * **`plots/`**: Contains specialized visualization scripts for performance analysis. See the internal [`plots/README.md`](./plots/README.md) for detailed usage of the accuracy curve and bar chart generators.

### Prerequisites
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, plotly

### Data Setup
Place your transcriptomics data in a `./Data/` directory or use the given data:
- `data.csv`: Gene expression matrix (Samples x Genes).
- `labels.csv`: Target classes for each sample.

### Running the Analysis
1. **Initial EDA**:
   Run `python Data.py` to generate PCA plots and class distribution visualizations.

2. **Run Nested CV**:
   Execute `python full_pipeline.py`. This will iterate through 5 outer folds. In each fold, it calculates feature importance once and tests all 5 classifiers across a range of Top-N features.

3. **Verify with Baseline**:
   Run `python baseline.py` to compare your RF-ranked results against random feature selection.

4. **Map Genes to Biological IDs**:
   The dataset uses dummy names (`gene_0`, `gene_1`, etc.). To identify the actual genes, you must provide the **original platform specifications** (e.g., from Synapse or the UCI repository) as an external reference file for `gene_mapper.py`. 
   
   > **Note:** The mapping relies on the fact that dummy attributes are ordered consistently with the original submission. Due to data restrictions, the mapping reference file is not included in this repository and must be obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) or [Synapse](https://www.synapse.org/#!Synapse:syn4301332).
