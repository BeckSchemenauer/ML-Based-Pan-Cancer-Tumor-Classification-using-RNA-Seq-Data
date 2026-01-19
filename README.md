# ML-Based-Pan-Cancer-Tumor-Classification-using-RNA-Seq-Data

A machine learning framework for analyzing high-dimensional gene expression data (RNA-Seq). This project implements feature selection via Random Forest importance within a nested cross-validation loop to identify stable genes and evaluate various classification models.

## Project Structure

* **`code/`**: Contains the core machine learning pipeline and utility scripts.
    * **`data.py`**: Core utility for data loading, standard scaling, and Exploratory Data Analysis (EDA). Includes functions for class distribution plots and PCA (2D/3D).
    * **`prediction.py`**: A modular "model bank" containing scikit-learn and XGBoost pipelines. Models include Logistic Regression, SVM, KNN, MLP (Neural Network), and XGBoost.
    * **`full_pipeline.py`**: The primary execution script. Implements an efficient nested CV that ranks features using Random Forest importance and evaluates all models on the top-N features.
    * **`baseline.py`**: Provides a control experiment by evaluating models using random feature selection instead of RF importance.
    * **`gene_mapper.py`**: A post-processing script that maps internal dummy names back to biological gene symbols using an external reference.
* **`plots/`**: Contains specialized visualization scripts for performance analysis, including accuracy curves and grouped bar charts. See the internal [`plots/README.md`](./plots/README.md) for details.

### Prerequisites
- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, plotly

### Data Setup
Due to licensing and size restrictions, the raw transcriptomics data is not included in this repository. You must manually download the **Gene expression cancer RNA-Seq** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq).

Place the following files in a `./Data/` directory:
- `data.csv`: The gene expression matrix (Samples x Genes).
- `labels.csv`: Target classes corresponding to each sample index.

### Running the Analysis
1. **Initial EDA**:
   Run `python Data.py` to generate PCA plots and class distribution visualizations.

2. **Run Nested CV**:
   Execute `python full_pipeline.py`. This will iterate through 5 outer folds. In each fold, it calculates feature importance once and tests all 5 classifiers across a range of Top-N features.

3. **Verify with Baseline**:
   Run `python baseline.py` to compare your RF-ranked results against random feature selection.

4. **Map Genes to Biological IDs**:
   The dataset uses dummy names (e.g., `gene_0`). To identify the actual genes, you must provide an external reference file (such as the original platform specifications) for `gene_mapper.py`.

   > **Note:** Due to data restrictions, this mapping reference is not included in the repository. It must be obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq) or [Synapse](https://www.synapse.org/#!Synapse:syn4301332), where attributes are ordered consistently with the dummy names.

## Licensing & Attribution

### Code
This project's source code is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code for any purpose.

### Data Attribution
The dataset used in this project is the **Gene expression cancer RNA-Seq**, originally part of the RNA-Seq (HiSeq) PANCAN data set.

* **License**: The dataset is licensed under a [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. This allows for sharing and adaptation provided appropriate credit is given.
* **Citation**: Fiorini, S. (2016). gene expression cancer RNA-Seq [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5R88H.
* **Original Source**: The primary data set is maintained by the [Cancer Genome Atlas Pan-Cancer Analysis Project](https://www.synapse.org/#!Synapse:syn4301332).
