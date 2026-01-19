import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

# tableau palette
tableau_colors = list(cm.Set2.colors) + list(cm.Set3.colors)


def plot_nested_cv_results(paths, labels, max_N=None, mask=None):
    """
    Plot curves from multiple nested CV result CSVs.
    mask: list of indices to skip drawing (but still advance color cycle)
    """

    if mask is None:
        mask = []

    plt.figure(figsize=(7, 4.5))

    for i, (path, label) in enumerate(zip(paths, labels)):
        color = tableau_colors[i % len(tableau_colors)]

        # if masked → skip drawing, but don't skip color index (so same model is always same color)
        if i in mask:
            continue

        df = pd.read_csv(path)

        if max_N is not None:
            df = df[df["N"] <= max_N]

        N = df["N"].values
        mean = df["mean_acc"].values * 100
        std = df["std"].values * 100

        # Shaded SD region
        plt.fill_between(
            N,
            mean - std,
            mean + std,
            alpha=0.20,
            color=color
        )

        # Accuracy line
        plt.plot(
            N,
            mean,
            marker='o',
            linewidth=2,
            label=label,
            color=color
        )

    # Labels & title
    plt.xlabel("Number of Genes (N)", fontsize=12)
    plt.ylabel("Prediction Accuracy (%)", fontsize=12)
    plt.title("Prediction Accuracy vs Number of Genes\n(5-fold nested CV, mean ± SD)",
              fontsize=13)

    # Style rules
    ax = plt.gca()
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="black")
    plt.tight_layout()
    plt.show(dpi=300)


paths = [
    "nested_cv_logreg/logreg_nested_cv_results.csv",
    "nested_cv_logreg/logreg_nested_cv_results_random.csv",
    "nested_cv_svm/svm_nested_cv_results.csv",
    "nested_cv_knn/knn_nested_cv_results.csv",
    "nested_cv_xgb/xgb_nested_cv_results.csv",
    "nested_cv_nn/nn_nested_cv_results.csv",
]

labels = [
    "Logistic Regression (RF)",
    "Logistic Regression (Random)",
    "SVM",
    "k-NN",
    "XGBoost",
    "Neural Network",
]

plot_nested_cv_results(paths, labels, max_N=40, mask=[ 2, 3, 4, 5])
