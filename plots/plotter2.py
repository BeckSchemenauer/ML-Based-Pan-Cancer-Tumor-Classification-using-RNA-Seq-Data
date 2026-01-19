import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# palette
pastel_colors = list(cm.Set2.colors) + list(cm.Set3.colors)

# file paths and Labels
file_map = {
    "Logistic Regression": 'nested_cv_logreg/logreg_nested_cv_results.csv',
    "SVM": 'nested_cv_svm/svm_nested_cv_results.csv',
    "XGBoost": 'nested_cv_xgb/xgb_nested_cv_results.csv',
    "k-NN": 'nested_cv_knn/knn_nested_cv_results.csv',
    "Neural Network": 'nested_cv_nn/nn_nested_cv_results.csv',
}
labels = list(file_map.keys())

dfs = []
for label, path in file_map.items():
    df = pd.read_csv(path)

    # Inject N=39 if missing
    if 39 not in df['N'].values:
        max_N = df['N'].max()
        if max_N < 39:
            data_at_max_N = df[df['N'] == max_N].copy()
            data_at_max_N['N'] = 39
            df = pd.concat([df, data_at_max_N], ignore_index=True)

    df['model'] = label
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# Filter for N = 5, 10, 15, 39
genes_of_interest = [5, 10, 15, 39]
subset_df = combined_df[combined_df['N'].isin(genes_of_interest)]

# --- Plot Generation ---

fig, ax = plt.subplots(figsize=(12, 6))

N_values = sorted(subset_df['N'].unique())
models = labels
n_models = len(models)
bar_width = 0.15
x = np.arange(len(N_values))


def get_metrics(n, model):
    data = subset_df[(subset_df['N'] == n) & (subset_df['model'] == model)]
    if data.empty:
        return np.nan, np.nan
    return data['mean_acc'].iloc[0] * 100, data['std'].iloc[0] * 100


# Iterate through models to plot bars
for i, model in enumerate(models):
    color = pastel_colors[i % len(pastel_colors)]
    accuracies = []
    stds = []

    for n in N_values:
        acc, std = get_metrics(n, model)
        accuracies.append(acc)
        stds.append(std)

    offset = bar_width * (i - n_models / 2 + 0.5)

    ax.bar(
        x + offset,
        accuracies,
        bar_width,
        label=model,
        color=color,
    )

    ax.errorbar(
        x + offset,
        accuracies,
        yerr=stds,
        fmt='none',
        capsize=4,
        color='black',
        alpha=0.8,
        linewidth=1.2
    )

# --- Apply Style Rules ---

ax.set_xlabel("Number of Genes (N)", fontsize=12)
ax.set_ylabel("Prediction Accuracy (%)", fontsize=12)
ax.set_title("Model Accuracy Comparison at Key Gene Counts\n(Mean ± SD)", fontsize=14)


# Format N-labels
def format_n_label(n_val):
    return f"N={n_val}"


ax.set_xticks(x)
ax.set_xticklabels([format_n_label(n) for n in N_values])

# Horizontal grid behind bars
ax.set_axisbelow(True)
ax.yaxis.grid(True)
ax.xaxis.grid(False)

# Despine top and right
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# **Fixed y-axis range**
ax.set_ylim(90, 101)

# --- Legend outside, above ---
ax.legend(
    title="Model",
    frameon=True,
    facecolor="white",
    edgecolor="black"
)

plt.tight_layout()
plt.savefig("AAA_grouped_bar_chart.png",
            dpi=300, bbox_inches="tight")
