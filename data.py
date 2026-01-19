import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import matplotlib.cm as cm

DATA_FILE = './Data/data.csv'
LABEL_FILE = './Data/labels.csv'
N_PCA_COMPONENTS = 10


def load_data(data_path, label_path):
    try:
        features = pd.read_csv(data_path, index_col=0)
        labels = pd.read_csv(label_path, index_col=0)
        aligned_labels = labels.loc[features.index]

        print("Data loading complete.")
        print(f"Features shape (samples, genes): {features.shape}")
        print(f"Labels shape (samples, 1): {aligned_labels.shape}\n")

        return features, aligned_labels['Class']

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please make sure '{DATA_FILE}' and '{LABEL_FILE}' are in the same directory as this script.")
        return None, None


def preprocess_data(X):
    missing_values = X.isnull().sum().sum()
    if missing_values > 0:
        print(f"Warning: Found {missing_values} missing values. Consider imputation.")
    else:
        print("No missing values found.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X_scaled_df


def plot_dist(y):
    plt.figure(figsize=(10, 6), dpi=300)

    ax = sns.countplot(x=y, order=y.value_counts().index, palette="viridis")

    plt.title('Distribution of Cancer Types (Classes)', fontsize=16)
    plt.xlabel('Cancer Type', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=0)

    ax.set_axisbelow(True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    sns.despine(top=True, right=True)

    plt.tight_layout()
    plt.savefig('eda_plot_1_class_distribution.png')
    plt.show()


def plot_pca_results(X, y):
    pastel_colors = list(cm.Set2.colors) + list(cm.Set3.colors)

    color_pca = pastel_colors[12]     # PCA bar color
    color_cum = pastel_colors[11]     # Cumulative variance line color

    pca = PCA(n_components=N_PCA_COMPONENTS)
    pca.fit(X)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(12, 7))

    # Bars for individual explained variance
    plt.bar(
        range(1, N_PCA_COMPONENTS + 1),
        explained_variance,
        alpha=0.7,
        align='center',
        label='Individual explained variance',
        color=color_pca,
    )

    # Step line for cumulative variance
    plt.step(
        range(1, N_PCA_COMPONENTS + 1),
        cumulative_variance,
        where='mid',
        label='Cumulative explained variance',
        color=color_cum,
        linewidth=2.2,
    )

    # Titles and labels
    plt.title('PCA Explained Variance (Scree Plot)', fontsize=16)
    plt.xlabel('Principal Component', fontsize=12)
    plt.ylabel('Explained Variance Ratio', fontsize=12)
    plt.xticks(range(1, N_PCA_COMPONENTS + 1))
    plt.ylim(0, 1)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('eda_plot_2_pca_variance.png', dpi=300)


    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X)

    pc1_var = pca_2d.explained_variance_ratio_[0] * 100
    pc2_var = pca_2d.explained_variance_ratio_[1] * 100

    pca_df = pd.DataFrame(data=X_pca_2d, columns=['PC1', 'PC2'], index=X.index)
    pca_df['Cancer Type'] = y

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='Cancer Type',
        palette=sns.color_palette("viridis", n_colors=y.nunique()),
        data=pca_df,
        legend="full",
        alpha=0.9
    )

    plt.title('PCA of Cancer Types (PC1 vs PC2)', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pc1_var:.2f}% Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pc2_var:.2f}% Variance)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(linestyle='--', alpha=0.5)
    plt.savefig('eda_plot_3_pca_2d_plot.png')


def plot_pca_3d(X, y):
    pca = PCA(n_components=3)
    pcs = pca.fit_transform(X)

    df_plot = pd.DataFrame({
        'PC1': pcs[:, 0],
        'PC2': pcs[:, 1],
        'PC3': pcs[:, 2],
        'Class': y
    })

    fig = px.scatter_3d(
        df_plot,
        x='PC1', y='PC2', z='PC3',
        color='Class',
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="3D PCA of Gene Expression Data",
        opacity=1
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
            zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)",
        ),
        width=1600,
        height=1000,
        autosize=True,
        title_font=dict(size=20),
        legend=dict(title="Class", itemsizing="constant")
    )

    fig.show()


def perform_eda(X, y):
    plot_dist(y)
    plot_pca_results(X, y)
    #plot_pca_3d(X, y)


def main():
    X_features, y_labels = load_data(DATA_FILE, LABEL_FILE)

    if X_features is not None and y_labels is not None:
        X_scaled = preprocess_data(X_features)
        perform_eda(X_scaled, y_labels)


if __name__ == "__main__":
    main()
