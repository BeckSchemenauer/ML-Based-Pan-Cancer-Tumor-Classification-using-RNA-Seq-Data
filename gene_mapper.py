import pandas as pd


def map_feature_importances_to_genes(
        expression_file: str,
        importance_file: str,
        top_n: int = 50
):
    # Load the expression file (tab-separated)
    expr_df = pd.read_csv(expression_file, sep="\t", index_col=0)

    # Get the ordered list of genes from the file
    genes = expr_df.index.tolist()

    # Load the feature importance file
    fi_df = pd.read_csv(importance_file)

    # Handle possible header variations
    if "Unnamed: 0" in fi_df.columns:
        fi_df.rename(columns={"Unnamed: 0": "feature"}, inplace=True)
    if "importance_sum" in fi_df.columns:
        fi_df.rename(columns={"importance_sum": "importance"}, inplace=True)
    elif "total_importance_sum_outer5_inner5" in fi_df.columns:
        fi_df.rename(columns={"total_importance_sum_outer5_inner5": "importance"}, inplace=True)

    # Ensure consistent ordering: gene_0 → first index in expression file
    fi_df["gene_name"] = fi_df["feature"].apply(
        lambda f: genes[int(f.split("_")[1])] if f.startswith("gene_") else f
    )

    # Sort by importance (descending)
    fi_df = fi_df.sort_values(by="importance", ascending=False)

    # Print top N mappings
    print(f"\nTop {top_n} features mapped to gene IDs:")
    for _, row in fi_df.head(top_n).iterrows():
        print(f"{row['feature']:<10} → {row['gene_name']}  (importance={row['importance']:.6f})")

    print("raw gene names")
    for _, row in fi_df.head(top_n).iterrows():
        print(f"{row['gene_name']}")

    return fi_df


# Example usage:
mapped_df = map_feature_importances_to_genes("Data/unc.edu_PANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv", "nested_cv_outputs/feature_importances_sum_outer5_inner5.csv", top_n=50)
