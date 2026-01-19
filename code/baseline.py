from typing import Optional, List, Dict, Callable, Any
from data import load_data, DATA_FILE, LABEL_FILE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
from prediction import log_reg, knn_model, simple_nn, xgb_model, svm_model, get_results

# Evaluate N = 1..50 randomly
TOP_N_LIST = [i for i in range(1, 51)]


def nested_cv_random_then_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 5,   # unused, but kept for compatibility
    outdir: str = "nested_cv_outputs",
    classifier_fn: Callable[..., Any] = log_reg,
    classifier_kwargs: Optional[dict] = None,
    base_random_state: int = 42,
    model_name: str = "logreg",
) -> Dict[int, float]:

    print(f"{model_name} starting")

    os.makedirs(outdir, exist_ok=True)

    if classifier_kwargs is None:
        classifier_kwargs = {}

    n_features = X.shape[1]
    feature_names = X.columns.to_numpy()

    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=base_random_state
    )

    accs_per_n: Dict[int, List[float]] = {n: [] for n in TOP_N_LIST}

    for outer_fold_idx, (outer_tr_idx, outer_te_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_outer_tr, X_outer_te = X.iloc[outer_tr_idx], X.iloc[outer_te_idx]
        y_outer_tr, y_outer_te = y[outer_tr_idx], y[outer_te_idx]

        # RNG seeded per fold for reproducibility
        fold_seed = base_random_state + 1000 * outer_fold_idx

        for n in TOP_N_LIST:
            n_use = min(n, n_features)

            rng = np.random.RandomState(fold_seed + n_use)
            chosen_idx = rng.choice(n_features, size=n_use, replace=False)
            chosen_features = feature_names[chosen_idx]

            X_tr_fs = X_outer_tr[chosen_features].values
            X_te_fs = X_outer_te[chosen_features].values

            clf = classifier_fn(
                X_tr_fs,
                y_outer_tr,
                **classifier_kwargs
            )

            _, acc, _ = get_results(
                clf,
                X_te_fs,
                y_outer_te,
                return_proba=False,
                show_plot=False
            )

            accs_per_n[n].append(acc)

    # Compute statistics
    mean_accs = {n: float(np.mean(v)) for n, v in accs_per_n.items()}
    std_accs = {n: float(np.std(v)) for n, v in accs_per_n.items()}

    # Save CSV
    results_df = pd.DataFrame({
        "N": TOP_N_LIST,
        "mean_acc": [mean_accs[n] for n in TOP_N_LIST],
        "std": [std_accs[n] for n in TOP_N_LIST],
    })

    csv_path = os.path.join(outdir, f"{model_name}_nested_cv_results_random.csv")
    results_df.to_csv(csv_path, index=False)

    print(f"{model_name} done")

    return mean_accs, std_accs


def main():
    X_features, y_labels = load_data(DATA_FILE, LABEL_FILE)

    le = LabelEncoder()
    y_all = le.fit_transform(np.ravel(y_labels.values))

    # Logistic Regression
    nested_cv_random_then_classifier(
        X=X_features,
        y=y_all,
        outdir="nested_cv_logreg",
        classifier_fn=log_reg,
        model_name="logreg",
        base_random_state=42
    )
    #
    # # SVM
    # nested_cv_random_then_classifier(
    #     X=X_features,
    #     y=y_all,
    #     outdir="nested_cv_svm",
    #     classifier_fn=svm_model,
    #     model_name="svm",
    #     base_random_state=42
    # )
    #
    # # KNN
    # nested_cv_random_then_classifier(
    #     X=X_features,
    #     y=y_all,
    #     outdir="nested_cv_knn",
    #     classifier_fn=knn_model,
    #     model_name="knn",
    #     base_random_state=42
    # )
    #
    # # Simple NN
    # nested_cv_random_then_classifier(
    #     X=X_features,
    #     y=y_all,
    #     outdir="nested_cv_nn",
    #     classifier_fn=simple_nn,
    #     model_name="nn",
    #     base_random_state=42
    # )

    # # XGBoost
    # nested_cv_random_then_classifier(
    #     X=X_features,
    #     y=y_all,
    #     outdir="nested_cv_xgb",
    #     classifier_fn=xgb_model,
    #     model_name="xgb",
    #     base_random_state=42
    # )


if __name__ == "__main__":
    main()
