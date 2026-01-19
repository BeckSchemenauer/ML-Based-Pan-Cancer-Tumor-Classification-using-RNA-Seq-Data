from typing import Optional, List, Dict
from data import load_data, DATA_FILE, LABEL_FILE
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import os
from typing import Callable, Any
from prediction import log_reg, knn_model, simple_nn, xgb_model, svm_model, get_results


TOP_N_LIST = [i for i in range(1, 50)]


def nested_cv_rf_importance_then_all_models(
    X: pd.DataFrame,
    y: np.ndarray,
    outer_splits: int = 5,
    inner_splits: int = 5,
    outdir_root: str = "nested_cv_sharedfs",
    rf_kwargs: Optional[dict] = None,
    base_random_state: int = 42,
):
    """
    Single nested CV with RF-based feature importance.
    Uses the SAME ranked features for all models and all N in TOP_N_LIST.
    Saves per-model CSVs in separate subdirectories under outdir_root.
    """

    # define models once
    models = {
        "logreg": log_reg,
        "svm": svm_model,
        "knn": knn_model,
        "nn": simple_nn,
        "xgb": xgb_model,
    }

    # create root output dir + per-model subdirs
    os.makedirs(outdir_root, exist_ok=True)
    model_outdirs = {}
    for model_name in models.keys():
        model_dir = os.path.join(outdir_root, f"nested_cv_{model_name}")
        os.makedirs(model_dir, exist_ok=True)
        model_outdirs[model_name] = model_dir

    if rf_kwargs is None:
        rf_kwargs = dict(
            n_estimators=500,
            max_features="sqrt",
            n_jobs=-1,
            class_weight="balanced_subsample"
        )

    outer_cv = StratifiedKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=base_random_state
    )

    # Collect accuracies per model per N across outer folds
    accs_per_model_per_n: Dict[str, Dict[int, List[float]]] = {
        m: {n: [] for n in TOP_N_LIST} for m in models.keys()
    }

    # global accumulator over all outer folds (each fi_sum is itself a sum over inner folds)
    total_importance = np.zeros(X.shape[1], dtype=float)

    for outer_fold_idx, (outer_tr_idx, outer_te_idx) in enumerate(outer_cv.split(X, y), start=1):
        print(f"starting outer fold {outer_fold_idx}")
        X_outer_tr, X_outer_te = X.iloc[outer_tr_idx], X.iloc[outer_te_idx]
        y_outer_tr, y_outer_te = y[outer_tr_idx], y[outer_te_idx]

        #  Inner CV: sum RF feature importances
        inner_cv = StratifiedKFold(
            n_splits=inner_splits,
            shuffle=True,
            random_state=base_random_state + outer_fold_idx
        )
        inner_importances = []

        for inner_fold_idx, (inner_tr_idx, inner_va_idx) in enumerate(inner_cv.split(X_outer_tr, y_outer_tr), start=1):
            X_inner_tr, X_inner_va = X_outer_tr.iloc[inner_tr_idx], X_outer_tr.iloc[inner_va_idx]
            y_inner_tr, y_inner_va = y_outer_tr[inner_tr_idx], y_outer_tr[inner_va_idx]

            rf = RandomForestClassifier(
                random_state=base_random_state + 100 * outer_fold_idx + inner_fold_idx,
                **rf_kwargs
            )
            rf.fit(X_inner_tr, y_inner_tr)
            inner_importances.append(rf.feature_importances_)

        # sum importances across the 5 inner folds for this outer fold
        inner_importances = np.vstack(inner_importances)
        fi_sum = inner_importances.sum(axis=0)
        fi_sum_series = pd.Series(fi_sum, index=X.columns, name=f"outer{outer_fold_idx}_inner_sum")

        # accumulate into global sum across outer folds
        total_importance += fi_sum

        # build ranked list from this outer fold's inner-summed importances
        ranked_features = fi_sum_series.sort_values(ascending=False).index.tolist()

        # ----- Evaluate each N with all classifiers on the same features -----
        for n in TOP_N_LIST:
            n_use = min(n, X.shape[1])
            topn_feats = ranked_features[:n_use]

            X_tr_fs = X_outer_tr[topn_feats].values
            X_te_fs = X_outer_te[topn_feats].values

            # loop over models
            for model_name, model_fn in models.items():
                clf = model_fn(X_tr_fs, y_outer_tr)

                _, acc, _ = get_results(
                    clf,
                    X_te_fs,
                    y_outer_te,
                    return_proba=False,
                    show_plot=False,
                )
                accs_per_model_per_n[model_name][n].append(acc)

        print(f"[Outer fold {outer_fold_idx}] done for all models.")

    # ----- Mean/std across outer folds, per model -----
    mean_std_per_model: Dict[str, Dict[str, Dict[int, float]]] = {}

    for model_name in models.keys():
        mean_accs = {
            n: float(np.mean(accs))
            for n, accs in accs_per_model_per_n[model_name].items()
        }
        std_accs = {
            n: float(np.std(accs))
            for n, accs in accs_per_model_per_n[model_name].items()
        }

        mean_std_per_model[model_name] = {
            "mean_accs": mean_accs,
            "std_accs": std_accs,
        }

        print(f"\n=== {model_name}: Mean accuracies across outer folds (± std) ===")
        for n in TOP_N_LIST:
            print(
                f"{model_name} | N={n:<5}  mean={mean_accs[n]:.4f}  std={std_accs[n]:.4f}"
            )

        # ----- Save CSV (N, mean_acc, std) for this model -----
        results_df = pd.DataFrame(
            {
                "N": TOP_N_LIST,
                "mean_acc": [mean_accs[n] for n in TOP_N_LIST],
                "std": [std_accs[n] for n in TOP_N_LIST],
            }
        )

        csv_path = os.path.join(
            model_outdirs[model_name], f"{model_name}_nested_cv_results.csv"
        )
        results_df.to_csv(csv_path, index=False)
        print(f"Saved {model_name} results to: {csv_path}")

    return mean_std_per_model


def main():
    X_features, y_labels = load_data(DATA_FILE, LABEL_FILE)

    # encode labels to integers
    le = LabelEncoder()
    y_all = le.fit_transform(np.ravel(y_labels.values))

    # run nested CV that does RF feature selection once
    results = nested_cv_rf_importance_then_all_models(
        X=X_features,
        y=y_all,
        outer_splits=5,
        inner_splits=5,
        outdir_root="nested_cv_sharedfs",
        base_random_state=42,
    )

    return results


if __name__ == "__main__":
    main()
