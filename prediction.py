from typing import Tuple, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from xgboost import XGBClassifier


def log_reg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 2000,
    class_weight: Optional[dict] = None,
    random_state: int = 42,
):
    """
    Multinomial logistic regression for 6-class labels (lbfgs uses multinomial by default).
    Returns a fitted Pipeline(StandardScaler -> LogisticRegression).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])
    model.fit(X_train, y_train)
    return model


def knn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_neighbors: int = 7,
    weights: str = "distance",
    metric: str = "minkowski",
    p: int = 2,
):
    """
    K-Nearest Neighbors classifier on PCs.
    Returns a fitted Pipeline(StandardScaler -> KNN).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p
        ))
    ])
    model.fit(X_train, y_train)
    return model


def simple_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layer_sizes: Tuple[int, ...] = (64, 32),
    activation: str = "relu",
    alpha: float = 1e-4,
    max_iter: int = 500,
    random_state: int = 42,
    early_stopping: bool = True
):
    """
    Simple feed-forward neural net using sklearn's MLPClassifier.
    Returns a fitted Pipeline(StandardScaler -> MLPClassifier).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            learning_rate_init=0.001,
            learning_rate='adaptive',
            early_stopping=early_stopping
        ))
    ])
    model.fit(X_train, y_train)
    return model


def xgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42
):
    """
    XGBoost classifier — strong on tabular/omics data.
    Note: no scaler needed for trees.
    """
    clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf


def svm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    class_weight: Optional[dict] = None,
    random_state: int = 42,
    probability: bool = True,
):
    """
    SVM classifier with optional probability estimates.
    Returns a fitted Pipeline(StandardScaler -> SVC).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=class_weight,
            random_state=random_state,
            probability=probability
        ))
    ])
    model.fit(X_train, y_train)
    return model


def get_results(model, X_test: np.ndarray, y_test: np.ndarray, return_proba: bool = False,
                display_labels: np.ndarray | None = None, show_plot: bool = True):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    #print("\nConfusion Matrix:")
    #print(cm)

    if show_plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(cmap="Blues", values_format=".0f")
        plt.title(f"Confusion Matrix (Accuracy = {acc:.4f})")
        plt.show()

    if return_proba and hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        return y_pred, acc, cm, y_proba

    return y_pred, acc, cm
