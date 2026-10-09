"""Model factory and evaluation helpers."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import TextPreprocessor

MODEL_NAMES = ("naive_bayes", "linear_svm", "logistic_regression", "mlp")
VECTORIZERS = ("bow", "tfidf")


def make_vectorizer(kind: str = "tfidf", max_features: int | None = 50_000):
    common = dict(ngram_range=(1, 2), min_df=2, max_features=max_features,
                  token_pattern=r"(?u)\b\w+\b")
    if kind == "bow":
        return CountVectorizer(**common)
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, **common)
    raise ValueError(f"vectorizer must be one of {VECTORIZERS}")


def make_classifier(name: str, seed: int = 42):
    if name == "naive_bayes":
        return MultinomialNB(alpha=0.1)
    if name == "linear_svm":
        return LinearSVC(C=1.0, class_weight="balanced", max_iter=5000,
                         random_state=seed)
    if name == "logistic_regression":
        return LogisticRegression(C=10.0, solver="liblinear", max_iter=2000,
                                  class_weight="balanced", random_state=seed)
    if name == "mlp":
        # Small feed-forward neural network (the "deep learning" baseline).
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-4,
                             early_stopping=True, max_iter=50,
                             random_state=seed)
    raise ValueError(f"model must be one of {MODEL_NAMES}")


def build_pipeline(model: str = "linear_svm", mode: str = "lemma",
                   vectorizer: str = "tfidf", seed: int = 42,
                   include_preprocessor: bool = True) -> Pipeline:
    """Raw text in, spam (1) / ham (0) out."""
    max_features = 20_000 if model == "mlp" else 50_000
    steps = []
    if include_preprocessor:
        steps.append(("preprocess", TextPreprocessor(mode)))
    steps += [
        ("vectorize", make_vectorizer(vectorizer, max_features)),
        ("classify", make_classifier(model, seed)),
    ]
    return Pipeline(steps)


def spam_scores(pipe, X):
    """Continuous spam score for ROC-AUC (probability or margin)."""
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return pipe.decision_function(X)


def evaluate(y_true, y_pred, scores=None) -> dict:
    """Metrics reported throughout the project. Spam is the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        # Share of legitimate messages wrongly flagged - the costly error.
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }
    if scores is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, scores)
    return {k: (round(float(v), 4) if isinstance(v, float) else v)
            for k, v in out.items()}
