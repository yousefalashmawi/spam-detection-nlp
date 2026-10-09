"""Train the final spam classifiers and save them to models/.

Experiment E4 shows that one model trained on SMS + e-mail together loses
precision on SMS (short, informal texts look unlike corporate e-mail ham). The
deployed system therefore keeps one specialised model per channel:

    models/sms_classifier.joblib     trained on the SMS Spam Collection
    models/email_classifier.joblib   trained on Enron-Spam

For each channel the preprocessing + algorithm are chosen by stratified k-fold
cross-validation on the TRAINING split only; the chosen pipeline is then refit
on the whole training split and scored once on the untouched test split.
Metrics go to results/final_model.json.

Usage:  python src/train.py            (grid search, ~5-10 min)
        python src/train.py --fast     (skip the search, use linear SVM + lemma)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_enron, load_sms  # noqa: E402
from experiments import cached_preprocess, split  # noqa: E402
from models import build_pipeline, evaluate, spam_scores  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
CHANNELS = {"sms": load_sms, "email": load_enron}
CACHE_NAME = {"sms": "sms", "email": "enron"}  # share cache with experiments.py
GRID_MODES = ("clean", "stopwords", "stem", "lemma")
GRID_MODELS = ("naive_bayes", "linear_svm", "logistic_regression")


def select_config(channel, tr, te, fast):
    if fast:
        return {"mode": "lemma", "model": "linear_svm", "cv_f1": None}, []
    folds = 5 if channel == "sms" else 3
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    y = tr["label"].to_numpy()
    rows = []
    full = __import__("pandas").concat([tr, te])
    for mode in GRID_MODES:
        X = cached_preprocess(full, CACHE_NAME[channel], mode)[: len(tr)]
        for model in GRID_MODELS:
            pipe = build_pipeline(model, mode, "tfidf", include_preprocessor=False)
            f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1").mean()
            rows.append({"mode": mode, "model": model, "cv_f1": round(float(f1), 4)})
            print(f"  {channel:5s} {mode:9s} {model:20s} cv F1={f1:.4f}")
    best = max(rows, key=lambda r: r["cv_f1"])
    return best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    report = {}

    for channel, loader in CHANNELS.items():
        tr, te = split(loader())
        best, grid = select_config(channel, tr, te, args.fast)
        print(f"{channel}: selected {best}")

        pipe = build_pipeline(best["model"], best["mode"], "tfidf")
        t0 = time.perf_counter()
        pipe.fit(tr["text"].tolist(), tr["label"].to_numpy())
        secs = round(time.perf_counter() - t0, 1)

        X, y = te["text"].tolist(), te["label"].to_numpy()
        t0 = time.perf_counter()
        pred = pipe.predict(X)
        ms_per_msg = round(1000 * (time.perf_counter() - t0) / len(X), 3)
        metrics = evaluate(y, pred, spam_scores(pipe, X))

        joblib.dump(pipe, MODELS / f"{channel}_classifier.joblib", compress=3)
        report[channel] = {
            "selected": best, "cv_grid": grid,
            "train_messages": len(tr), "test_messages": len(te),
            "train_seconds": secs, "predict_ms_per_message": ms_per_msg,
            "test": metrics,
        }
        print(f"{channel}: test {metrics}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "final_model.json").write_text(json.dumps(report, indent=2))
    plot_final(report)
    print("saved models ->", MODELS)


def plot_final(report: dict):
    import pandas as pd
    from experiments import FIGURES, LABEL, plot_confusion

    FIGURES.mkdir(exist_ok=True)
    for channel, r in report.items():
        sel = r["selected"]
        name = {'sms': 'SMS', 'email': 'E-mail'}[channel]
        title = (f"{name}: {LABEL[sel['model']]}, {sel['mode']} text\n"
                 f"test set, n = {r['test_messages']:,}")
        plot_confusion(pd.Series(r["test"]), title, f"confusion_final_{channel}.png")


if __name__ == "__main__":
    np.random.seed(42)
    main()
