"""Run every experiment reported in the project and write results + figures.

Usage:  python src/experiments.py            (full run, ~5-10 min on a laptop)
        python src/experiments.py --quick    (smaller grid, for a smoke test)

Experiments
  E0  dataset statistics
  E1  preprocessing ablation  (raw / clean / stopwords / stem / lemma  x  BoW / TF-IDF  x  models)
  E2  learning curves         (train on progressively larger subsets, small -> full, both corpora)
  E3  model comparison        (5-fold CV on training data + held-out test, incl. a neural network)
  E4  generalisation          (cross-corpus: SMS -> e-mail, e-mail -> SMS, combined)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import describe, load_enron, load_sms  # noqa: E402
from models import build_pipeline, evaluate, spam_scores  # noqa: E402
from preprocess import MODES, preprocess  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CACHE = RESULTS / ".cache"
SEED = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------- helpers
def split(df: pd.DataFrame):
    """One fixed, stratified 80/20 split per corpus. The test part is never
    used for training or model selection."""
    return train_test_split(df, test_size=TEST_SIZE, stratify=df["label"],
                            random_state=SEED)


def cached_preprocess(df: pd.DataFrame, name: str, mode: str) -> list[str]:
    """Preprocessing the e-mails is the slowest step, so cache it per mode."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}_{mode}.joblib"
    if path.exists():
        cached = joblib.load(path)
        if len(cached) == len(df):
            return cached
    out = [preprocess(t, mode) for t in df["text"]]
    joblib.dump(out, path)
    return out


def fit_eval(model, mode, vec, X_tr, y_tr, X_te, y_te, seed=SEED):
    pipe = build_pipeline(model, mode, vec, seed, include_preprocessor=False)
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    train_s = time.perf_counter() - t0
    pred = pipe.predict(X_te)
    m = evaluate(y_te, pred, spam_scores(pipe, X_te))
    m["train_seconds"] = round(train_s, 2)
    return m


def prepared(corpora, mode):
    """Preprocessed train/test text for every corpus under one mode."""
    out = {}
    for name, (tr, te) in corpora.items():
        full = pd.concat([tr, te])
        texts = cached_preprocess(full, name, mode)
        out[name] = (texts[: len(tr)], tr["label"].to_numpy(),
                     texts[len(tr):], te["label"].to_numpy())
    return out


# ------------------------------------------------------------ experiments
def e1_preprocessing(corpora, models, quick):
    rows = []
    modes = ("raw", "stopwords", "lemma") if quick else MODES
    for mode in modes:
        data = prepared(corpora, mode)
        for name, (X_tr, y_tr, X_te, y_te) in data.items():
            for vec in ("bow", "tfidf"):
                for model in models:
                    m = fit_eval(model, mode, vec, X_tr, y_tr, X_te, y_te)
                    rows.append({"corpus": name, "preprocessing": mode,
                                 "vectorizer": vec, "model": model, **m})
                    print(f"  E1 {name:5s} {mode:9s} {vec:5s} {model:20s} "
                          f"F1={m['f1']:.4f}")
    return pd.DataFrame(rows)


def e2_learning_curve(corpora, models, quick):
    rows = []
    sizes = {"sms": [100, 250, 500, 1000, 2000, None],
             "enron": [100, 500, 1000, 2500, 5000, 10000, 20000, None]}
    repeats = 2 if quick else 5
    data = prepared(corpora, "lemma")
    for name, (X_tr, y_tr, X_te, y_te) in data.items():
        X_tr = np.array(X_tr, dtype=object)
        for size in sizes[name]:
            n = len(X_tr) if size is None or size >= len(X_tr) else size
            reps = 1 if n == len(X_tr) else repeats
            for rep in range(reps):
                if n == len(X_tr):
                    idx = np.arange(n)
                else:
                    idx, _ = train_test_split(np.arange(len(X_tr)), train_size=n,
                                              stratify=y_tr, random_state=SEED + rep)
                for model in models:
                    m = fit_eval(model, "lemma", "tfidf", list(X_tr[idx]),
                                 y_tr[idx], X_te, y_te, seed=SEED + rep)
                    rows.append({"corpus": name, "train_size": n, "repeat": rep,
                                 "model": model, **m})
            print(f"  E2 {name:5s} n={n:6d} done")
    raw = pd.DataFrame(rows)
    agg = (raw.groupby(["corpus", "train_size", "model"])
              [["accuracy", "precision", "recall", "f1", "false_positive_rate"]]
              .agg(["mean", "std"]).round(4))
    agg.columns = ["_".join(c) for c in agg.columns]
    return raw, agg.reset_index()


def e3_models(corpora, models):
    rows = []
    data = prepared(corpora, "lemma")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, (X_tr, y_tr, X_te, y_te) in data.items():
        for model in models:
            pipe = build_pipeline(model, "lemma", "tfidf", SEED,
                                  include_preprocessor=False)
            cvr = cross_validate(pipe, X_tr, y_tr, cv=cv, n_jobs=1,
                                 scoring=["accuracy", "f1", "precision", "recall"])
            test = fit_eval(model, "lemma", "tfidf", X_tr, y_tr, X_te, y_te)
            rows.append({
                "corpus": name, "model": model,
                "cv_accuracy_mean": round(cvr["test_accuracy"].mean(), 4),
                "cv_f1_mean": round(cvr["test_f1"].mean(), 4),
                "cv_f1_std": round(cvr["test_f1"].std(), 4),
                **{f"test_{k}": v for k, v in test.items()},
            })
            print(f"  E3 {name:5s} {model:20s} cvF1={rows[-1]['cv_f1_mean']:.4f} "
                  f"testF1={test['f1']:.4f}")
    return pd.DataFrame(rows)


def e4_generalisation(corpora, model):
    data = prepared(corpora, "lemma")
    s_tr, s_ytr, s_te, s_yte = data["sms"]
    e_tr, e_ytr, e_te, e_yte = data["enron"]
    combos = {
        "sms": (s_tr, s_ytr),
        "enron": (e_tr, e_ytr),
        "combined": (s_tr + e_tr, np.concatenate([s_ytr, e_ytr])),
    }
    tests = {"sms": (s_te, s_yte), "enron": (e_te, e_yte)}
    rows = []
    for train_name, (X_tr, y_tr) in combos.items():
        for test_name, (X_te, y_te) in tests.items():
            m = fit_eval(model, "lemma", "tfidf", X_tr, y_tr, X_te, y_te)
            rows.append({"trained_on": train_name, "tested_on": test_name,
                         "model": model, **m})
            print(f"  E4 train={train_name:8s} test={test_name:5s} F1={m['f1']:.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
# Monochrome styling: models are distinguished by line style, marker and hatch.
STYLE = {
    "naive_bayes": dict(color="0.55", linestyle=":", marker="^", hatch="..", face="0.85"),
    "linear_svm": dict(color="black", linestyle="-", marker="o", hatch="", face="0.15"),
    "logistic_regression": dict(color="0.3", linestyle="--", marker="s", hatch="//", face="0.55"),
    "mlp": dict(color="0.45", linestyle="-.", marker="D", hatch="xx", face="0.7"),
}
LABEL = {"naive_bayes": "Naive Bayes", "linear_svm": "Linear SVM",
         "logistic_regression": "Logistic regression", "mlp": "MLP"}
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "axes.edgecolor": "black", "text.color": "black",
                     "axes.labelcolor": "black", "xtick.color": "black",
                     "ytick.color": "black"})


def plot_learning_curves(agg: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, corpus in zip(axes, ["sms", "enron"]):
        sub = agg[agg["corpus"] == corpus]
        for model, g in sub.groupby("model"):
            g = g.sort_values("train_size")
            st = STYLE[model]
            ax.errorbar(g["train_size"], g["f1_mean"], yerr=g["f1_std"].fillna(0),
                        color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                        markersize=5, capsize=3, linewidth=1.3, label=LABEL[model])
        ax.set_xscale("log")
        ax.set_title("SMS Spam Collection" if corpus == "sms" else "Enron-Spam e-mails")
        ax.set_xlabel("Training messages (log scale)")
        ax.grid(alpha=0.3, color="0.6")
    axes[0].set_ylabel("Test F1 (spam class)")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "learning_curves.png", dpi=200)
    plt.close(fig)


def plot_preprocessing(e1: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, corpus in zip(axes, ["sms", "enron"]):
        sub = e1[(e1["corpus"] == corpus) & (e1["vectorizer"] == "tfidf")]
        piv = sub.pivot(index="preprocessing", columns="model", values="f1")
        piv = piv.reindex([m for m in MODES if m in piv.index])
        x = np.arange(len(piv.index))
        width = 0.8 / len(piv.columns)
        for k, model in enumerate(piv.columns):
            st = STYLE[model]
            ax.bar(x + (k - (len(piv.columns) - 1) / 2) * width, piv[model], width,
                   color=st["face"], edgecolor="black", hatch=st["hatch"],
                   linewidth=0.6, label=LABEL[model])
        ax.set_xticks(x, piv.index)
        lo = max(0.0, piv.min().min() - 0.02)
        ax.set_ylim(lo, min(1.0, piv.max().max() + 0.01))
        ax.set_title(("SMS" if corpus == "sms" else "Enron e-mail") + " (TF-IDF)")
        ax.set_xlabel("Preprocessing level")
        ax.set_ylabel("Test F1 (spam class)")
        ax.grid(axis="y", alpha=0.3, color="0.6")
    axes[0].legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "preprocessing_ablation.png", dpi=200)
    plt.close(fig)


def plot_confusion(row: pd.Series, title: str, fname: str):
    cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    shade = np.log1p(cm) / np.log1p(cm.max())
    ax.imshow(shade, cmap="Greys", vmin=0, vmax=1.15)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if shade[i, j] > 0.6 else "black", fontsize=13)
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=200)
    plt.close(fig)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    sms, enron = load_sms(), load_enron()
    stats = {"sms": describe(sms), "enron": describe(enron),
             "sms_raw_messages": int(len(load_sms(dedupe=False))),
             "enron_raw_messages": int(len(load_enron(dedupe=False)))}
    print("E0", json.dumps(stats))
    corpora = {"sms": split(sms), "enron": split(enron)}
    for name, (tr, te) in corpora.items():
        stats[name]["train"], stats[name]["test"] = len(tr), len(te)

    classic = ["naive_bayes", "linear_svm", "logistic_regression"]

    print("E1 preprocessing ablation")
    e1 = e1_preprocessing(corpora, classic, args.quick)
    e1.to_csv(RESULTS / "e1_preprocessing.csv", index=False)
    plot_preprocessing(e1)

    print("E2 learning curves")
    e2_raw, e2 = e2_learning_curve(corpora, classic, args.quick)
    e2_raw.to_csv(RESULTS / "e2_learning_curve_runs.csv", index=False)
    e2.to_csv(RESULTS / "e2_learning_curve.csv", index=False)
    plot_learning_curves(e2)

    print("E3 model comparison")
    e3 = e3_models(corpora, classic + ([] if args.quick else ["mlp"]))
    e3.to_csv(RESULTS / "e3_models.csv", index=False)
    best = e3.groupby("model")["cv_f1_mean"].mean().idxmax()
    for corpus in ("sms", "enron"):
        r = e3[(e3["corpus"] == corpus) & (e3["model"] == best)].iloc[0]
        r = r.rename(lambda c: c.replace("test_", ""))
        plot_confusion(r, f"{best.replace('_', ' ')} - {corpus.upper()} test set",
                       f"confusion_e3_{corpus}.png")

    print(f"E4 generalisation (model = {best})")
    e4 = e4_generalisation(corpora, best)
    e4.to_csv(RESULTS / "e4_generalisation.csv", index=False)

    summary = {"datasets": stats, "best_model_by_cv_f1": best,
               "quick_run": args.quick}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done ->", RESULTS)


if __name__ == "__main__":
    main()
