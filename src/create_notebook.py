"""Generate notebooks/Spam_Detection_Experiments.ipynb.

The notebook calls the project modules in src/ so the code shown in the
appendix is exactly the code that produced the reported numbers.
Execute it with:  python src/execute_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))  # noqa: E731
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))  # noqa: E731

md("""
# Spam detection with NLP - executed experiment notebook

DLBAIPNLP01 - Project: NLP, Task 2 (Text Classification for Spam Detection).
The notebook runs every step of the system: data collection, preprocessing,
encoding, training, evaluation from small to large data sets, cross-corpus
generalisation, final model selection and classification of new messages.
All functions are imported from the `src/` package of the repository.
""")

code("""
import sys, platform, time, json, warnings
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np, pandas as pd, sklearn, nltk, matplotlib
from IPython.display import Image, display
pd.set_option("display.width", 120); pd.set_option("display.max_colwidth", 70)

import data, preprocess, models, experiments
print("Python", platform.python_version(), "| scikit-learn", sklearn.__version__,
      "| NLTK", nltk.__version__, "| pandas", pd.__version__, "| NumPy", np.__version__)
print("Machine:", platform.machine(), platform.system())
""")

md("""
## 1. Data collection

Two public labelled corpora are downloaded automatically: the SMS Spam
Collection (small scenario) and Enron-Spam (large scenario). Exact duplicate
texts are removed before splitting so that no message can appear in both the
training and the test set.
""")

code("""
sms_raw, enron_raw = data.load_sms(dedupe=False), data.load_enron(dedupe=False)
sms, enron = data.load_sms(), data.load_enron()
stats = pd.DataFrame({"SMS": data.describe(sms), "Enron e-mail": data.describe(enron)}).T
stats.insert(0, "raw messages", [len(sms_raw), len(enron_raw)])
stats.insert(1, "duplicates removed", [len(sms_raw) - len(sms), len(enron_raw) - len(enron)])
print(stats.to_string())
""")

code("""
print("Example SMS messages:")
print(sms.groupby("label").head(3)[["label", "text"]].to_string(index=False))
""")

code("""
corpora = {"sms": experiments.split(sms), "enron": experiments.split(enron)}
for name, (tr, te) in corpora.items():
    print(f"{name:6s} train={len(tr):6d} (spam {tr.label.mean():.3f})  "
          f"test={len(te):5d} (spam {te.label.mean():.3f})")
""")

md("""
## 2. Preprocessing

Five cumulative preprocessing levels: `raw` (lower-case only), `clean`
(placeholder tokens for URLs, e-mail addresses, phone numbers, currency and
numbers; punctuation removed), `stopwords` (+ NLTK English stop-word removal),
`stem` (+ Porter stemming) and `lemma` (+ WordNet lemmatising).
""")

code("""
example = sms[sms.label == 1].text.iloc[0]
print("Original :", example)
for mode in preprocess.MODES:
    print(f"{mode:9s}:", preprocess.preprocess(example, mode))
""")

code("""
# Preprocess every message once per level (cached for the experiments below).
import shutil
shutil.rmtree(experiments.CACHE, ignore_errors=True)
for mode in preprocess.MODES:
    t0 = time.perf_counter()
    for name, (tr, te) in corpora.items():
        experiments.cached_preprocess(pd.concat([tr, te]), name, mode)
    print(f"{mode:9s} preprocessed both corpora in {time.perf_counter() - t0:6.1f} s")
""")

code("""
# Encoding: unigram + bigram TF-IDF (or Bag-of-Words counts), min_df = 2.
X_tr, y_tr, X_te, y_te = experiments.prepared(corpora, "lemma")["sms"]
vec = models.make_vectorizer("tfidf").fit(X_tr)
M = vec.transform(X_tr)
print("SMS vocabulary size:", len(vec.vocabulary_), "| training matrix:", M.shape,
      f"| non-zero share: {M.nnz / (M.shape[0] * M.shape[1]):.5f}")
""")

md("""
## 3. Experiment E1 - effect of each preprocessing step

Every preprocessing level x encoding (BoW, TF-IDF) x classifier (Naive Bayes,
linear SVM, logistic regression) is trained on the training split and scored
on the fixed test split of each corpus.
""")

code("""
classic = ["naive_bayes", "linear_svm", "logistic_regression"]
t0 = time.perf_counter()
e1 = experiments.e1_preprocessing(corpora, classic, quick=False)
e1.to_csv(experiments.RESULTS / "e1_preprocessing.csv", index=False)
experiments.plot_preprocessing(e1)
print(f"E1 finished in {time.perf_counter() - t0:.0f} s")
""")

code("""
print(e1.pivot_table(index=["corpus", "vectorizer", "preprocessing"], columns="model",
                     values="f1").round(4).to_string())
print()
print("Mean F1 by encoding:")
print(e1.groupby(["corpus", "vectorizer"]).f1.mean().round(4).to_string())
display(Image(str(experiments.FIGURES / "preprocessing_ablation.png"), width=800))
""")

md("""
## 4. Experiment E2 - from small to large training sets

Models are trained on stratified random subsets of the training split (five
different subsets per size) and always scored on the same full test split.
""")

code("""
t0 = time.perf_counter()
e2_runs, e2 = experiments.e2_learning_curve(corpora, classic, quick=False)
e2_runs.to_csv(experiments.RESULTS / "e2_learning_curve_runs.csv", index=False)
e2.to_csv(experiments.RESULTS / "e2_learning_curve.csv", index=False)
experiments.plot_learning_curves(e2)
print(f"E2 finished in {time.perf_counter() - t0:.0f} s")
""")

code("""
cols = ["corpus", "train_size", "model", "accuracy_mean", "f1_mean", "f1_std",
        "false_positive_rate_mean"]
print(e2[cols].to_string(index=False))
display(Image(str(experiments.FIGURES / "learning_curves.png"), width=800))
""")

md("""
## 5. Experiment E3 - model comparison including a neural network

Lemmatised text and TF-IDF features; 5-fold stratified cross-validation on the
training split plus one evaluation on the held-out test split. `mlp` is a
feed-forward neural network with one hidden layer of 64 units.
""")

code("""
t0 = time.perf_counter()
e3 = experiments.e3_models(corpora, classic + ["mlp"])
e3.to_csv(experiments.RESULTS / "e3_models.csv", index=False)
print(f"E3 finished in {time.perf_counter() - t0:.0f} s")
print(e3[["corpus", "model", "cv_f1_mean", "cv_f1_std", "test_accuracy", "test_precision",
          "test_recall", "test_f1", "test_roc_auc", "test_train_seconds"]].to_string(index=False))
""")

md("""
## 6. Experiment E4 - generalisation across message types

The best classifier from E3 (by mean cross-validated F1) is trained on SMS
only, e-mail only, and both corpora combined, and tested on both test sets.
""")

code("""
best = e3.groupby("model")["cv_f1_mean"].mean().idxmax()
print("Best model by mean CV F1:", best)
e4 = experiments.e4_generalisation(corpora, best)
e4.to_csv(experiments.RESULTS / "e4_generalisation.csv", index=False)
print(e4[["trained_on", "tested_on", "accuracy", "precision", "recall", "f1",
          "false_positive_rate"]].to_string(index=False))
""")

md("""
## 7. Final system: model selection and held-out evaluation

For each channel (SMS, e-mail) the preprocessing level and classifier are
selected by cross-validation on the training split only. The selected pipeline
is refitted on the full training split, evaluated once on the test split, and
saved to `models/`.
""")

code("""
import train
sys.argv = ["train.py"]
t0 = time.perf_counter()
train.main()
print(f"Final training finished in {time.perf_counter() - t0:.0f} s")
""")

code("""
final = json.loads((ROOT / "results" / "final_model.json").read_text())
rows = []
for ch, r in final.items():
    rows.append({"channel": ch, "model": r["selected"]["model"], "preprocessing": r["selected"]["mode"],
                 "cv_f1": r["selected"]["cv_f1"], **{k: r["test"][k] for k in
                 ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate",
                  "tp", "fp", "tn", "fn"]},
                 "predict_ms": r["predict_ms_per_message"]})
print(pd.DataFrame(rows).T.to_string(header=False))
for ch in final:
    display(Image(str(experiments.FIGURES / f"confusion_final_{ch}.png"), width=300))
""")

md("""
## 8. Error analysis

Misclassified test messages of the final SMS model, and the n-grams that the
final e-mail model weights most strongly towards each class.
""")

code("""
import joblib
sms_model = joblib.load(ROOT / "models" / "sms_classifier.joblib")
_, sms_te = corpora["sms"]
pred = sms_model.predict(sms_te.text.tolist())
err = sms_te.assign(pred=pred)[pred != sms_te.label.to_numpy()]
# Sexually explicit spam texts are counted but not printed.
EXPLICIT = ("sex", "xxx", "babe", "bitch", "shit", "nasty", "rude", "cum", "hot")
def show(texts):
    clean = [t for t in texts if not any(w in t.lower() for w in EXPLICIT)]
    for t in clean: print("  -", t[:110])
    if len(clean) < len(texts): print(f"  ({len(texts) - len(clean)} explicit message(s) not printed)")
print(f"False positives (ham flagged as spam): {(err.label == 0).sum()}")
show(err[err.label == 0].text.tolist())
print(f"False negatives (spam missed): {(err.label == 1).sum()}")
show(err[err.label == 1].text.tolist())
""")

code("""
email_model = joblib.load(ROOT / "models" / "email_classifier.joblib")
names = email_model.named_steps["vectorize"].get_feature_names_out()
coef = email_model.named_steps["classify"].coef_.ravel()
order = np.argsort(coef)
print("Strongest spam indicators:", ", ".join(names[order[::-1][:15]]))
print("Strongest ham indicators :", ", ".join(names[order[:15]]))

nb_ = sms_model.named_steps["classify"]; v = sms_model.named_steps["vectorize"].get_feature_names_out()
ratio = nb_.feature_log_prob_[1] - nb_.feature_log_prob_[0]
print("SMS model, highest spam/ham log-probability ratio:", ", ".join(v[np.argsort(ratio)[::-1][:15]]))
""")

md("""
## 9. Classifying new messages

The saved pipelines accept raw text. Short texts are routed to the SMS model,
long or multi-line texts to the e-mail model.
""")

code("""
from predict import classify
new_messages = [
    "URGENT! Your mobile number has won a 2000 pound prize. Call 09061743810 to claim now",
    "hey, running 10 min late, can you order me a coffee?",
    "Are you free for a call about the budget tomorrow at 3?",
    "FREE entry! Win cash and a holiday, txt WIN to 80086 today",
    "Subject: lowest prices on meds\\nBuy cheap medication online without prescription. "
    "Limited offer, click here to order now and save 80 percent.",
    "Subject: schedule for thursday\\nHi Sally, please find attached the revised schedule "
    "for the pipeline meeting. Let me know if the volumes look right. Thanks",
]
for r in classify(new_messages):
    print(f"{r['label'].upper():4s} [{r['channel']:5s}] score={r['score']:+7.3f}  {r['message'][:70]!r}")
""")

md("""
## 10. Automated tests
""")

code("""
import subprocess
out = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")],
                     capture_output=True, text=True, cwd=ROOT)
print(out.stdout[-600:])
""")

out = ROOT / "notebooks" / "Spam_Detection_Experiments.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
