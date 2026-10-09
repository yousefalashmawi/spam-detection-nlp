"""Build the project report PDF, the executed-notebook appendix and the combined
submission PDF from saved experiment outputs.

Usage:  python src/build_report.py --repo https://github.com/<user>/<repo>

Every number in the report is read from results/ (written by the executed
notebook), so report and code cannot drift apart.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import sys
import textwrap
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Flowable, Image, KeepTogether, PageBreak, Paragraph,
                                Preformatted, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data as D  # noqa: E402
import experiments as E  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default="https://github.com/<your-username>/spam-detection-nlp")
ap.add_argument("--date", default="9 October 2026")
ARGS = ap.parse_args()

AUTHOR = "Yousef Mohamed Mahmoud Mahmoud Alashmawy"
COURSE = "DLBAIPNLP01"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
R, F = ROOT / "results", ROOT / "figures"

# ------------------------------------------------------------------ fonts
FONT_SETS = [
    ("/System/Library/Fonts/Supplemental/", "Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf",
     "/System/Library/Fonts/Supplemental/Andale Mono.ttf"),
    ("/usr/share/fonts/truetype/liberation/", "LiberationSans-Regular.ttf",
     "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
]
for base, reg, bold, ital, mono in FONT_SETS:
    if Path(base, reg).exists():
        pdfmetrics.registerFont(TTFont("Sans", base + reg))
        pdfmetrics.registerFont(TTFont("Sans-Bold", base + bold))
        pdfmetrics.registerFont(TTFont("Sans-Italic", base + ital))
        pdfmetrics.registerFont(TTFont("Mono", mono))
        break
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold",
                              italic="Sans-Italic", boldItalic="Sans-Bold")

INK = colors.black
S = {
    "body": ParagraphStyle("body", fontName="Sans", fontSize=10.5, leading=15.3, spaceAfter=8, textColor=INK),
    "title": ParagraphStyle("title", fontName="Sans-Bold", fontSize=21, leading=26, spaceAfter=10, textColor=INK),
    "h1": ParagraphStyle("h1", keepWithNext=1, fontName="Sans-Bold", fontSize=15, leading=19, spaceBefore=4, spaceAfter=10, textColor=INK),
    "h2": ParagraphStyle("h2", keepWithNext=1, fontName="Sans-Bold", fontSize=11.5, leading=15, spaceBefore=5, spaceAfter=6, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Sans", fontSize=8.5, leading=10.8, spaceAfter=7, textColor=INK),
    "capt": ParagraphStyle("capt", fontName="Sans-Bold", fontSize=9, leading=11.5, spaceAfter=4, textColor=INK),
    "small": ParagraphStyle("small", fontName="Sans", fontSize=9, leading=12.5, spaceAfter=6, textColor=INK),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=7, leading=9, spaceAfter=4),
    "ref": ParagraphStyle("ref", fontName="Sans", fontSize=9.5, leading=14.25, spaceAfter=8,
                          leftIndent=36, firstLineIndent=-36, textColor=INK),
}
W = A4[0] - 100
story: list = []
manuscript: list[str] = []


def p(txt, style="body"):
    story.append(Paragraph(txt, S[style]))
    manuscript.append(re.sub("<[^>]+>", "", txt))


def h1(t): p(t, "h1")
def h2(t): p(t, "h2")


PENDING: list = []


def caption(kind, n, title):
    """Title above a figure/table; kept on the same page as the element."""
    txt = f"{'Fig.' if kind == 'fig' else 'Tab.'} {n} {title}"
    PENDING.append(Paragraph(txt, S["capt"]))
    manuscript.append(txt)


def place(*flowables):
    head = []
    # keep a heading that directly precedes the caption on the same page
    if story and isinstance(story[-1], Paragraph) and story[-1].style.name in ("h1", "h2"):
        head = [story.pop()]
    story.append(KeepTogether(head + PENDING + list(flowables)))
    PENDING.clear()


def source(txt):
    p("Source: " + txt, "cap")


def pic(path, width=W):
    im = Image(str(path))
    im.drawWidth, im.drawHeight = width, width * im.imageHeight / im.imageWidth
    place(im, Spacer(1, 4))


def table(rows, widths=None, fs=8.5, bold_first_col=False):
    cooked = [[Paragraph(html.escape(str(v)), ParagraphStyle(
        "tc", fontName="Sans-Bold" if i == 0 or (bold_first_col and j == 0) else "Sans",
        fontSize=fs, leading=fs + 2.6, textColor=INK)) for j, v in enumerate(row)]
        for i, row in enumerate(rows)]
    t = Table(cooked, colWidths=widths or [W / len(rows[0])] * len(rows[0]),
              repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e5e5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.black),
    ]))
    place(t, Spacer(1, 4))
    manuscript.append("\n".join(" | ".join(str(v) for v in r) for r in rows))


def page():
    story.append(PageBreak())
    manuscript.append("\n---\n")


def pct(v, d=2): return f"{100 * v:.{d}f}%"
def f4(v): return f"{v:.4f}"
def f3(v): return f"{v:.3f}"


# ------------------------------------------------------------------ data
sms, enron = D.load_sms(), D.load_enron()
sms_raw_n, enron_raw_n = len(D.load_sms(dedupe=False)), len(D.load_enron(dedupe=False))
ds = {"sms": D.describe(sms), "enron": D.describe(enron)}
sms_tr, sms_te = E.split(sms)
en_tr, en_te = E.split(enron)
e1 = pd.read_csv(R / "e1_preprocessing.csv")
e2 = pd.read_csv(R / "e2_learning_curve.csv")
e3 = pd.read_csv(R / "e3_models.csv")
e4 = pd.read_csv(R / "e4_generalisation.csv")
FIN = json.loads((R / "final_model.json").read_text())
MN = E.LABEL


def e1f(corpus, mode, model, vec="tfidf"):
    return e1[(e1.corpus == corpus) & (e1.preprocessing == mode) & (e1.model == model)
              & (e1.vectorizer == vec)].f1.iloc[0]


def e2f(corpus, size, model, col="f1_mean"):
    sub = e2[(e2.corpus == corpus) & (e2.model == model)]
    return sub[sub.train_size == size][col].iloc[0]


def e3r(corpus, model): return e3[(e3.corpus == corpus) & (e3.model == model)].iloc[0]
def e4r(tr, te): return e4[(e4.trained_on == tr) & (e4.tested_on == te)].iloc[0]


lin = ["linear_svm", "logistic_regression"]
sms_clean_gain = np.mean([e1f("sms", "clean", m) - e1f("sms", "raw", m) for m in lin])
nb_sw_drop = e1f("sms", "clean", "naive_bayes") - e1f("sms", "stopwords", "naive_bayes")
enron_range = (e1[(e1.corpus == "enron") & (e1.vectorizer == "tfidf") & (e1.model == "linear_svm")].f1)
vec_mean = e1.groupby(["corpus", "vectorizer"]).f1.mean()
sms_full, en_full = int(e2[e2.corpus == "sms"].train_size.max()), int(e2[e2.corpus == "enron"].train_size.max())
best_e3 = e3.groupby("model")["cv_f1_mean"].mean().idxmax()
fs, fe = FIN["sms"], FIN["email"]
ts, te_ = fs["test"], fe["test"]
comb_sms = e4r("combined", "sms")

# final-model error analysis (same objects the notebook inspects)
sms_model = joblib.load(ROOT / "models" / "sms_classifier.joblib")
email_model = joblib.load(ROOT / "models" / "email_classifier.joblib")
pred = sms_model.predict(sms_te.text.tolist())
errs = sms_te.assign(pred=pred)[pred != sms_te.label.to_numpy()]
fp_texts = errs[errs.label == 0].text.tolist()
fn_texts = errs[errs.label == 1].text.tolist()
names = email_model.named_steps["vectorize"].get_feature_names_out()
coef = email_model.named_steps["classify"].coef_.ravel()
order = np.argsort(coef)
spam_ng = [n for n in names[order[::-1]][:10]]
ham_ng = [n for n in names[order][:10]]


# ---------------------------------------------------------- graphical abstract
class Pipeline(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = W, 120

    def draw(self):
        c = self.canv
        bw, gap, y, ht = 84, (W - 5 * 84) / 4, 22, 78
        boxes = [
            ("Corpora", [f"SMS: {ds['sms']['messages']:,}", f"E-mail: {ds['enron']['messages']:,}", "labelled spam / ham"]),
            ("Preprocess", ["normalise entities", "stop words", "stem / lemmatise"]),
            ("Encode", ["Bag-of-Words", "TF-IDF", "1- and 2-grams"]),
            ("Classify", ["Naive Bayes, SVM", "logistic regression", "neural network"]),
            ("Evaluate", ["held-out test sets", "small to large data", "cross-corpus"]),
        ]
        for i, (title, lines) in enumerate(boxes):
            x = i * (bw + gap)
            c.setFillColor(colors.HexColor("#f2f2f2"))
            c.setStrokeColor(colors.black)
            c.roundRect(x, y, bw, ht, 6, fill=1)
            c.setFillColor(colors.black)
            c.setFont("Sans-Bold", 9)
            c.drawCentredString(x + bw / 2, y + ht - 17, title)
            c.setFont("Sans", 7.3)
            for k, ln in enumerate(lines):
                c.drawCentredString(x + bw / 2, y + ht - 33 - 12 * k, ln)
            if i < 4:
                x1, x2, ym = x + bw + 3, x + bw + gap - 3, y + ht / 2
                c.line(x1, ym, x2, ym)
                c.line(x2, ym, x2 - 5, ym + 3)
                c.line(x2, ym, x2 - 5, ym - 3)
        c.setFont("Sans", 7.6)
        c.drawString(0, 4, f"Result: channel-specific models. SMS accuracy {pct(ts['accuracy'])}, "
                           f"e-mail accuracy {pct(te_['accuracy'])}; legitimate messages flagged: "
                           f"{pct(ts['false_positive_rate'])} (SMS), {pct(te_['false_positive_rate'])} (e-mail).")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#b0b0b0"))
    canvas.line(50, 45, A4[0] - 50, 45)
    canvas.setFont("Sans", 8)
    canvas.setFillColor(INK)
    canvas.drawString(50, 31, f"{COURSE} | Spam detection with NLP")
    canvas.drawRightString(A4[0] - 50, 31, str(doc.page))
    canvas.restoreState()


# =================================================================== REPORT
p("Detecting Spam Messages with Natural Language Processing", "title")
h2("Preprocessing, text encoding and supervised classifiers evaluated from small to large message collections")
p(f"Project report | {COURSE} - Project: NLP | Task 2: Text Classification for Spam Detection", "small")
p(f"{AUTHOR}<br/>{ARGS.date}", "small")
p(f"Code repository: <link href='{ARGS.repo}'>{html.escape(ARGS.repo)}</link>", "small")

h2("Abstract")
p("Spam messages are unsolicited bulk messages that waste attention and can carry phishing links or "
  "malware. This project develops a natural language processing (NLP) system that classifies text "
  "messages and e-mails as spam or legitimate (ham). Two public corpora were used: the SMS Spam "
  f"Collection with {ds['sms']['messages']:,} unique messages as the small scenario and Enron-Spam with "
  f"{ds['enron']['messages']:,} unique e-mails as the large scenario. Messages were normalised, stop "
  "words were removed, words were stemmed or lemmatised, and the text was encoded as Bag-of-Words or "
  "TF-IDF n-gram vectors. Naive Bayes, a linear support vector machine (SVM), logistic regression and a "
  "small feed-forward neural network were trained and evaluated on held-out test data, with learning "
  "curves from 100 training messages up to the full corpora and tests across the two message types.")
p(f"Replacing numbers, phone numbers, URLs and currency symbols with placeholder tokens was the most "
  f"useful preprocessing step for SMS, raising F1 by {100 * sms_clean_gain:.1f} points for the linear "
  "models, while preprocessing hardly changed e-mail results. Linear models matched or beat the neural "
  "network at a fraction of its training time. Models trained on one message type failed on the other, "
  "so the final system routes each message to a channel-specific model. On untouched test data the SMS "
  f"model reached {pct(ts['accuracy'])} accuracy and flagged {ts['fp']} of {ts['fp'] + ts['tn']} legitimate "
  f"messages; the e-mail model reached {pct(te_['accuracy'])} accuracy with an F1 score of {f3(te_['f1'])}.")

h2("Graphical abstract")
caption("fig", 1, "Graphical abstract")
place(Pipeline(), Spacer(1, 4))
source("Own representation of the system developed in this project. Figures in the bottom line are "
       "measured on the held-out test sets (see Section 4).")
page()

# ------------------------------------------------------------- 1 Introduction
h1("1. Introduction")
p("Text classification assigns a document to one of a set of predefined categories; spam detection is "
  "its classic binary case, in which every incoming message is labelled as spam or ham (Jurafsky &amp; "
  "Martin, 2013). A filter is only useful if it removes most spam while almost never hiding a "
  "legitimate message, because a lost personal or business message costs the user far more than one "
  "unwanted advertisement. This project builds such a filter and examines each step of the NLP pipeline "
  "in different scenarios. It addresses four questions:")
p("<b>Q1</b> Which preprocessing steps (normalisation, stop-word removal, stemming, lemmatising) and "
  "which text encoding (Bag-of-Words or TF-IDF) improve classification?<br/>"
  "<b>Q2</b> How does performance develop from a small to a large training set?<br/>"
  "<b>Q3</b> Do probabilistic, discriminative and neural classifiers differ in accuracy and cost?<br/>"
  "<b>Q4</b> Does a model trained on one kind of message generalise to another kind?")

h2("1.1 Related work")
p("Naive Bayes classifiers have long been a standard approach to spam filtering. Metsis et al. (2006) "
  "compared several Naive Bayes variants on e-mail and, for that purpose, assembled the Enron-Spam "
  "corpus, which combines legitimate mail from Enron employees with collected spam. For mobile "
  "messages, Almeida et al. (2011) published the SMS Spam Collection and reported results of established "
  "classifiers as baselines, noting that short, abbreviation-rich SMS texts are harder to filter than "
  "e-mails. More recent work uses deep learning: Elakkiya et al. (2021) combine textual features with an "
  "attention mechanism to detect spam in social media, and Huan et al. (2022) join convolutional and "
  "bidirectional long short-term memory layers so that a text classifier captures both local n-gram "
  "patterns and longer-range word order. Such models are more expressive but need more data and "
  "computation. This project therefore compares classical linear models on sparse n-gram features with "
  "a neural network on the same features, and asks whether the extra cost is justified for this task.")

h2("1.2 Data collection")
p("The task suggests the UCI Spambase data set. Spambase, however, contains only 57 pre-computed numeric "
  "features such as word and character frequencies for 4,601 e-mails, not the e-mail text (Hopkins et "
  "al., 1999). Stop-word removal, stemming and text encoding, which the task asks to evaluate, cannot be "
  "applied to it. Two public corpora with raw text and labels were therefore used (Tab. 1). The SMS "
  "Spam Collection (Almeida &amp; Hidalgo, 2011) provides the small scenario. Enron-Spam (Metsis et al., "
  "2006), in the CSV version prepared by Wiechmann (n.d.) with subject and body per e-mail, provides the "
  "large scenario. Exact duplicate texts were removed before any split; otherwise the same message could "
  "appear in both training and test data and inflate the scores.")
caption("tab", 1, "Data sets after de-duplication")
table([
    ["Corpus", "Type", "Raw", "Unique", "Spam share", "Median length", "Train / test"],
    ["SMS Spam Collection", "SMS", f"{sms_raw_n:,}", f"{ds['sms']['messages']:,}", pct(ds['sms']['spam_share'], 1),
     f"{ds['sms']['median_tokens']:.0f} words", f"{len(sms_tr):,} / {len(sms_te):,}"],
    ["Enron-Spam", "E-mail", f"{enron_raw_n:,}", f"{ds['enron']['messages']:,}", pct(ds['enron']['spam_share'], 1),
     f"{ds['enron']['median_tokens']:.0f} words", f"{len(en_tr):,} / {len(en_te):,}"],
], widths=[100, 45, 50, 52, 62, 75, W - 384])
source("Own representation of the loaded data; data from Almeida and Hidalgo (2011) and Metsis et al. "
       "(2006) via Wiechmann (n.d.). Raw counts exclude e-mails without any text.")
p("The two corpora differ strongly. SMS spam is a minority class (about one message in eight) and SMS "
  "texts are short and informal, whereas Enron-Spam is almost balanced and its e-mails are more than ten "
  "times longer. This difference is used deliberately in Q4.")

# ------------------------------------------------------------- 2 Method
h1("2. Method")
h2("2.1 Preprocessing")
p("Five cumulative preprocessing levels were implemented (Tab. 2) so that the effect of each step can "
  "be measured separately. Stop words come from the NLTK English list (Bird et al., 2009). Stemming "
  "uses the Porter algorithm, which strips suffixes by rules (Porter, 1980); lemmatising maps words to "
  "dictionary base forms with WordNet (Miller, 1995), first as nouns and then as verbs. The placeholder "
  "tokens of the clean level are not stemmed.")
caption("tab", 2, "Preprocessing levels")
table([
    ["Level", "Operation", "Example output for a spam SMS"],
    ["raw", "Lower-casing only (baseline)", "free entry in 2 a wkly comp to win fa cup final tkts 21st may ..."],
    ["clean", "URLs, e-mail addresses, phone numbers, currency symbols and numbers replaced by tokens; punctuation removed",
     "free entry in tok_num a wkly comp to win fa cup final tkts st may ..."],
    ["stopwords", "clean + NLTK English stop words removed", "free entry tok_num wkly comp win fa cup final tkts st may ..."],
    ["stem", "stopwords + Porter stemming", "free entri tok_num wkli comp win fa cup final tkt st may ..."],
    ["lemma", "stopwords + WordNet lemmatising", "free entry tok_num wkly comp win fa cup final tkts st may ..."],
], widths=[55, 190, W - 245], fs=8)
source("Own representation of the implemented preprocessing (src/preprocess.py); example message from "
       "the SMS Spam Collection (Almeida &amp; Hidalgo, 2011).")

h2("2.2 Encoding and classifiers")
p("Preprocessed text was converted into sparse vectors of unigrams and bigrams that occur in at least "
  "two training messages, limited to 50,000 features. Two weightings were compared: raw counts "
  "(Bag-of-Words) and TF-IDF with sublinear term frequency, which down-weights terms that occur in many "
  "documents (Manning et al., 2008). The encoders were fitted on training data only. Four supervised "
  "classifiers from scikit-learn (Pedregosa et al., 2011) were trained on these vectors (Tab. 3).")
caption("tab", 3, "Classifier settings")
table([
    ["Classifier", "Type", "Settings"],
    ["Multinomial Naive Bayes", "Probabilistic, generative", "Additive smoothing alpha = 0.1"],
    ["Linear SVM", "Discriminative, maximum margin", "C = 1, balanced class weights"],
    ["Logistic regression", "Discriminative, probabilistic", "C = 10, liblinear solver, balanced class weights"],
    ["MLP (neural network)", "Feed-forward, deep learning baseline",
     "One hidden layer of 64 ReLU units, Adam, early stopping, at most 50 epochs, 20,000 input features"],
], widths=[115, 140, W - 255], fs=8)
source("Own representation of the implementation (src/models.py), based on Pedregosa et al. (2011).")

h2("2.3 Evaluation protocol")
p("Each corpus was split once into a stratified 80% training and 20% test part (random seed 42). The "
  "test parts were never used for training or for choosing a configuration. Spam is the positive class. "
  "Reported metrics are accuracy, precision (share of flagged messages that are spam), recall (share of "
  "spam that is caught), F1 (harmonic mean of precision and recall), ROC-AUC and the false-positive rate, "
  "the share of legitimate messages wrongly flagged. Accuracy alone is misleading on the imbalanced SMS "
  "data: a filter that lets every message through is already about 87% accurate.")
p("Four experiments answer Q1 to Q4. <b>E1</b> trains every combination of preprocessing level, "
  "encoding and classical classifier on each training set. <b>E2</b> trains on stratified random subsets "
  "from 100 messages to the full training set, five different subsets per size, and always tests on the "
  "same full test set. <b>E3</b> compares all four classifiers with 5-fold cross-validation on the "
  "training data and on the test set. <b>E4</b> trains the best classifier on SMS only, e-mail only and "
  "both combined and tests it on both test sets. Finally, for each channel the preprocessing level and "
  "classifier were selected by cross-validated F1 on the training data, refitted and evaluated once on "
  "the test set. The experiments ran in Python 3.13 with scikit-learn 1.9 and NLTK 3.10 on a Linux "
  "machine with two CPU cores; the executed notebook in the appendix shows all code and outputs.")

# ------------------------------------------------------------- 3 Results
h1("3. Results")
h2("3.1 Effect of preprocessing and encoding (E1)")
rows = [["Corpus", "Level", "Naive Bayes", "Linear SVM", "Logistic regression"]]
for corpus, cname in [("sms", "SMS"), ("enron", "E-mail")]:
    for mode in ["raw", "clean", "stopwords", "stem", "lemma"]:
        rows.append([cname, mode] + [f4(e1f(corpus, mode, m)) for m in ["naive_bayes", "linear_svm", "logistic_regression"]])
caption("tab", 4, "Test F1 by preprocessing level (TF-IDF encoding)")
table(rows, widths=[60, 75, (W - 135) / 3, (W - 135) / 3, (W - 135) / 3], fs=8)
source("Own representation of measured experiment outputs (results/e1_preprocessing.csv). Bag-of-Words "
       "results are in the same file.")
p(f"For SMS, normalisation was the decisive step: replacing numbers, phone numbers, currency symbols "
  f"and URLs by tokens raised F1 from {f4(e1f('sms', 'raw', 'linear_svm'))} to "
  f"{f4(e1f('sms', 'clean', 'linear_svm'))} for the SVM and by {100 * sms_clean_gain:.1f} points on "
  "average for the two linear models (Tab. 4, Fig. 2). SMS spam typically asks the reader to call a "
  "premium number, text a short code or claim a cash prize; before normalisation, every number is a "
  "different rare word, afterwards the shared pattern becomes one strong feature. Stop-word removal, "
  "stemming and lemmatising changed the linear models by at most about one point, in both directions. "
  f"They harmed Naive Bayes: removing stop words lowered its F1 by {100 * nb_sw_drop:.1f} points. In "
  "very short texts, function words such as 'you', 'your' or 'to' carry part of the persuasive style "
  "of spam, and Naive Bayes, which weighs every word independently, loses more from removing them. On "
  f"e-mail, all levels lie within {100 * (enron_range.max() - enron_range.min()):.1f} points of each other; "
  "long texts contain enough evidence for the classifier to ignore noise. TF-IDF outperformed raw "
  f"counts on average (SMS {f3(vec_mean[('sms', 'tfidf')])} vs. {f3(vec_mean[('sms', 'bow')])}; e-mail "
  f"{f3(vec_mean[('enron', 'tfidf')])} vs. {f3(vec_mean[('enron', 'bow')])}).")
caption("fig", 2, "Test F1 by preprocessing level")
pic(F / "preprocessing_ablation.png")
source("Own representation of measured experiment outputs. Note the different vertical scales.")

h2("3.2 From small to large training sets (E2)")
caption("fig", 3, "Learning curves")
pic(F / "learning_curves.png")
source("Own representation of measured experiment outputs (results/e2_learning_curve.csv). Points are "
       "means over five random training subsets, bars one standard deviation; the full training set is "
       "a single run.")
p(f"Performance rose steeply up to about 1,000 training messages and then levelled off (Fig. 3). With "
  f"100 messages the SVM reached an F1 of {f3(e2f('sms', 100, 'linear_svm'))} on SMS and "
  f"{f3(e2f('enron', 100, 'linear_svm'))} on e-mail; with the full training sets "
  f"({sms_full:,} and {en_full:,} messages) it reached {f3(e2f('sms', sms_full, 'linear_svm'))} and "
  f"{f3(e2f('enron', en_full, 'linear_svm'))}. The spread between random subsets shrank from about "
  f"{100 * e2f('sms', 100, 'linear_svm', 'f1_std'):.0f} F1 points at 100 SMS messages to about "
  f"{100 * e2f('sms', 2000, 'linear_svm', 'f1_std'):.0f} point at 2,000, and on e-mail to "
  f"{100 * e2f('enron', 500, 'linear_svm', 'f1_std'):.1f} points from 500 messages onwards, so small training sets give not only lower but also less reliable results. The "
  "false-positive rate on e-mail fell from "
  f"{pct(e2f('enron', 100, 'linear_svm', 'false_positive_rate_mean'), 1)} to "
  f"{pct(e2f('enron', en_full, 'linear_svm', 'false_positive_rate_mean'), 1)}. Naive Bayes was "
  "competitive with little data but improved least as data grew, which matches the usual behaviour of "
  "a generative model with strong independence assumptions compared with discriminative models. The "
  "SMS curves are still rising at the full corpus size, so more labelled SMS data would probably help; "
  "the e-mail curves have almost converged.")

h2("3.3 Comparison of classifiers (E3)")
rows = [["Corpus", "Classifier", "CV F1 (mean ± sd)", "Accuracy", "Precision", "Recall", "F1", "AUC", "Train time"]]
for corpus, cname in [("sms", "SMS"), ("enron", "E-mail")]:
    for m in ["naive_bayes", "linear_svm", "logistic_regression", "mlp"]:
        r = e3r(corpus, m)
        rows.append([cname, MN[m], f"{f3(r.cv_f1_mean)} ± {f3(r.cv_f1_std)}", f4(r.test_accuracy),
                     f4(r.test_precision), f4(r.test_recall), f4(r.test_f1), f4(r.test_roc_auc),
                     f"{r.test_train_seconds:.1f} s"])
caption("tab", 5, "Classifier comparison (lemmatised text, TF-IDF)")
table(rows, widths=[40, 72, 72, 46, 46, 42, 42, 46, W - 406], fs=7.5)
source("Own representation of measured experiment outputs (results/e3_models.csv). Training time is "
       "for one fit on the full training set.")
mlp_s, svm_s = e3r("sms", "mlp"), e3r("sms", "linear_svm")
mlp_e, svm_e = e3r("enron", "mlp"), e3r("enron", "linear_svm")
p(f"Cross-validated and test scores agree closely, so the results do not depend on a lucky split "
  f"(Tab. 5). The linear SVM and logistic regression were the strongest models; across both corpora "
  f"the SVM had the highest mean cross-validated F1. The neural network did not improve on them: on "
  f"SMS its F1 was {f3(mlp_s.test_f1)} against {f3(svm_s.test_f1)} for the SVM, with the lowest and "
  f"most variable cross-validation score, a sign of overfitting on only {len(sms_tr):,} messages. On "
  f"e-mail it reached {f3(mlp_e.test_f1)} but needed {mlp_e.test_train_seconds / svm_e.test_train_seconds:.0f} "
  f"times as long to train. With thousands of sparse n-gram features, spam and ham are already almost "
  "linearly separable, so a hidden layer adds cost without adding accuracy. All models achieved a "
  "ROC-AUC above 0.99, meaning their ranking of messages is very good and differences lie mainly in "
  "where the decision threshold falls (AUC = area under the ROC curve).")

h2("3.4 Generalisation across message types (E4)")
rows = [["Trained on", "Tested on", "Accuracy", "Precision", "Recall", "F1", "Ham flagged"]]
lab = {"sms": "SMS", "enron": "E-mail", "combined": "SMS + e-mail"}
for _, r in e4.iterrows():
    rows.append([lab[r.trained_on], lab[r.tested_on], f4(r.accuracy), f4(r.precision), f4(r.recall),
                 f4(r.f1), pct(r.false_positive_rate)])
caption("tab", 6, f"Cross-corpus evaluation ({MN[best_e3]}, lemmatised text, TF-IDF)")
table(rows, widths=[85, 60, 60, 60, 55, 55, W - 375], fs=8)
source("Own representation of measured experiment outputs (results/e4_generalisation.csv).")
p(f"A model is only reliable on the kind of text it was trained on (Tab. 6). The e-mail model flagged "
  f"{pct(e4r('enron', 'sms').false_positive_rate, 0)} of legitimate SMS as spam, because casual text "
  "messages look nothing like Enron business mail, and the SMS model detected fewer than half of the "
  "spam e-mails. Training on both corpora together largely solved this, but SMS performance still "
  f"dropped to an F1 of {f3(comb_sms.f1)}, and {pct(comb_sms.false_positive_rate, 1)} of legitimate SMS "
  "were flagged, compared with "
  f"{pct(e4r('sms', 'sms').false_positive_rate, 1)} for the SMS-only model. The six times larger and "
  "almost balanced e-mail corpus dominates the joint model. The final system therefore uses one model "
  "per channel.")

# ------------------------------------------------------------- 4 Final system
h1("4. Final system")
p("For each channel, all combinations of four preprocessing levels and three classical classifiers were "
  "scored by cross-validated F1 on the training data only (5 folds for SMS, 3 for e-mail). The best "
  "combination was refitted on the full training set and saved as a pipeline that accepts raw text. A "
  "simple router sends multi-line texts or texts longer than 60 words to the e-mail model and all others "
  "to the SMS model. The command-line tool <i>predict.py</i> classifies new messages with these "
  "pipelines. Tab. 7 reports their single evaluation on the untouched test sets.")
caption("tab", 7, "Final models on the held-out test sets")
table([
    ["", "SMS model", "E-mail model"],
    ["Selected by cross-validation", f"{MN[fs['selected']['model']]}, {fs['selected']['mode']} text",
     f"{MN[fe['selected']['model']]}, {fe['selected']['mode']} text"],
    ["Cross-validated F1 (training data)", f3(fs["selected"]["cv_f1"]), f3(fe["selected"]["cv_f1"])],
    ["Test messages", f"{fs['test_messages']:,}", f"{fe['test_messages']:,}"],
    ["Accuracy", pct(ts["accuracy"]), pct(te_["accuracy"])],
    ["Precision / recall", f"{pct(ts['precision'])} / {pct(ts['recall'])}", f"{pct(te_['precision'])} / {pct(te_['recall'])}"],
    ["F1 / ROC-AUC", f"{f3(ts['f1'])} / {f3(ts['roc_auc'])}", f"{f3(te_['f1'])} / {f3(te_['roc_auc'])}"],
    ["Legitimate messages flagged", f"{ts['fp']} of {ts['fp'] + ts['tn']:,} ({pct(ts['false_positive_rate'])})",
     f"{te_['fp']} of {te_['fp'] + te_['tn']:,} ({pct(te_['false_positive_rate'])})"],
    ["Spam missed", f"{ts['fn']} of {ts['fn'] + ts['tp']:,}", f"{te_['fn']} of {te_['fn'] + te_['tp']:,}"],
    ["Training / prediction time", f"{fs['train_seconds']:.1f} s / {fs['predict_ms_per_message']:.2f} ms per message",
     f"{fe['train_seconds']:.1f} s / {fe['predict_ms_per_message']:.2f} ms per message"],
], widths=[150, (W - 150) / 2, (W - 150) / 2], fs=8.5, bold_first_col=True)
source("Own representation of measured experiment outputs (results/final_model.json).")
caption("fig", 4, "Confusion matrices of the final models")
im1, im2 = Image(str(F / "confusion_final_sms.png")), Image(str(F / "confusion_final_email.png"))
for im in (im1, im2):
    im.drawWidth, im.drawHeight = 200, 200 * im.imageHeight / im.imageWidth
grid = Table([[im1, im2]], colWidths=[W / 2, W / 2])
place(grid)
manuscript.append("[confusion matrices]")
source("Own representation of measured experiment outputs; rows are true classes, columns predicted classes.")
def pick(texts, key):
    hit = [t for t in texts if key.lower() in t.lower()]
    return html.escape((hit or texts)[0][:80])


p(f"For SMS, cross-validation selected Naive Bayes on lightly cleaned text, without stop-word removal "
  f"or lemmatising, in line with E1. It flagged only {ts['fp']} legitimate messages while missing "
  f"{ts['fn']} spam messages (Fig. 4). This is the preferable trade-off for a spam filter: a missed spam "
  "message is a nuisance, a hidden legitimate one can be costly. Several missed spam messages imitate "
  f"harmless content, such as a joke (\"{pick(fn_texts, 'toilet')}\") or a notification "
  f"(\"{pick(fn_texts, 'missed call')}\"). The flagged legitimate messages use product-like wording: "
  f"\"{pick(fp_texts, 'nokia')}\" and \"{pick(fp_texts, 'unlimited')}\". For e-mail, the linear SVM reached an F1 of "
  f"{f3(te_['f1'])}. Its strongest spam indicators include {', '.join(spam_ng[:6])}, while ham "
  f"indicators include {', '.join(ham_ng[:6])}; part of the signal is thus specific to Enron (Section 5).")

# ------------------------------------------------------------- 5 Discussion
h1("5. Discussion and limitations")
p("The experiments show that the value of each pipeline step depends on the kind of text. For short, "
  "noisy SMS, normalisation of numbers, currencies and links mattered more than any linguistic "
  "processing, and stemming or lemmatising brought no consistent gain. Their main advantage, merging "
  "word forms, is small when bigrams and thousands of training examples are available, and they remove "
  "stylistic cues such as abbreviations. For long e-mails the choice of preprocessing hardly mattered. "
  "Data size mattered more than model choice: going from 100 to 1,000 training messages gained more F1 "
  "than any change of classifier at full size. Among the classifiers, discriminative linear models were "
  "the best compromise of accuracy, speed and transparency, because their weights directly show which "
  "n-grams drive a decision.")
p("The most important finding for deployment is the lack of transfer between message types (E4). A "
  "filter is only as good as the match between its training data and the messages it will see, so the "
  "system keeps one model per channel and should be retrained on data from its actual environment.")
p("The study has limitations. Enron-Spam consists of e-mails from the early 2000s, and its ham comes "
  "from one company; several strong ham indicators are Enron-specific names and terms, so the e-mail "
  "scores are likely optimistic for a general mailbox. Modern spam relies more on phishing links, "
  "images and several languages. Each corpus was split only once, although cross-validation and "
  "repeated subsamples in E2 reduce this risk. The decision threshold was left at the default; in "
  "practice it could be tuned to a target false-positive rate. Finally, transformer-based language "
  "models were not tested; given the near-perfect ROC-AUC of the linear models, their expected benefit "
  "here is small compared with their computational cost, but they could help with more varied data.")

h1("6. Conclusion")
p("This project developed and evaluated an NLP system for spam detection. It collects two public "
  "labelled corpora, preprocesses the text with normalisation, stop-word removal, stemming or "
  "lemmatising, encodes it as Bag-of-Words or TF-IDF n-grams, trains Naive Bayes, a linear SVM, logistic "
  "regression and a neural network, and evaluates them from small to large data sets and across message "
  "types. Normalising entities was the most useful preprocessing step for SMS (Q1), performance improved "
  "quickly up to about 1,000 training messages and then levelled off (Q2), linear models were as accurate "
  "as the neural network at much lower cost (Q3), and models did not transfer between SMS and e-mail (Q4). "
  f"The final channel-specific system classifies new messages with {pct(ts['accuracy'])} (SMS) and "
  f"{pct(te_['accuracy'])} (e-mail) accuracy on held-out test data while flagging very few legitimate "
  "messages. Future work should retrain and test the system on recent, multilingual messages, tune the "
  "decision threshold, and compare it with transformer-based models.")

# ------------------------------------------------------------- References
page()
h1("References")
refs = [
    "Almeida, T. A., Gómez Hidalgo, J. M., &amp; Yamakami, A. (2011). Contributions to the study of SMS spam "
    "filtering: New collection and results. In <i>Proceedings of the 11th ACM Symposium on Document "
    "Engineering</i> (pp. 259–262). Association for Computing Machinery. https://doi.org/10.1145/2034691.2034742",
    "Almeida, T. A., &amp; Hidalgo, J. M. G. (2011). <i>SMS spam collection</i> [Data set]. UCI Machine "
    "Learning Repository. https://doi.org/10.24432/C5CC84",
    "Bird, S., Klein, E., &amp; Loper, E. (2009). <i>Natural language processing with Python</i>. O'Reilly Media.",
    "Elakkiya, E., Selvakumar, S., &amp; Leela Velusamy, R. (2021). TextSpamDetector: Textual content based "
    "deep learning framework for social spam detection using conjoint attention mechanism. <i>Journal of "
    "Ambient Intelligence and Humanized Computing, 12</i>(10), 9287–9302. https://doi.org/10.1007/s12652-020-02640-5",
    "Hopkins, M., Reeber, E., Forman, G., &amp; Suermondt, J. (1999). <i>Spambase</i> [Data set]. UCI "
    "Machine Learning Repository. https://doi.org/10.24432/C53G6X",
    "Huan, H., Guo, Z., Cai, T., &amp; He, Z. (2022). A text classification method based on a convolutional "
    "and bidirectional long short-term memory model. <i>Connection Science, 34</i>(1), 2108–2124. "
    "https://doi.org/10.1080/09540091.2022.2098926",
    "Jurafsky, D., &amp; Martin, J. H. (2013). <i>Speech and language processing: An introduction to natural "
    "language processing, computational linguistics, and speech recognition</i> (2nd ed.). Pearson Prentice Hall.",
    "Manning, C. D., Raghavan, P., &amp; Schütze, H. (2008). <i>Introduction to information retrieval</i>. "
    "Cambridge University Press.",
    "Metsis, V., Androutsopoulos, I., &amp; Paliouras, G. (2006). Spam filtering with naive Bayes – Which "
    "naive Bayes? In <i>Proceedings of the Third Conference on Email and Anti-Spam (CEAS 2006)</i>. "
    "Mountain View, CA, United States.",
    "Miller, G. A. (1995). WordNet: A lexical database for English. <i>Communications of the ACM, 38</i>(11), "
    "39–41. https://doi.org/10.1145/219717.219748",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., "
    "Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., "
    "Perrot, M., &amp; Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine "
    "Learning Research, 12</i>, 2825–2830.",
    "Porter, M. F. (1980). An algorithm for suffix stripping. <i>Program, 14</i>(3), 130–137. "
    "https://doi.org/10.1108/eb046814",
    "Wiechmann, M. (n.d.). <i>enron_spam_data</i> [Data set and source code]. GitHub. "
    "https://github.com/MWiechmann/enron_spam_data",
]
for r in refs:
    p(r, "ref")
h2("Code and appendix")
p(f"The complete source code, tests, results and trained models are available at {html.escape(ARGS.repo)}. "
  "The appendix that follows is a PDF rendering of the executed notebook "
  "<i>Spam_Detection_Experiments.ipynb</i>, which contains every code cell and its observed output; the "
  "same notebook is part of the repository.", "small")

report_pdf = OUT / "Spam_Detection_Project_Report.pdf"
SimpleDocTemplate(str(report_pdf), pagesize=A4, leftMargin=50, rightMargin=50, topMargin=48,
                  bottomMargin=60, title="Detecting Spam Messages with NLP - DLBAIPNLP01",
                  author=AUTHOR).build(story, onFirstPage=footer, onLaterPages=footer)
(ROOT / "report").mkdir(exist_ok=True)
(ROOT / "report" / "Spam_Detection_Project_Report.md").write_text("\n\n".join(manuscript))

# =================================================================== NOTEBOOK
nb = nbformat.read(ROOT / "notebooks" / "Spam_Detection_Experiments.ipynb", as_version=4)
assert all(c.get("execution_count") is not None for c in nb.cells if c.cell_type == "code"), "notebook not executed"
assert not any(o.output_type == "error" for c in nb.cells for o in c.get("outputs", [])), "notebook has errors"
ns = [Paragraph("Appendix: Executed experiment notebook", S["title"]),
      Paragraph("Spam_Detection_Experiments.ipynb | code and observed outputs", S["small"])]
TR = str.maketrans({"␣": " ", "↪": ">", "…": "...", "±": "+/-"})


def terminal(txt):
    txt = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", txt)
    out, line, cur = [], [], 0
    for ch in txt:
        if ch == "\n":
            out.append("".join(line)); line, cur = [], 0
        elif ch == "\r":
            cur = 0
        elif ord(ch) >= 32 or ch == "\t":
            if cur < len(line): line[cur] = ch
            else: line.append(ch)
            cur += 1
    if line: out.append("".join(line))
    return "\n".join(out)


def pre(txt):
    lines = []
    for ln in txt.translate(TR).splitlines():
        lines.extend(textwrap.wrap(ln, width=108, replace_whitespace=False, drop_whitespace=False,
                                   subsequent_indent="    ") or [""])
    for i in range(0, len(lines), 60):
        ns.append(Preformatted("\n".join(lines[i:i + 60]), S["code"]))
        ns.append(Spacer(1, 4))


for ci, cell in enumerate(nb.cells, 1):
    if cell.cell_type == "markdown":
        for para in cell.source.split("\n\n"):
            if para.startswith("#"):
                head, _, rest = para.partition("\n")
                ns.append(Paragraph(html.escape(head.lstrip("# ").strip()), S["h2"]))
                if rest.strip(): ns.append(Paragraph(html.escape(rest).replace("\n", " "), S["small"]))
            else:
                ns.append(Paragraph(html.escape(para).replace("\n", " ").replace("`", ""), S["small"]))
    else:
        ns.append(Paragraph(f"In [{cell.execution_count}] - notebook cell {ci}", S["capt"]))
        pre(cell.source)
        text = "".join(o.text for o in cell.get("outputs", []) if o.output_type == "stream" and o.name == "stdout")
        if text.strip():
            ns.append(Paragraph("Observed output", S["cap"]))
            pre(terminal(text))
        for o in cell.get("outputs", []):
            if o.output_type in ("display_data", "execute_result"):
                d = o.get("data", {})
                if "image/png" in d:
                    im = Image(io.BytesIO(base64.b64decode(d["image/png"])))
                    sc = min(W / im.imageWidth, 260 / im.imageHeight)
                    im.drawWidth, im.drawHeight = im.imageWidth * sc, im.imageHeight * sc
                    ns.append(im); ns.append(Spacer(1, 8))
                elif "text/plain" in d:
                    pre(d["text/plain"])

nb_pdf = OUT / "Spam_Detection_Executed_Notebook.pdf"
SimpleDocTemplate(str(nb_pdf), pagesize=A4, leftMargin=50, rightMargin=50, topMargin=48,
                  bottomMargin=60, title="Executed spam detection notebook",
                  author=AUTHOR).build(ns, onFirstPage=footer, onLaterPages=footer)

n_rep = len(PdfReader(report_pdf).pages)
combined = OUT / "YousefAlashmawy-NLP-Submission-October2026.pdf"
w = PdfWriter()
w.append(str(report_pdf)); w.append(str(nb_pdf))
w.add_metadata({"/Title": "Detecting Spam Messages with NLP - report and executed code", "/Author": AUTHOR})
with combined.open("wb") as fh:
    w.write(fh)
print("report pages:", n_rep, "| notebook pages:", len(PdfReader(nb_pdf).pages))
print("words in report:", len(" ".join(manuscript).split()))
for f in (report_pdf, nb_pdf, combined):
    print(f)
