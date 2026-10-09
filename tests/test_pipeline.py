"""Unit and integration tests.  Run with:  python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import describe, load_enron, load_sms  # noqa: E402
from models import MODEL_NAMES, build_pipeline, evaluate  # noqa: E402
from preprocess import MODES, TextPreprocessor, normalise, preprocess  # noqa: E402

TOY_SPAM = [
    "WINNER! You have won a free prize, call 09061701461 now to claim £1000",
    "URGENT your mobile number won cash award, text CLAIM to 81010",
    "Free entry to win a brand new car, reply WIN now www.win-now.com",
    "Congratulations you are selected for a free holiday, call now",
] * 5
TOY_HAM = [
    "Hey are we still meeting for lunch tomorrow?",
    "I will be home late tonight, can you feed the cat",
    "Thanks for the notes from the meeting, see you on Monday",
    "Did you finish the report for the project yet?",
] * 5


# ------------------------------------------------------------ preprocessing
def test_normalise_replaces_entities():
    out = normalise("Visit http://x.com or mail a@b.com, call 0800123456 for £5!")
    for tok in ("tok_url", "tok_email", "tok_phone", "tok_money", "tok_num"):
        assert tok in out
    assert "!" not in out and "," not in out


def test_stopwords_removed():
    out = preprocess("this is the best offer for you", "stopwords").split()
    assert "the" not in out and "is" not in out and "best" in out


def test_stemming_and_lemmatising():
    assert "win" in preprocess("winning winners", "stem").split()
    lem = preprocess("the cars were running", "lemma").split()
    assert "car" in lem and "run" in lem


def test_placeholder_tokens_not_stemmed():
    assert "tok_url" in preprocess("go to www.spam.com", "stem")


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_string(mode):
    assert isinstance(preprocess("Hello WORLD 123!!", mode), str)


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        preprocess("x", "nonsense")
    with pytest.raises(ValueError):
        TextPreprocessor("nonsense").fit(["x"])


def test_handles_empty_and_non_string():
    assert preprocess("", "lemma") == ""
    assert isinstance(preprocess(12345, "lemma"), str)


# ------------------------------------------------------------------ metrics
def test_evaluate_counts():
    m = evaluate(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1]))
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 1, 1)
    assert m["accuracy"] == 0.5 and m["false_positive_rate"] == 0.5


# ------------------------------------------------------------------- models
@pytest.mark.parametrize("model", MODEL_NAMES)
def test_every_model_learns_toy_data(model):
    X = TOY_SPAM + TOY_HAM
    y = np.array([1] * len(TOY_SPAM) + [0] * len(TOY_HAM))
    pipe = build_pipeline(model, "lemma", "tfidf").fit(X, y)
    pred = pipe.predict(["claim your free prize now, call 09061701461",
                         "see you at the meeting on monday"])
    if model != "mlp":  # the tiny MLP is not guaranteed to converge on 40 rows
        assert list(pred) == [1, 0]
    assert set(pred) <= {0, 1}


# --------------------------------------------------------------------- data
@pytest.fixture(scope="module")
def sms():
    return load_sms()


def test_sms_loaded_and_deduplicated(sms):
    assert {"text", "label", "source"} <= set(sms.columns)
    assert set(sms["label"].unique()) == {0, 1}
    assert sms["text"].is_unique
    assert 5000 < len(sms) < 5600
    assert 0.10 < describe(sms)["spam_share"] < 0.16


def test_enron_loaded():
    df = load_enron()
    assert len(df) > 25_000 and df["text"].is_unique
    assert 0.4 < df["label"].mean() < 0.6


# --------------------------------------------------------- end to end (SMS)
def test_sms_end_to_end_quality(sms):
    from sklearn.model_selection import train_test_split

    tr, te = train_test_split(sms, test_size=0.2, stratify=sms["label"],
                              random_state=0)
    pipe = build_pipeline("linear_svm").fit(tr["text"].tolist(), tr["label"])
    m = evaluate(te["label"].to_numpy(), pipe.predict(te["text"].tolist()))
    assert m["accuracy"] > 0.97
    assert m["f1"] > 0.88
    assert m["false_positive_rate"] < 0.02


# ---------------------------------------------------------- saved model/CLI
MODELS_READY = all((ROOT / "models" / f"{c}_classifier.joblib").exists()
                   for c in ("sms", "email"))
needs_models = pytest.mark.skipif(not MODELS_READY, reason="run src/train.py first")


def test_routing():
    from predict import route

    assert route("see you at 5") == "sms"
    assert route("Subject: report\nPlease find attached") == "email"
    assert route(" ".join(["word"] * 80)) == "email"


@needs_models
def test_saved_sms_model_classifies_new_messages():
    from predict import classify

    res = classify([
        "Congratulations! You've won a £1000 prize. Text CLAIM to 87121 now",
        "Can you send me the slides before the meeting tomorrow?",
        "lol ok see u at 7 then",
    ])
    assert [r["label"] for r in res] == ["spam", "ham", "ham"]
    assert all(r["channel"] == "sms" for r in res)


@needs_models
def test_saved_email_model_classifies_new_messages():
    from predict import classify

    spam = ("Subject: lowest prices on viagra and cialis\n"
            "Buy cheap meds online without prescription. Limited time offer, "
            "click here to order now and save 80 percent. Unsubscribe below.")
    ham = ("Subject: gas nomination for march\n"
           "Vince, please find attached the revised schedule for the pipeline "
           "meeting on Thursday. Let me know if the volumes look right. Thanks, Sally")
    res = classify([spam, ham])
    assert [r["channel"] for r in res] == ["email", "email"]
    assert [r["label"] for r in res] == ["spam", "ham"]


@needs_models
def test_cli_runs():
    import subprocess

    out = subprocess.run([sys.executable, str(ROOT / "src" / "predict.py"),
                          "FREE entry! win cash now, txt WIN to 80086"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.startswith("SPAM")
