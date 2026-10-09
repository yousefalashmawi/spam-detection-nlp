"""Text preprocessing: normalisation, stop-word removal, stemming / lemmatising.

The preprocessor is a scikit-learn transformer so it can sit inside a Pipeline;
a saved model then accepts raw message text directly.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

from sklearn.base import BaseEstimator, TransformerMixin

MODES = ("raw", "clean", "stopwords", "stem", "lemma")
"""
raw        - lower-case only (baseline, no NLP preprocessing)
clean      - lower-case, URLs/e-mails/numbers/currency replaced by tokens,
             punctuation removed
stopwords  - clean + English stop words removed
stem       - stopwords + Porter stemming
lemma      - stopwords + WordNet lemmatising
"""

_URL = re.compile(r"(https?://\S+|www\.\S+)")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_PHONE = re.compile(r"\b\d{5,}\b")
_CURRENCY = re.compile(r"[£$€]")
_NUMBER = re.compile(r"\b\d+\b")
_NON_ALNUM = re.compile(r"[^a-z_ ]+")
_SPACES = re.compile(r"\s+")


def _ensure_nltk(resource: str, path: str) -> None:
    import nltk

    try:
        nltk.data.find(path)
    except LookupError:
        # The download is only needed once per machine.
        os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
        nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def stop_words() -> frozenset[str]:
    _ensure_nltk("stopwords", "corpora/stopwords")
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


@lru_cache(maxsize=1)
def _lemmatizer():
    _ensure_nltk("wordnet", "corpora/wordnet")
    _ensure_nltk("omw-1.4", "corpora/omw-1.4")
    from nltk.stem import WordNetLemmatizer

    return WordNetLemmatizer()


def normalise(text: str) -> str:
    """Lower-case and replace spam-typical entities with placeholder tokens."""
    text = str(text).lower()
    text = _URL.sub(" tok_url ", text)
    text = _EMAIL.sub(" tok_email ", text)
    text = _PHONE.sub(" tok_phone ", text)
    text = _CURRENCY.sub(" tok_money ", text)
    text = _NUMBER.sub(" tok_num ", text)
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def preprocess(text: str, mode: str = "lemma") -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "raw":
        return _SPACES.sub(" ", str(text).lower()).strip()

    tokens = normalise(text).split()
    if mode == "clean":
        return " ".join(tokens)

    sw = stop_words()
    tokens = [t for t in tokens if t not in sw and len(t) > 1]
    if mode == "stem":
        stem = _stemmer().stem
        tokens = [t if t.startswith("tok_") else stem(t) for t in tokens]
    elif mode == "lemma":
        lemma = _lemmatizer().lemmatize
        tokens = [t if t.startswith("tok_") else lemma(lemma(t), pos="v")
                  for t in tokens]
    return " ".join(tokens)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """scikit-learn wrapper around :func:`preprocess`."""

    def __init__(self, mode: str = "lemma"):
        self.mode = mode

    def fit(self, X, y=None):  # noqa: N803 (sklearn convention)
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        return self

    def transform(self, X):  # noqa: N803
        return [preprocess(t, self.mode) for t in X]
