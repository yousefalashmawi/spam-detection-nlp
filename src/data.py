"""Dataset loading for the spam detection project.

Two public, labelled corpora are used so the system can be evaluated from a
small dataset up to a much larger one:

* SMS Spam Collection (UCI Machine Learning Repository) - 5,574 SMS messages.
* Enron-Spam (Metsis, Androutsopoulos & Paliouras, 2006) - 33,716 e-mails,
  in the preprocessed CSV form published by M. Wiechmann on GitHub.

Labels are encoded as 1 = spam, 0 = ham (legitimate).
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SMS_URL = (
    "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/"
    "master/data/sms.tsv"
)
ENRON_URL = (
    "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/"
    "master/enron_spam_data.zip"
)


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (fixed URLs)
        return resp.read()


def ensure_sms(data_dir: Path = DATA_DIR) -> Path:
    path = data_dir / "sms.tsv"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_download(SMS_URL))
    return path


def ensure_enron(data_dir: Path = DATA_DIR) -> Path:
    path = data_dir / "enron_spam_data.csv"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(_download(ENRON_URL))) as zf:
            zf.extract("enron_spam_data.csv", data_dir)
    return path


def _finalise(df: pd.DataFrame, source: str, dedupe: bool) -> pd.DataFrame:
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0]
    if dedupe:
        # Exact duplicates would otherwise leak between train and test splits
        # and inflate the reported scores.
        df = df.drop_duplicates(subset="text")
    df = df.assign(source=source).reset_index(drop=True)
    return df[["text", "label", "source"]]


def load_sms(dedupe: bool = True, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return the SMS Spam Collection as columns text, label (0/1), source."""
    path = ensure_sms(data_dir)
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                     quoting=3, encoding="utf-8")
    df["label"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    return _finalise(df, "sms", dedupe)


def load_enron(dedupe: bool = True, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return Enron-Spam with subject and body joined into one text field."""
    path = ensure_enron(data_dir)
    df = pd.read_csv(path)
    subject = df["Subject"].fillna("")
    body = df["Message"].fillna("")
    df = pd.DataFrame({
        "text": (subject + " " + body).str.strip(),
        "label": (df["Spam/Ham"].str.strip().str.lower() == "spam").astype(int),
    })
    return _finalise(df, "enron", dedupe)


def load_combined(dedupe: bool = True) -> pd.DataFrame:
    return pd.concat([load_sms(dedupe), load_enron(dedupe)], ignore_index=True)


def describe(df: pd.DataFrame) -> dict:
    """Small summary used in the report (size, class balance, text length)."""
    lengths = df["text"].str.split().str.len()
    return {
        "messages": int(len(df)),
        "spam": int(df["label"].sum()),
        "ham": int((df["label"] == 0).sum()),
        "spam_share": round(float(df["label"].mean()), 4),
        "median_tokens": float(lengths.median()),
        "mean_tokens": round(float(lengths.mean()), 1),
    }
