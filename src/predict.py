"""Classify new messages as SPAM or HAM with the trained models.

Usage:
  python src/predict.py "WINNER!! Claim your free prize now, call 09061701461"
  python src/predict.py --channel email --file inbox.txt   (one message per line)
  echo "see you at lunch?" | python src/predict.py

--channel auto (default) sends short texts to the SMS model and long or
multi-line texts to the e-mail model.
"""
from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preprocess  # noqa: E402,F401  (needed so the pickled pipelines can load)

MODELS = Path(__file__).resolve().parent.parent / "models"
EMAIL_MIN_WORDS = 60


@lru_cache(maxsize=2)
def load_model(channel: str):
    path = MODELS / f"{channel}_classifier.joblib"
    if not path.exists():
        sys.exit(f"No model at {path}. Run `python src/train.py` first.")
    return joblib.load(path)


def route(message: str) -> str:
    return "email" if "\n" in message.strip() or len(message.split()) > EMAIL_MIN_WORDS else "sms"


def _score(model, text: str) -> float:
    """Signed confidence: >0 leans spam. Margin for SVM/LR, log-odds for NB."""
    if hasattr(model, "decision_function"):
        try:
            return float(model.decision_function([text])[0])
        except AttributeError:
            pass
    import math

    p = min(max(model.predict_proba([text])[0, 1], 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def classify(messages: list[str], channel: str = "auto") -> list[dict]:
    out = []
    for msg in messages:
        ch = route(msg) if channel == "auto" else channel
        model = load_model(ch)
        label = int(model.predict([msg])[0])
        out.append({"message": msg, "channel": ch,
                    "label": "spam" if label == 1 else "ham",
                    "score": round(_score(model, msg), 3)})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("messages", nargs="*")
    ap.add_argument("--file", type=Path, help="text file, one message per line")
    ap.add_argument("--channel", choices=("auto", "sms", "email"), default="auto")
    args = ap.parse_args()

    msgs = list(args.messages)
    if args.file:
        msgs += [ln.strip() for ln in args.file.read_text().splitlines() if ln.strip()]
    if not msgs and not sys.stdin.isatty():
        msgs = [ln.strip() for ln in sys.stdin if ln.strip()]
    if not msgs:
        ap.error("give at least one message")

    for r in classify(msgs, args.channel):
        preview = r["message"][:64] + ("..." if len(r["message"]) > 64 else "")
        print(f"{r['label'].upper():4s} [{r['channel']:5s}] score={r['score']:+7.3f}  {preview}")


if __name__ == "__main__":
    main()
