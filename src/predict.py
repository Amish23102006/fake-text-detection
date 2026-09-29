"""Classify text with the trained model.

    python -m src.predict --text "Some text to check ..."
    python -m src.predict --file essay.txt
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib

from .features import extract_features


def predict_text(model, text: str) -> dict:
    if hasattr(model, "predict_proba"):
        p_ai = float(model.predict_proba([text])[0][1])
    else:  # e.g. SVC without probability: squash the decision score
        p_ai = 1 / (1 + math.exp(-float(model.decision_function([text])[0])))
    return {"label": "AI-generated" if p_ai >= 0.5 else "Human-written",
            "p_ai": p_ai, "features": extract_features(text)}


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--file")
    p.add_argument("--model", default="models/best_model.joblib")
    a = p.parse_args()
    text = a.text if a.text else Path(a.file).read_text(encoding="utf-8")
    r = predict_text(joblib.load(a.model), text)
    print(f"{r['label']}  (P[AI] = {r['p_ai']:.2f})")
    print("Text is short - treat this with caution." if len(text.split()) < 50 else "")


if __name__ == "__main__":
    main()
