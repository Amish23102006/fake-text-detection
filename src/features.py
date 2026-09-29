"""Hand-engineered stylometric features for AI-vs-human text detection.

Three feature families, each targeting a different signal:

1. Sentence length      - "burstiness": humans mix short and long sentences,
                          LLM text tends to be more uniform.
2. Vocabulary richness  - how varied the word choice is (length-robust).
3. Punctuation patterns - how, and how often, punctuation is used.

No external NLP corpora are needed (no NLTK downloads): sentence and word
splitting use plain regular expressions, so the project runs offline.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")

MATTR_WINDOW = 50


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


# --------------------------------------------------------------------------
# Family 1: sentence length
# --------------------------------------------------------------------------
def sentence_length_features(text: str) -> Dict[str, float]:
    lengths = np.array([len(tokenize(s)) for s in split_sentences(text)], dtype=float)
    lengths = lengths[lengths > 0]
    if lengths.size == 0:
        return {"sent_len_mean": 0.0, "sent_len_var": 0.0, "sent_len_cv": 0.0}
    mean = lengths.mean()
    var = lengths.var()
    return {
        "sent_len_mean": float(mean),
        "sent_len_var": float(var),
        # coefficient of variation: a burstiness signal that ignores scale
        "sent_len_cv": float(np.sqrt(var) / mean) if mean else 0.0,
    }


# --------------------------------------------------------------------------
# Family 2: vocabulary richness
# --------------------------------------------------------------------------
def _mattr(tokens: List[str], window: int = MATTR_WINDOW) -> float:
    """Moving-average type-token ratio; unlike plain TTR it does not shrink
    just because a text is longer."""
    if not tokens:
        return 0.0
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens)
    ratios = [
        len(set(tokens[i : i + window])) / window
        for i in range(len(tokens) - window + 1)
    ]
    return float(np.mean(ratios))


def vocabulary_richness_features(text: str) -> Dict[str, float]:
    tokens = tokenize(text)
    if not tokens:
        return {"ttr": 0.0, "mattr": 0.0, "hapax_ratio": 0.0, "avg_word_len": 0.0}
    counts = pd.Series(tokens).value_counts()
    return {
        "ttr": len(counts) / len(tokens),
        "mattr": _mattr(tokens),
        "hapax_ratio": float((counts == 1).sum() / len(tokens)),
        "avg_word_len": float(np.mean([len(t) for t in tokens])),
    }


# --------------------------------------------------------------------------
# Family 3: punctuation patterns
# --------------------------------------------------------------------------
def punctuation_features(text: str) -> Dict[str, float]:
    n_chars = max(len(text), 1)
    n_words = max(len(tokenize(text)), 1)
    n_sents = max(len(split_sentences(text)), 1)

    def count(chars: str) -> int:
        return sum(text.count(c) for c in chars)

    punct_total = sum(1 for c in text if c in ".,;:!?-—–'\"()[]")
    return {
        "punct_density": punct_total / n_chars,
        "comma_per_sent": count(",") / n_sents,
        "semicolon_colon_per_word": count(";:") / n_words,
        "dash_per_word": count("-—–") / n_words,
        "exclaim_question_per_sent": count("!?") / n_sents,
        "quote_paren_per_word": count("\"'()[]") / n_words,
    }


FEATURE_FAMILIES = {
    "sentence_length": sentence_length_features,
    "vocabulary_richness": vocabulary_richness_features,
    "punctuation": punctuation_features,
}


def extract_features(text: str, families: Optional[List[str]] = None) -> Dict[str, float]:
    families = families or list(FEATURE_FAMILIES)
    feats: Dict[str, float] = {}
    for name in families:
        feats.update(FEATURE_FAMILIES[name](str(text)))
    return feats


class StylometricFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer: iterable of texts -> numeric feature matrix."""

    def __init__(self, families: Optional[List[str]] = None):
        self.families = families

    def fit(self, X, y=None):
        self.feature_names_ = list(extract_features("Placeholder text.", self.families))
        return self

    def transform(self, X):
        rows = [extract_features(t, self.families) for t in X]
        return pd.DataFrame(rows, columns=self.feature_names_).to_numpy()
