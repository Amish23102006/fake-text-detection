"""Dataset loading.

Real results require a real labeled dataset (see README). `make_demo_data`
builds a small SYNTHETIC dataset that exists only to smoke-test the pipeline;
any score measured on it says nothing about real-world performance.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import pandas as pd

LABEL_HUMAN, LABEL_AI = 0, 1


def load_dataset(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Load a CSV with a text column and a label column (0 = human, 1 = AI)."""
    df = pd.read_csv(Path(path))
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"Columns {missing} not found. Available: {list(df.columns)}")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df = df.dropna().drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    if set(df["label"].unique()) - {0, 1}:
        raise ValueError("Labels must be 0 (human) or 1 (AI-generated).")
    return df


def split_xy(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    return df["text"], df["label"]


# ---------------------------------------------------------------- demo data
_TOPICS = ["the city", "my school", "this project", "the weather", "the market",
           "our team", "the old bridge", "that meeting", "the new phone", "her garden"]
_AI_OPENERS = ["Overall,", "In conclusion,", "Furthermore,", "Additionally,", "Moreover,"]
_AI_BODY = ["plays an important role in modern society", "offers several key benefits",
            "presents both opportunities and challenges", "continues to evolve over time",
            "is essential for sustainable long-term growth"]
_HUMAN_BITS = ["honestly", "I mean", "kinda", "no idea why", "still", "weirdly enough",
               "for what it's worth", "anyway"]


def _ai_text(rng: random.Random) -> str:
    sents = []
    for _ in range(rng.randint(4, 6)):
        t = rng.choice(_TOPICS)
        sents.append(f"{rng.choice(_AI_OPENERS)} {t} {rng.choice(_AI_BODY)}.")
    return " ".join(sents)


def _human_text(rng: random.Random) -> str:
    sents = []
    for _ in range(rng.randint(3, 7)):
        t = rng.choice(_TOPICS)
        n = rng.choice([2, 3, 5, 9, 14])
        filler = " ".join(rng.choice(_HUMAN_BITS) for _ in range(n // 3 + 1))
        end = rng.choice([".", "!", "?", "...", " - well,"])
        sents.append(f"{filler}, {t} was a lot{end}".replace(", ,", ","))
    return " ".join(sents)


def make_demo_data(n_per_class: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = [{"text": _human_text(rng), "label": LABEL_HUMAN} for _ in range(n_per_class)]
    rows += [{"text": _ai_text(rng), "label": LABEL_AI} for _ in range(n_per_class)]
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df
