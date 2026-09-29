"""Benchmark 4 models with stratified cross-validation, pick the best by F1,
evaluate it once on a held-out test set, and save everything.

Usage:
    python -m src.train --data data/your_dataset.csv
    python -m src.train --demo            # synthetic smoke test only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from .data import load_dataset, make_demo_data, split_xy
from .features import FEATURE_FAMILIES
from .models import build_models

SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"}


def run(args) -> dict:
    out_models, out_results = Path(args.models_dir), Path(args.results_dir)
    out_models.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    df = make_demo_data(seed=args.seed) if args.demo else load_dataset(
        args.data, args.text_col, args.label_col)
    if args.max_samples and len(df) > args.max_samples:
        per_class = args.max_samples // 2
        df = pd.concat([
            df[df["label"] == lab].sample(
                min(per_class, int((df["label"] == lab).sum())), random_state=args.seed)
            for lab in (0, 1)
        ]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    X, y = split_xy(df)
    print(f"Dataset: {len(df)} texts | class balance: {y.value_counts().to_dict()}")

    # Hold out a test set FIRST; cross-validation only ever sees the training part.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    for name, model in build_models(args.seed).items():
        res = cross_validate(model, X_tr, y_tr, cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for m in SCORING:
            row[f"{m}_mean"] = res[f"test_{m}"].mean()
            row[f"{m}_std"] = res[f"test_{m}"].std()
        rows.append(row)
        print(f"  {name:20s} F1={row['f1_mean']:.3f} ± {row['f1_std']:.3f}  "
              f"P={row['precision_mean']:.3f}  R={row['recall_mean']:.3f}")

    cv_df = pd.DataFrame(rows).sort_values("f1_mean", ascending=False)
    cv_df.to_csv(out_results / "cv_results.csv", index=False)
    best_name = cv_df.iloc[0]["model"]
    print(f"\nBest model by CV F1: {best_name}")

    # Final fit on all training data, single evaluation on held-out test set.
    best = build_models(args.seed)[best_name].fit(X_tr, y_tr)
    pred = best.predict(X_te)
    report = classification_report(y_te, pred, target_names=["human", "ai"], digits=3)
    print("\nHeld-out test report:\n" + report)
    (out_results / "test_report.txt").write_text(f"Best model: {best_name}\n\n{report}")

    ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["human", "ai"]).plot()
    plt.title(f"{best_name} - held-out test")
    plt.savefig(out_results / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Ablation: which feature family matters? (best model, CV on train split)
    ablation = {}
    for fam in FEATURE_FAMILIES:
        m = build_models(args.seed, families=[fam])[best_name]
        ablation[fam] = float(cross_validate(m, X_tr, y_tr, cv=cv, scoring="f1")["test_score"].mean())
    ablation["all_families"] = float(cv_df.iloc[0]["f1_mean"])
    (out_results / "ablation_f1.json").write_text(json.dumps(ablation, indent=2))
    print("Ablation (CV F1 per feature family):", json.dumps(ablation, indent=2))

    joblib.dump(best, out_models / "best_model.joblib")
    return {"best": best_name, "cv": cv_df, "ablation": ablation, "demo": args.demo}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", help="CSV with text and label columns (0=human, 1=AI)")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="label")
    p.add_argument("--demo", action="store_true", help="synthetic data, pipeline smoke test only")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--max-samples", type=int, default=0, help="cap dataset size (balanced)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models-dir", default="models")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    if not args.demo and not args.data:
        p.error("provide --data <csv> or use --demo")
    if args.demo:
        print("*** DEMO MODE: synthetic data. Scores are NOT real-world results. ***")
    run(args)


if __name__ == "__main__":
    main()