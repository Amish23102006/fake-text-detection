"""The four candidate models benchmarked in this project."""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import StylometricFeatures


def build_models(seed: int = 42, families=None):
    def pipe(clf):
        return Pipeline([
            ("features", StylometricFeatures(families=families)),
            ("scale", StandardScaler()),
            ("clf", clf),
        ])

    return {
        "LogisticRegression": pipe(LogisticRegression(max_iter=1000, random_state=seed)),
        "RandomForest": pipe(RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        "GradientBoosting": pipe(GradientBoostingClassifier(random_state=seed)),
        "SVM_RBF": pipe(SVC(kernel="rbf", C=1.0, random_state=seed)),
    }
