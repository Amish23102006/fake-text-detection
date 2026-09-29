from datasets import load_dataset
import pandas as pd

ds = load_dataset("inokusan/human_ai_text_classification")
df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()])
df = df[["text", "label"]].dropna()
df["n_words"] = df["text"].str.split().str.len()

print("Before:", df.shape)
print(df.groupby("label")["n_words"].describe()[["count", "mean", "50%"]])

# 1) keep multi-sentence texts of reasonable length
df = df[(df["n_words"] >= 50) & (df["n_words"] <= 600)].copy()

# 2) length-match: same number of human and AI texts in every length bin
df["bin"] = pd.cut(df["n_words"], bins=list(range(50, 650, 50)), include_lowest=True)
parts = []
for _, g in df.groupby("bin", observed=True):
    counts = g["label"].value_counts()
    if len(counts) < 2:
        continue
    n = counts.min()
    for lab in (0, 1):
        parts.append(g[g["label"] == lab].sample(n, random_state=42))
matched = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nAfter length matching:", matched.shape)
print(matched.groupby("label")["n_words"].describe()[["count", "mean", "50%"]])

matched[["text", "label"]].to_csv("data/dataset.csv", index=False)
print("Saved data/dataset.csv")