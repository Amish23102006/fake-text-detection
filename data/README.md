# Data

Put your labeled CSV here (it is git-ignored). Required format:

| text | label |
|------|-------|
| "..." | 0 |   <- human-written
| "..." | 1 |   <- AI-generated

Column names are configurable via `--text-col` / `--label-col`.

## Suggested public datasets (check current availability and licenses)
- Kaggle: "LLM - Detect AI Generated Text" (essays, `text` + `generated`)
- Kaggle: DAIGT-style AI-vs-human essay datasets
- Hugging Face: `Hello-SimpleAI/HC3` (human vs ChatGPT answers)
- Hugging Face: GPT-wiki-intro (human intro vs generated intro per topic)

## Converting a paired dataset (one human column, one AI column)
```python
import pandas as pd
df = pd.read_csv("raw.csv")
out = pd.concat([
    pd.DataFrame({"text": df["human_col"], "label": 0}),
    pd.DataFrame({"text": df["ai_col"],    "label": 1}),
])
out.to_csv("data/dataset.csv", index=False)
```

## Watch out for shortcuts
If AI texts are systematically longer, or about different topics than human
texts, a model can score well without learning style. Check length
distributions per class and, if possible, use topic-matched pairs.
