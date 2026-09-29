# Fake Text Detection: AI-generated vs Human-written

A binary NLP classifier that distinguishes AI-generated from human-written text
using **hand-engineered stylometric features** and classical ML. It benchmarks
four models with stratified cross-validation, evaluates the winner once on a
held-out test set, and includes an ablation showing which feature family carries
the signal.

## Approach

**Three feature families** (`src/features.py`)

| Family | Features | Idea |
|---|---|---|
| Sentence length | mean, variance, coefficient of variation | Humans are "burstier"; LLM text is more uniform |
| Vocabulary richness | TTR, moving-average TTR, hapax ratio, avg word length | Word-choice variety, robust to text length |
| Punctuation patterns | density, commas/sentence, `;:`, dashes, `!?`, quotes/parens | Stylistic punctuation habits |

**Four models** (`src/models.py`): Logistic Regression, Random Forest,
Gradient Boosting, RBF-SVM. Each is a pipeline: features -> StandardScaler -> classifier.

**Evaluation protocol** (`src/train.py`)
1. Split off a stratified 20% test set *before* anything else.
2. 5-fold stratified CV on the training part: accuracy, precision, recall, F1.
3. Pick the best model by mean CV F1.
4. Refit on all training data; evaluate **once** on the test set.
5. Ablation: CV F1 using each feature family alone.

No NLTK downloads are needed; tokenization is regex-based, so it runs offline.

## Quick start

```bash
pip install -r requirements.txt
pytest                                   # unit tests
python -m src.train --demo               # pipeline smoke test (SYNTHETIC data)
python -m src.train --data data/dataset.csv
python -m src.predict --text "Paste some text here ..."
streamlit run app.py                     # web demo
```

`--demo` uses trivially separable synthetic text (it scores ~100%) and exists
only to verify the code runs. **Never report demo numbers.**

## Results

*Fill this in from your own run on a real dataset (`results/cv_results.csv`,
`results/test_report.txt`, `results/ablation_f1.json`). Report the dataset
name and size, and use test-set numbers as the headline figure.*

| Model | CV Accuracy | CV Precision | CV Recall | CV F1 |
|---|---|---|---|---|
| ... | | | | |

Held-out test accuracy: `__`

## Limitations

- Stylometric features are a weak proxy for authorship: fluent human writers
  and edited or paraphrased AI text can be misclassified.
- Results depend heavily on the dataset (generator model, topic, length, domain)
  and may not transfer to other LLMs or genres.
- Very short texts give unreliable feature estimates.
- Not suitable as sole evidence for academic-integrity or hiring decisions.

## Project layout

```
src/features.py   feature engineering (3 families)
src/models.py     4 candidate models
src/data.py       dataset loading + synthetic demo data
src/train.py      CV benchmark, test evaluation, ablation, saving
src/predict.py    CLI inference
app.py            Streamlit demo
tests/            unit tests
```