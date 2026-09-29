# Fake Text Detection: AI-generated vs Human-written

A binary NLP classifier that distinguishes AI-generated from human-written text using hand-engineered stylometric features and classical ML. It benchmarks four models with stratified cross-validation, evaluates the winner once on a held-out test set, and includes an ablation showing which feature family carries the signal.

**Headline result:** ~83% accuracy (82.9%, 95% CI about ±2.6 points) on a held-out, length-matched test set of 800 texts.

## Approach

**Three feature families** (`src/features.py`)

| Family | Features | Idea |
|---|---|---|
| Sentence length | mean, variance, coefficient of variation | Humans are "burstier"; LLM text is more uniform |
| Vocabulary richness | TTR, moving-average TTR, hapax ratio, avg word length | Word-choice variety, robust to text length |
| Punctuation patterns | density, commas/sentence, `;:`, dashes, `!?`, quotes/parens | Stylistic punctuation habits |

**Four models** (`src/models.py`): Logistic Regression, Random Forest, Gradient Boosting, RBF-SVM. Each is a pipeline: features -> StandardScaler -> classifier.

**Evaluation protocol** (`src/train.py`)
1. Split off a stratified 20% test set before anything else.
2. 5-fold stratified CV on the training part: accuracy, precision, recall, F1.
3. Pick the best model by mean CV F1.
4. Refit on all training data; evaluate **once** on the test set.
5. Ablation: CV F1 using each feature family alone.

No NLTK downloads are needed; tokenization is regex-based, so it runs offline.

## Data

Public dataset [`inokusan/human_ai_text_classification`](https://huggingface.co/datasets/inokusan/human_ai_text_classification) (Hugging Face), which merges several public human-vs-AI sources into a balanced set.

The raw data had human texts noticeably longer than AI texts (median 176 vs 118 words), and some texts were a single word. Because vocabulary features shift with length, `prepare_data.py`:
1. keeps texts of 50-600 words, and
2. length-matches the classes (equal numbers of human and AI texts in every 50-word bin).

The experiment below used a balanced random sample of 4,000 texts from the matched set (3,200 train / 800 test).

## Results

**Cross-validation** (5-fold, training split, mean ± std):

| Model | CV F1 | CV Precision | CV Recall |
|---|---|---|---|
| Logistic Regression | _fill from results/cv_results.csv_ | | |
| Random Forest | 0.838 ± 0.009 | 0.831 | 0.845 |
| Gradient Boosting | 0.835 ± 0.008 | 0.824 | 0.847 |
| **RBF-SVM (selected)** | **0.840 ± 0.009** | 0.836 | 0.844 |

**Held-out test set** (RBF-SVM, n = 800, evaluated once): accuracy **0.829**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Human | 0.856 | 0.790 | 0.822 |
| AI | 0.805 | 0.868 | 0.835 |

The model catches AI text slightly more reliably than human text (recall 0.868 vs 0.790).

**Ablation** (CV F1, RBF-SVM, one feature family at a time):

| Feature family | CV F1 |
|---|---|
| Sentence length | 0.745 |
| Punctuation | 0.730 |
| Vocabulary richness | 0.643 |
| **All three combined** | **0.840** |

Each family carries signal alone, and combining them gives a large gain, so they capture complementary information.

## Quick start

```bash
pip install -r requirements.txt
pytest                                   # unit tests (or: python -m pytest)
python -m src.train --demo               # pipeline smoke test (SYNTHETIC data)

# reproduce the results
python prepare_data.py                   # download, filter, length-match -> data/dataset.csv
python -m src.train --data data/dataset.csv --max-samples 4000

python -m src.predict --text "Paste some text here ..."
streamlit run app.py                     # web demo
```

`--demo` uses trivially separable synthetic text (it scores ~100%) and exists only to verify the code runs. **Never report demo numbers.**

## Limitations

- Stylometric features are a weak proxy for authorship: fluent human writers and edited or paraphrased AI text can be misclassified.
- The dataset merges several sources. If human and AI texts came from different sources, some accuracy may reflect source differences rather than writing style. Length matching removes one shortcut, not all of them.
- Results are from a single run and seed on a 4,000-text sample, and depend on the dataset (generator model, topic, length, domain). They may not transfer to other LLMs or genres.
- Very short texts give unreliable feature estimates.
- Not suitable as sole evidence for academic-integrity or hiring decisions.

## Project layout

```
prepare_data.py   download + filter + length-match the dataset
src/features.py   feature engineering (3 families)
src/models.py     4 candidate models
src/data.py       dataset loading + synthetic demo data
src/train.py      CV benchmark, test evaluation, ablation, saving
src/predict.py    CLI inference
app.py            Streamlit demo
tests/            unit tests
```
