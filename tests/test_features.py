import numpy as np
from src.data import make_demo_data
from src.features import (StylometricFeatures, extract_features,
                          punctuation_features, sentence_length_features,
                          vocabulary_richness_features)


def test_uniform_sentences_have_zero_variance():
    r = sentence_length_features("One two three. Four five six. Seven eight nine.")
    assert r["sent_len_var"] == 0 and r["sent_len_mean"] == 3


def test_varied_sentences_have_higher_variance():
    r = sentence_length_features("Hi. This sentence is a fair bit longer than the first one was.")
    assert r["sent_len_var"] > 0 and r["sent_len_cv"] > 0


def test_repetitive_text_has_lower_richness():
    rich = vocabulary_richness_features("a b c d e f g h i j")
    poor = vocabulary_richness_features("a a a a a a a a a a")
    assert rich["ttr"] > poor["ttr"] and rich["mattr"] > poor["mattr"]


def test_punctuation_counts():
    r = punctuation_features("Wait, what? Really!")
    assert r["exclaim_question_per_sent"] == 1.0 and r["comma_per_sent"] == 0.5


def test_empty_text_does_not_crash():
    f = extract_features("")
    assert all(np.isfinite(v) for v in f.values())


def test_transformer_shape():
    df = make_demo_data(5)
    X = StylometricFeatures().fit_transform(df["text"])
    assert X.shape[0] == len(df) and np.isfinite(X).all()
