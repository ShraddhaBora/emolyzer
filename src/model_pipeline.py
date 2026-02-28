"""
model_pipeline.py
~~~~~~~~~~~~~~~~~
Defines the Scikit-Learn Pipeline (TF-IDF + Logistic Regression),
training logic, evaluation helpers, and inference utilities for Emolyzer.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from src.data_utils import EMOTION_MAP, EMOTION_COLORS
from src.preprocessing import NegationPreprocessor

# ─── Constants ────────────────────────────────────────────────────────────────

EMOTION_LABELS = list(EMOTION_MAP.values())   # ordered list of class names
TEST_SIZE      = 0.20
RANDOM_STATE   = 42


# ─── Pipeline Factory ─────────────────────────────────────────────────────────

def build_pipeline(max_features: int = 30_000, C: float = 1.0) -> Pipeline:
    """
    Constructs and returns a Scikit-Learn Pipeline consisting of:
      1. TfidfVectorizer  – unigrams + bigrams, sublinear TF scaling
      2. LogisticRegression – multi-class with L2 regularisation

    Args:
        max_features: vocabulary cap for TF-IDF.
        C: inverse regularisation strength for Logistic Regression.
    """
    return Pipeline([
        # Step 1: Expand contractions + mark negation (e.g. NOT_like)
        ("negation", NegationPreprocessor()),
        # Step 2: TF-IDF vectorization
        # token_pattern uses \w{2,} which captures underscores, so
        # 'NOT_like' is correctly kept as a single feature token.
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),          # unigrams + bigrams
            sublinear_tf=True,           # log normalization
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{2,}",     # \w includes _ so NOT_xxx is one token
            min_df=2,
        )),
        # Step 3: Multinomial Logistic Regression
        ("clf", LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )),
    ])


# ─── Training & Cross Validation ──────────────────────────────────────────────

def build_models(C: float = 1.0) -> dict:
    """Returns a dictionary of un-fitted underlying classifiers."""
    # We wrap SGDClassifier in CalibratedClassifierCV so it can output .predict_proba()
    svm_base = SGDClassifier(loss="hinge", max_iter=1000, random_state=RANDOM_STATE)
    
    return {
        "Logistic Regression": LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": CalibratedClassifierCV(svm_base, cv=3)
    }

def train_and_cross_validate(df: pd.DataFrame, max_features: int = 30_000, C: float = 1.0):
    """
    1. Splits data into Train (80%) and Holdout Test (20%).
    2. Runs 5-Fold Cross Validation on the Train set for 3 different algorithms.
    3. Selects the algorithm with the highest Mean CV F1-Score.
    4. Trains that 'Best Model' fully on the Train set.
    
    Returns:
        best_pipeline (Pipeline): The fully fitted winning pipeline.
        best_model_name (str): Name of the winning algorithm.
        cv_results (dict): CV statistics for all evaluated models.
        X_test, y_test: The 20% holdout set.
    """
    X = df["text"].tolist()
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Pre-vectorize the training data for faster CV (otherwise we rebuild TFIDF 5 times per model)
    print("Vectorizing training data for CV...")
    vectorizer = Pipeline([
        ("negation", NegationPreprocessor()),
        ("tfidf", TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), sublinear_tf=True,
            strip_accents="unicode", analyzer="word", token_pattern=r"\w{2,}", min_df=2
        ))
    ])
    X_train_vec = vectorizer.fit_transform(X_train)
    
    models = build_models(C=C)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    cv_results = {}
    best_model_name = None
    best_mean_f1 = -1.0
    
    print("Running 5-Fold Cross-Validation...")
    for name, clf in models.items():
        print(f"  Evaluating {name}...")
        scores = cross_validate(clf, X_train_vec, y_train, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        
        mean_acc = np.mean(scores["test_accuracy"])
        std_acc = np.std(scores["test_accuracy"])
        mean_f1 = np.mean(scores["test_f1_macro"])
        std_f1 = np.std(scores["test_f1_macro"])
        
        cv_results[name] = {
            "mean_accuracy": float(mean_acc), "std_accuracy": float(std_acc),
            "mean_f1": float(mean_f1), "std_f1": float(std_f1)
        }
        
        if mean_f1 > best_mean_f1:
            best_mean_f1 = mean_f1
            best_model_name = name
            
    print(f"Best Model selected: {best_model_name} (CV Mean F1: {best_mean_f1:.4f})")
    
    # Build complete end-to-end pipeline with the best model and train on 100% of Train data
    best_clf = models[best_model_name]
    best_pipeline = Pipeline([
        ("negation", NegationPreprocessor()),
        ("tfidf", TfidfVectorizer(
            max_features=max_features, ngram_range=(1, 2), sublinear_tf=True,
            strip_accents="unicode", analyzer="word", token_pattern=r"\w{2,}", min_df=2
        )),
        ("clf", best_clf)
    ])
    
    print(f"Fitting final {best_model_name} pipeline on full training set...")
    best_pipeline.fit(X_train, y_train)

    return best_pipeline, best_model_name, cv_results, X_test, y_test


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict:
    """
    Runs evaluation on the held-out test set.

    Returns a dict containing:
      - accuracy        (float)
      - macro_f1        (float)
      - report_dict     (dict)  – sklearn classification_report as dict
      - report_str      (str)   – human-readable classification report
      - confusion_matrix (np.ndarray)
      - class_names     (list)  – emotion labels ordered by numeric class
    """
    y_pred = pipeline.predict(X_test)

    # Determine which class indices actually appear in the test set
    present_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    class_names    = [EMOTION_MAP[i] for i in present_labels]

    report_dict = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=class_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)

    return {
        "accuracy":         float(accuracy_score(y_test, y_pred)),
        "macro_f1":         float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_f1":      float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "test_samples":     int(len(y_test)),
        "report_dict":      report_dict,
        "report_str":       report_str,
        "confusion_matrix": cm,
        "class_names":      class_names,
        "present_labels":   present_labels,
    }


def get_feature_importance(pipeline: Pipeline, top_n: int = 12) -> dict:
    """
    Extracts the top-N most informative TF-IDF features (ngrams) for each
    emotion class from the fitted classifier coefficients.
    Supports LogisticRegression, MultinomialNB, and Linear SVC (if exposed).
    """
    vectorizer   = pipeline.named_steps["tfidf"]
    classifier   = pipeline.named_steps["clf"]
    
    # Handle CalibratedClassifierCV wrap for SVM
    if hasattr(classifier, 'base_estimator') and hasattr(classifier, 'calibrated_classifiers_'):
        # Just grab the underlying learned coefficients from the first fitted fold model
        clf_model = classifier.calibrated_classifiers_[0].estimator
    else:
        clf_model = classifier
        
    if not (hasattr(clf_model, "coef_") or hasattr(clf_model, "feature_log_prob_")):
        return {"Note": [("Feature interpretation not supported for this algorithm", 1.0)]}

    feature_names = np.array(vectorizer.get_feature_names_out())
    classes       = clf_model.classes_

    importance = {}
    for idx, class_idx in enumerate(classes):
        # Naive bayes uses feature_log_prob_, others use coef_
        if hasattr(clf_model, "feature_log_prob_"):
            coefs = clf_model.feature_log_prob_[idx]
        else:
            coefs = clf_model.coef_[idx]
            
        top_indices = np.argsort(coefs)[::-1][:top_n]
        emotion_name = EMOTION_MAP.get(int(class_idx), str(class_idx))
        importance[emotion_name] = [
            (feature_names[i], float(coefs[i])) for i in top_indices
        ]
    return importance

def get_top_misclassifications(confusion_matrix: np.ndarray, class_names: list, top_k: int = 5) -> list:
    """
    Extracts the top K most common misclassifications from the confusion matrix.
    Returns: list of dicts: [{'actual': str, 'predicted': str, 'count': int}]
    """
    misclassifications = []
    
    # Iterate through off-diagonal elements
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                count = int(confusion_matrix[i, j])
                if count > 0:
                    misclassifications.append({
                        "actual": class_names[i],
                        "predicted": class_names[j],
                        "count": count
                    })
                    
    # Sort descending by count
    misclassifications.sort(key=lambda x: x["count"], reverse=True)
    return misclassifications[:top_k]


# ─── Inference ───────────────────────────────────────────────────────────────

def _normalize_oov_tokens(text: str, vocab: dict) -> str:
    """
    For each whitespace-separated token that is NOT in the TF-IDF vocabulary,
    aggressively squash all runs of 3+ identical characters down to 1 and test
    if the resulting base-form is in the vocabulary.

    Examples (given 'love' and 'hate' exist in vocab):
      'loveee'   -> 'love'
      'haateeee' -> 'hate'
      'feel'     ->  unchanged (already in vocab)
      'too'      ->  unchanged (already in vocab)
      'asdfghj'  ->  returned as-is (squashed form still not in vocab)

    This is only applied per-token, so it cannot accidentally conflate
    completely different words (e.g. 'like' never becomes 'love').
    """
    import re as _re
    tokens = text.split()
    result = []
    for token in tokens:
        # Strip punctuation to get the bare word for vocab lookup
        bare = _re.sub(r"[^\w]", "", token).lower()
        if bare in vocab:
            result.append(token)         # In-vocab: keep original
        else:
            # Squash runs of 2+ identical chars down to 1 (covers 'aa', 'eee', etc.)
            squashed = _re.sub(r"(.)\1+", r"\1", bare)
            if squashed in vocab:
                result.append(squashed)  # Normalised form found
            else:
                result.append(token)     # Truly OOV: keep original
    return " ".join(result)


def predict_emotion(pipeline: Pipeline, text: str) -> dict:
    """
    Runs inference on a single input string.

    Returns:
        dict with keys:
          - predicted_emotion (str)
          - confidence        (float, 0–1)
          - probabilities     (dict: emotion → probability)
          - is_oov            (bool) – True if input contains no known vocabulary
    """
    text = text.strip()
    vectorizer = pipeline.named_steps["tfidf"]
    vocab      = vectorizer.vocabulary_

    # First attempt: run the text through the full pipeline as-is
    # (the NegationPreprocessor is applied inside predict_proba, so we must
    # apply it manually before the OOV check which goes directly to tfidf)
    negation_step = pipeline.named_steps["negation"]
    preprocessed  = negation_step.transform([text])[0]

    vector  = vectorizer.transform([preprocessed])
    is_oov  = vector.nnz == 0   # no non-zero entries → all OOV

    if is_oov:
        # Try to rescue the text by normalising elongated tokens
        normalised = _normalize_oov_tokens(preprocessed, vocab)
        vector_norm = vectorizer.transform([normalised])
        if vector_norm.nnz > 0:
            # At least one rescued token is in-vocab – use the normalised text
            proba_array  = pipeline.named_steps["clf"].predict_proba(vector_norm)[0]
            class_indices = pipeline.classes_
            probabilities = {
                EMOTION_MAP[int(c)]: float(p)
                for c, p in zip(class_indices, proba_array)
            }
            predicted_idx    = int(np.argmax(proba_array))
            predicted_class  = int(class_indices[predicted_idx])
            predicted_emotion = EMOTION_MAP.get(predicted_class, "Unknown")
            confidence        = float(proba_array[predicted_idx])
            return {
                "predicted_emotion": predicted_emotion,
                "confidence":        confidence,
                "probabilities":     probabilities,
                "is_oov":            False,
            }
        # Genuinely unknown – nothing we can do
        return {
            "predicted_emotion": "Unknown (Out of Vocabulary)",
            "confidence": 0.0,
            "probabilities": {EMOTION_MAP[int(c)]: 0.0 for c in pipeline.classes_},
            "is_oov": True,
        }

    proba_array   = pipeline.predict_proba([text])[0]
    class_indices = pipeline.classes_

    probabilities = {
        EMOTION_MAP[int(c)]: float(p)
        for c, p in zip(class_indices, proba_array)
    }

    predicted_idx    = int(np.argmax(proba_array))
    predicted_class  = int(class_indices[predicted_idx])
    predicted_emotion = EMOTION_MAP.get(predicted_class, "Unknown")
    confidence        = float(proba_array[predicted_idx])

    return {
        "predicted_emotion": predicted_emotion,
        "confidence":        confidence,
        "probabilities":     probabilities,
        "is_oov":            is_oov,
    }
