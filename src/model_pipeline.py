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
            stop_words="english",        # Drop neutral grammar/filler words
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
            strip_accents="unicode", analyzer="word", token_pattern=r"\w{2,}", 
            stop_words="english", min_df=2
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
            strip_accents="unicode", analyzer="word", token_pattern=r"\w{2,}",
            stop_words="english", min_df=2
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

_spell = None
_synonym_cache = {}

def _get_spellchecker():
    global _spell
    if _spell is None:
        try:
            from spellchecker import SpellChecker
            _spell = SpellChecker()
        except ImportError:
            pass
    return _spell

def _get_synonyms(word: str) -> set:
    if word in _synonym_cache:
        return _synonym_cache[word]
    
    synonyms = set()
    try:
        from nltk.corpus import wordnet
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                name = l.name().lower().replace("_", "")
                synonyms.add(name)
    except Exception:
        pass
        
    _synonym_cache[word] = synonyms
    return synonyms

def _normalize_oov_tokens(text: str, vocab: dict) -> str:
    """
    Attempts to normalize an unknown text sequence by:
    1. Looking up direct match in vocab (FAST EXIT).
    2. Squashing elongated characters (e.g. 'loovveee' -> 'love')
    3. Checking for spelling corrections against common words.
    4. Looking up synonyms via WordNet that exist in the training vocabulary.
    """
    import re as _re
    tokens = text.split()
    result = []
    
    # 0. Check if entire text features are already in vocab (Very Fast Exit)
    # If most tokens are known, we can skip the heavy per-token logic
    known_count = sum(1 for t in tokens if _re.sub(r"[^\w]", "", t).lower() in vocab)
    if known_count == len(tokens):
        return text

    spell = _get_spellchecker()
    
    for token in tokens:
        # Strip punctuation to get the bare word
        bare = _re.sub(r"[^\w]", "", token).lower()
        if not bare:
            result.append(token)
            continue
            
        # 1. Direct match (Fast per-token Exit)
        if bare in vocab:
            result.append(token)
            continue
            
        candidates = []
        
        # 2. Squash runs of 2+ identical chars (e.g. loovveee -> love)
        squashed = _re.sub(r"(.)\1+", r"\1", bare)
        if squashed:
            candidates.append(squashed)
        
        # 3. Spelling correction
        if spell:
            # Check if squashed is in vocab before heavy spellcorrection
            if squashed in vocab:
                result.append(squashed)
                continue
                
            corr_bare = spell.correction(bare)
            if corr_bare:
                candidates.append(corr_bare)
            
            corr_squash = spell.correction(squashed)
            if corr_squash:
                candidates.append(corr_squash)
            
        found_in_vocab = False
        for cand in candidates:
            if cand in vocab:
                result.append(cand)
                found_in_vocab = True
                break
                
        if found_in_vocab:
            continue
            
        # 4. Synonym Search (Extremely slow - only if others fail)
        all_to_check = set([bare] + candidates)
        found_synonym = False
        
        for word_to_check in all_to_check:
            syns = _get_synonyms(word_to_check)
            for syn in syns:
                if syn in vocab:
                    result.append(syn)
                    found_synonym = True
                    break
            if found_synonym:
                break
                
        if found_synonym:
            continue
            
        # 5. Genuinely unknown – keep original
        result.append(token)
        
    return " ".join(result)


def _augment_with_synonyms(text: str, vocab: dict) -> str:
    """
    For each token in `text` that exists in the TF-IDF vocabulary, look up its
    WordNet synonyms and inject those that also exist in the vocabulary.
    This gives the classifier richer signal at inference time.
    
    OPTIMIZATION: Skip if text is too long (NLTK is heavy) or if certain tokens
    already provide a very high signal.
    """
    import re as _re
    
    tokens = text.split()
    if len(tokens) > 20: # Threshold: Don't augment long sentences for speed
        return text

    try:
        from nltk.corpus import wordnet
    except Exception:
        return text   # NLTK not available — skip silently

    extras = []

    for token in tokens:
        bare = _re.sub(r"[^\w]", "", token).lower()
        if not bare:
            continue

        # Handle NOT_word → look up antonyms of `word`
        if bare.startswith("not_"):
            base_word = bare[4:]
            try:
                for syn in wordnet.synsets(base_word):
                    for lemma in syn.lemmas():
                        for ant in lemma.antonyms():
                            ant_word = ant.name().lower().replace("_", "")
                            if ant_word in vocab and ant_word not in extras:
                                extras.append(ant_word)
                                if len(extras) > 10: break # Cap extras per sentence
            except Exception:
                pass
        else:
            # Regular word → look up synonyms
            if bare not in vocab:
                continue   # don't bother for OOV tokens (handled elsewhere)
            try:
                for syn in wordnet.synsets(bare):
                    for lemma in syn.lemmas():
                        syn_word = lemma.name().lower().replace("_", "")
                        if syn_word in vocab and syn_word != bare and syn_word not in extras:
                            extras.append(syn_word)
                            if len(extras) > 10: break # Cap extras per sentence
            except Exception:
                pass
        if len(extras) > 30: break # Absolute cap to prevent slowdown

    if extras:
        return text + " " + " ".join(extras)
    return text


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
    vectorizer    = pipeline.named_steps["tfidf"]
    vocab         = vectorizer.vocabulary_
    negation_step = pipeline.named_steps["negation"]
    clf           = pipeline.named_steps["clf"]

    # 1. Apply negation logic + intensity boost injection
    preprocessed = negation_step.transform([text])[0]

    # 2. Normalize elongated OOV tokens (e.g. "loveeeee" → "love")
    normalised = _normalize_oov_tokens(preprocessed, vocab)

    # 3. Augment with WordNet synonyms/antonyms for richer signal
    augmented = _augment_with_synonyms(normalised, vocab)

    # 4. Convert into TF-IDF vector space
    vector = vectorizer.transform([augmented])
    is_oov = vector.nnz == 0   # no non-zero entries → all OOV

    if is_oov:
        return {
            "predicted_emotion": "Unknown (Out of Vocabulary)",
            "confidence": 0.0,
            "probabilities": {EMOTION_MAP[int(c)]: 0.0 for c in pipeline.classes_},
            "is_oov": True,
        }

    proba_array   = clf.predict_proba(vector)[0]
    class_indices = pipeline.classes_

    probabilities = {
        EMOTION_MAP[int(c)]: float(p)
        for c, p in zip(class_indices, proba_array)
    }

    predicted_idx     = int(np.argmax(proba_array))
    predicted_class   = int(class_indices[predicted_idx])
    predicted_emotion = EMOTION_MAP.get(predicted_class, "Unknown")
    confidence        = float(proba_array[predicted_idx])

    return {
        "predicted_emotion": predicted_emotion,
        "confidence":        confidence,
        "probabilities":     probabilities,
        "is_oov":            False,
    }

