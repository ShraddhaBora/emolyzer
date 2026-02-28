"""
preprocessing.py
~~~~~~~~~~~~~~~~
Text preprocessing utilities for Emolyzer.

Negation marking strategy:
  1. Lowercase the input.
  2. Expand contractions: "don't" → "do not", "can't" → "cannot".
  3. Walk each token; when a negation trigger word is encountered, prefix
     every subsequent token with "NOT_" until a sentence-boundary punctuation
     mark resets the scope.

Result:  "I don't like you at all."
       → "I do not NOT_like NOT_you NOT_at NOT_all."

This ensures the TF-IDF bigram "NOT_like" carries a strong negative signal,
which would otherwise be buried by the model seeing isolated "like" as joyful.
"""

import re
from sklearn.base import BaseEstimator, TransformerMixin


# ─── Contraction Expansion Table ──────────────────────────────────────────────

CONTRACTION_MAP = {
    # ── Negation contractions (most important) ────────────────────────────
    "don't":    "do not",
    "doesn't":  "does not",
    "didn't":   "did not",
    "can't":    "cannot",
    "cannot":   "cannot",
    "couldn't": "could not",
    "won't":    "will not",
    "wouldn't": "would not",
    "shouldn't":"should not",
    "shan't":   "shall not",
    "isn't":    "is not",
    "aren't":   "are not",
    "wasn't":   "was not",
    "weren't":  "were not",
    "haven't":  "have not",
    "hasn't":   "has not",
    "hadn't":   "had not",
    "needn't":  "need not",
    "mustn't":  "must not",
    "mightn't": "might not",
    "daren't":  "dare not",
    # ── Non-negation contractions ─────────────────────────────────────────
    "i'm":      "i am",
    "i've":     "i have",
    "i'll":     "i will",
    "i'd":      "i would",
    "it's":     "it is",
    "it'll":    "it will",
    "he's":     "he is",
    "she's":    "she is",
    "they're":  "they are",
    "they've":  "they have",
    "they'll":  "they will",
    "they'd":   "they would",
    "we're":    "we are",
    "we've":    "we have",
    "we'll":    "we will",
    "we'd":     "we would",
    "you're":   "you are",
    "you've":   "you have",
    "you'll":   "you will",
    "you'd":    "you would",
    "that's":   "that is",
    "there's":  "there is",
    "what's":   "what is",
    "who's":    "who is",
    "let's":    "let us",
    "could've": "could have",
    "should've":"should have",
    "would've": "would have",
    "might've": "might have",
    "must've":  "must have",
}

# Words that trigger negation marking for every subsequent token
NEGATION_TRIGGERS = frozenset({
    "not", "no", "never", "nobody", "nothing", "neither",
    "nowhere", "nor", "hardly", "scarcely", "barely", "without",
    "cannot",   # "can't" expands to "cannot" (single word), not "can not"
})

# Words that reset the negation scope (conjunctions and transition words)
SCOPE_RESET_WORDS = frozenset({
    "but", "and", "or", "because", "however", "although", "though", "while", "until"
})

# Pattern that resets the negation scope (sentence boundaries)
_SCOPE_RESET = re.compile(r"[.!?,;:]")


# ─── Core Functions ───────────────────────────────────────────────────────────

def squash_repeated_chars(text: str) -> str:
    """
    Reduces sequences of 3 or more identical characters down to 2.
    Example: "looooveeee" -> "loovee", "oooooooh" -> "oooh".
    """
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def expand_contractions(text: str) -> str:
    """
    Expands English contractions using word-boundary regex replacement.

    >>> expand_contractions("I don't like you")
    'i do not like you'
    """
    text = text.lower()
    text = squash_repeated_chars(text)
    for contraction, expansion in CONTRACTION_MAP.items():
        text = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            expansion,
            text,
        )
    return text


def mark_negations(text: str) -> str:
    """
    Full preprocessing for a single string:
      1. Lowercase + expand contractions.
      2. Mark tokens following a negation trigger with the 'NOT_' prefix.
      3. Reset negation scope at punctuation boundaries.

    >>> mark_negations("I don't like you")
    'i do not NOT_like NOT_you'

    >>> mark_negations("I am not happy but I love this.")
    'i am not NOT_happy but i love this.'
    """
    text   = expand_contractions(text)
    tokens = text.split()
    result = []
    negating = False

    for token in tokens:
        # Strip punctuation for trigger/reset checks
        clean = re.sub(r"[^\w]", "", token).lower()

        if _SCOPE_RESET.search(token) or clean in SCOPE_RESET_WORDS:
            # A sentence boundary or conjunction resets the negation scope
            negating = False
            result.append(token)
        elif clean in NEGATION_TRIGGERS:
            negating = True
            result.append(token)        # keep the trigger word itself unmodified
        elif negating and clean:
            result.append("NOT_" + clean)
        else:
            result.append(token)

    return " ".join(result)


# ─── Scikit-learn Transformer ─────────────────────────────────────────────────

class NegationPreprocessor(BaseEstimator, TransformerMixin):
    """
    Stateless Scikit-learn transformer that applies negation marking to each
    document in a corpus.

    Placing this as the FIRST step in a Pipeline guarantees that the same
    preprocessing is applied identically during fit() and predict(), preventing
    train/inference skew.
    """

    def fit(self, X, y=None):
        return self   # no internal state to learn

    def transform(self, X, y=None):
        return [mark_negations(text) for text in X]
