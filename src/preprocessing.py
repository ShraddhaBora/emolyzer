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
    "don't": "do not",
    "dont": "do not",
    "doesn't": "does not",
    "doesnt": "does not",
    "didn't": "did not",
    "didnt": "did not",
    "can't": "cannot",
    "cant": "cannot",
    "cannot": "cannot",
    "couldn't": "could not",
    "couldnt": "could not",
    "won't": "will not",
    "wont": "will not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "shouldn't": "should not",
    "shouldnt": "should not",
    "shan't": "shall not",
    "isn't": "is not",
    "isnt": "is not",
    "aren't": "are not",
    "arent": "are not",
    "wasn't": "was not",
    "wasnt": "was not",
    "weren't": "were not",
    "werent": "were not",
    "haven't": "have not",
    "havent": "have not",
    "hasn't": "has not",
    "hasnt": "has not",
    "hadn't": "had not",
    "hadnt": "had not",
    "needn't": "need not",
    "mustn't": "must not",
    "mustnt": "must not",
    "mightn't": "might not",
    "daren't": "dare not",
    # ── Non-negation contractions ─────────────────────────────────────────
    "i'm": "i am",
    "im": "i am",
    "i've": "i have",
    "ive": "i have",
    "i'll": "i will",
    "ill": "i will",  # Note: can overlap with sick "ill", but contextually okay for simple naive expansion
    "i'd": "i would",
    "id": "i would",  # Note: overlaps with identity "id"
    "it's": "it is",
    "its": "it is",  # Technically "it is" vs possessive, but useful for basic tokenization
    "it'll": "it will",
    "itll": "it will",
    "he's": "he is",
    "hes": "he is",
    "she's": "she is",
    "shes": "she is",
    "they're": "they are",
    "theyre": "they are",
    "they've": "they have",
    "theyve": "they have",
    "they'll": "they will",
    "theyll": "they will",
    "they'd": "they would",
    "theyd": "they would",
    "we're": "we are",
    "were": "we are",  # Overlaps with past tense "were", risky, but common in informal chat
    "we've": "we have",
    "weve": "we have",
    "we'll": "we will",
    "well": "we will",  # Overlaps with "well"
    "we'd": "we would",
    "wed": "we would",
    "you're": "you are",
    "youre": "you are",
    "you've": "you have",
    "youve": "you have",
    "you'll": "you will",
    "youll": "you will",
    "you'd": "you would",
    "youd": "you would",
    "that's": "that is",
    "thats": "that is",
    "there's": "there is",
    "theres": "there is",
    "what's": "what is",
    "whats": "what is",
    "who's": "who is",
    "whos": "who is",
    "let's": "let us",
    "lets": "let us",
    "could've": "could have",
    "couldve": "could have",
    "should've": "should have",
    "shouldve": "should have",
    "would've": "would have",
    "wouldve": "would have",
    "might've": "might have",
    "mightve": "might have",
    "must've": "must have",
    "mustve": "must have",
    "ain't": "is not",
    "aint": "is not",
}

# Words that trigger negation marking for every subsequent token
NEGATION_TRIGGERS = frozenset(
    {
        "not",
        "no",
        "never",
        "nobody",
        "nothing",
        "neither",
        "nowhere",
        "nor",
        "hardly",
        "scarcely",
        "barely",
        "without",
        "cannot",  # "can't" expands to "cannot" (single word), not "can not"
    }
)

# Words that reset the negation scope (conjunctions and transition words)
SCOPE_RESET_WORDS = frozenset(
    {
        "but",
        "and",
        "or",
        "because",
        "however",
        "although",
        "though",
        "while",
        "until",
        "that",
        "so",
        "which",
        "when",
        "after",
        "since",
    }
)

# Direct antonym mappings for negated common emotion words
# This helps the TF-IDF model see a strong directional signal instead of just OOV 'NOT_word'.
ANTONYM_MAP = {
    "like": ["dislike", "hate"],
    "love": ["hate", "despise"],
    "happy": ["sad", "unhappy", "depressed"],
    "good": ["bad", "awful"],
    "great": ["terrible", "awful"],
    "enjoy": ["dislike", "hate"],
    "care": ["apathy", "ignore"],
    "want": ["reject", "refuse"],
    "safe": ["fear", "danger"],
    "calm": ["anger", "panic"],
    "trust": ["fear", "doubt"],
    "excited": ["bored", "dull"],
    "fun": ["boring", "sad"],
    "funny": ["serious", "unfunny"],
    "best": ["worst"],
    "sure": ["doubt", "fear"],
    # "didn't see/expect/know that coming" idiom → surprise
    "see": ["surprised", "unexpected", "surprise", "shocked"],
    "expect": ["surprised", "unexpected", "surprise"],
    "knew": ["surprised", "unexpected", "surprise"],
    "know": ["surprised", "unexpected"],
    "believe": ["shocked", "surprised", "disbelief", "unexpected"],
    "imagine": ["shocked", "surprised", "unexpected"],
}


# Pattern that resets the negation scope (sentence boundaries)
_SCOPE_RESET = re.compile(r"[.!?,;:]")

# High-intensity emotion words that get their synonyms injected to boost the signal.
# Key = word as it appears after lowercasing/contraction expansion.
# Value = extra tokens to inject right after the word.
INTENSITY_BOOST_MAP = {
    # ── Anger ────────────────────────────────────────────────
    "infuriating": ["anger", "angry", "rage", "furious"],
    "infuriated": ["anger", "angry", "rage", "furious"],
    "furious": ["anger", "angry", "rage"],
    "outrageous": ["anger", "angry", "outrage"],
    "outraged": ["anger", "angry", "outrage"],
    "enraged": ["anger", "angry", "rage", "furious"],
    "disgusting": ["anger", "disgust", "hate"],
    "disgusted": ["anger", "disgust", "hate"],
    "hate": ["anger", "hate", "furious", "dislike"],
    "hated": ["anger", "hate"],
    "hating": ["anger", "hate"],
    "despise": ["anger", "hate", "disgust"],
    "despised": ["anger", "hate", "disgust"],
    "loathe": ["anger", "hate", "disgust"],
    "loathing": ["anger", "hate", "disgust"],
    "irate": ["anger", "angry", "furious"],
    "livid": ["anger", "angry", "furious", "rage"],
    "seething": ["anger", "rage", "furious"],
    "frustrated": ["anger", "angry", "frustrated"],
    "frustrating": ["anger", "angry", "frustrated"],
    "annoyed": ["anger", "annoyed", "angry"],
    "annoying": ["anger", "annoyed", "angry"],
    "bitter": ["anger", "sad"],
    "irritated": ["anger", "annoyed", "angry"],
    "irritating": ["anger", "annoyed", "angry"],
    "aggravated": ["anger", "angry", "frustrated"],
    "hostile": ["anger", "angry"],
    # ── Sadness ──────────────────────────────────────────────
    "heartbroken": ["sad", "sadness", "grief", "miserable"],
    "devastated": ["sad", "sadness", "grief"],
    "miserable": ["sad", "sadness", "unhappy"],
    "depressed": ["sad", "sadness", "grief", "unhappy"],
    "hopeless": ["sad", "sadness", "despair"],
    "grief": ["sad", "sadness", "sorrow"],
    "mourning": ["sad", "sadness", "grief"],
    "miss": ["sad", "sadness", "longing", "lonely"],
    "missing": ["sad", "sadness", "longing", "lonely"],
    "lonely": ["sad", "sadness", "alone"],
    "alone": ["sad", "sadness", "lonely"],
    "crying": ["sad", "sadness", "tears", "grief"],
    "cried": ["sad", "sadness", "tears"],
    "tears": ["sad", "sadness", "crying"],
    "weeping": ["sad", "sadness", "crying", "grief"],
    "empty": ["sad", "sadness", "lonely", "numb"],
    "numb": ["sad", "sadness", "empty"],
    "exhausted": ["sad", "sadness", "tired"],
    "drained": ["sad", "sadness", "exhausted"],
    "disappointed": ["sad", "sadness", "unhappy", "dejected"],
    "regret": ["sad", "sadness", "guilt"],
    "regrets": ["sad", "sadness", "guilt"],
    "hurts": ["sad", "sadness", "pain"],
    "hurting": ["sad", "sadness", "pain"],
    "painful": ["sad", "sadness", "pain"],
    "sorrow": ["sad", "sadness", "grief"],
    "gloomy": ["sad", "sadness", "unhappy"],
    "melancholy": ["sad", "sadness", "sorrow"],
    # ── Joy ──────────────────────────────────────────────────
    "ecstatic": ["joy", "happy", "elated", "excited"],
    "elated": ["joy", "happy", "excited"],
    "overjoyed": ["joy", "happy", "excited"],
    "thrilled": ["joy", "happy", "excited"],
    "delighted": ["joy", "happy", "pleased"],
    "exhilarated": ["joy", "happy", "excited"],
    "wonderful": ["joy", "happy", "amazing", "great"],
    "amazing": ["joy", "happy", "excited", "great"],
    "fantastic": ["joy", "happy", "great", "excited"],
    "magnificent": ["joy", "happy", "great"],
    "joyful": ["joy", "happy", "excited"],
    "joyous": ["joy", "happy", "excited"],
    "laughing": ["joy", "happy", "funny"],
    "laughed": ["joy", "happy", "funny"],
    "laugh": ["joy", "happy", "funny"],
    "smile": ["joy", "happy", "pleased"],
    "smiling": ["joy", "happy", "pleased"],
    "grinning": ["joy", "happy", "pleased"],
    "blessed": ["joy", "happy", "grateful", "love"],
    "grateful": ["joy", "happy", "love", "thankful"],
    "thankful": ["joy", "happy", "love", "grateful"],
    "proud": ["joy", "happy"],
    "celebrating": ["joy", "happy", "excited"],
    "celebrate": ["joy", "happy", "excited"],
    "like": ["joy", "happy"],
    "liked": ["joy", "happy"],
    "loving": ["love", "joy", "happy"],
    "enjoy": ["joy", "happy"],
    "enjoyed": ["joy", "happy"],
    "really": ["joy", "happy"],
    "very": ["joy", "happy"],
    # ── Fear ─────────────────────────────────────────────────
    "terrified": ["fear", "scared", "afraid"],
    "petrified": ["fear", "scared", "terrified"],
    "horrified": ["fear", "scared", "horror"],
    "dreading": ["fear", "anxiety", "scared"],
    "dread": ["fear", "anxiety", "scared"],
    "panicking": ["fear", "anxiety", "panic"],
    "panic": ["fear", "anxiety", "scared"],
    "anxious": ["fear", "anxiety", "worried", "scared"],
    "anxiety": ["fear", "anxiety", "worried"],
    "worried": ["fear", "worry", "anxious"],
    "worrying": ["fear", "worry", "anxious"],
    "nervous": ["fear", "anxiety", "nervous"],
    "scared": ["fear", "scared", "afraid"],
    "frightened": ["fear", "scared", "afraid"],
    "afraid": ["fear", "scared"],
    "nightmare": ["fear", "horror", "scared"],
    "horrifying": ["fear", "horror", "scared"],
    "trembling": ["fear", "scared", "anxiety"],
    # ── Surprise ─────────────────────────────────────────────
    "wow": ["surprise", "shocked", "amazed", "unexpected"],
    "whoa": ["surprise", "shocked", "amazed"],
    "woah": ["surprise", "shocked", "amazed"],
    "omg": ["surprise", "shocked", "amazed", "unexpected"],
    "unexpected": ["surprise", "shocked", "unbelievable"],
    "unbelievable": ["surprise", "shocked", "amazed"],
    "shocking": ["surprise", "shocked", "unbelievable"],
    "incredible": ["surprise", "amazed", "unbelievable"],
    "astonished": ["surprise", "shocked", "amazed"],
    "astounded": ["surprise", "shocked", "amazed"],
    "stunned": ["surprise", "shocked"],
    "flabbergasted": ["surprise", "shocked", "amazed"],
    "mindblown": ["surprise", "shocked", "amazed"],
    "mindblowing": ["surprise", "shocked", "amazed"],
    "suddenly": ["surprise", "unexpected"],
    "unaware": ["surprise", "unexpected"],
    # ── Love ─────────────────────────────────────────────────
    "adore": ["love", "affection", "adoration"],
    "adoring": ["love", "affection"],
    "cherish": ["love", "affection"],
    "devoted": ["love", "affection"],
    "affection": ["love", "affection", "care"],
    "compassion": ["love", "care", "kindness"],
    "warmth": ["love", "affection", "care"],
    "caring": ["love", "affection", "care"],
    "tender": ["love", "affection"],
    "adoration": ["love", "affection"],
    "love": ["love", "joy", "happy"],
    "loved": ["love", "joy", "happy"],
}


# ─── Core Functions ───────────────────────────────────────────────────────────


def squash_repeated_chars(text: str) -> str:
    """
    Reduces sequences of 3 or more identical characters down to 2.
    Example: "looooveeee" -> "loovee", "oooooooh" -> "oooh".
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


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
      4. Inject synonyms for high-intensity emotion words (INTENSITY_BOOST_MAP).

    >>> mark_negations("I don't like you")
    'i do not NOT_like NOT_you'

    >>> mark_negations("I am not happy but I love this.")
    'i am not NOT_happy but i love this.'
    """
    text = expand_contractions(text)
    tokens = text.split()
    result = []
    negating = False

    for token in tokens:
        # Strip punctuation for trigger/reset checks
        clean = re.sub(r"[^\w]", "", token).lower()

        # Fully-squashed form: collapse ALL consecutive repeats to 1
        # e.g. "hatee" → "hate",  "loovveee" → "love",  "furiousss" → "furious"
        squashed = re.sub(r"(.)\1+", r"\1", clean)

        if _SCOPE_RESET.search(token) or clean in SCOPE_RESET_WORDS:
            negating = False
            result.append(token)
        elif clean in NEGATION_TRIGGERS or squashed in NEGATION_TRIGGERS:
            negating = True
            result.append(token)
        elif negating and clean:
            result.append("NOT_" + clean)
            # Antonym injection for negated common emotion words
            base = (
                clean
                if clean in ANTONYM_MAP
                else (squashed if squashed in ANTONYM_MAP else None)
            )
            if base:
                result.extend(ANTONYM_MAP[base])
        else:
            result.append(token)
            # Determine which form hits the intensity map (exact or squashed)
            boost_key = (
                clean
                if clean in INTENSITY_BOOST_MAP
                else (squashed if squashed in INTENSITY_BOOST_MAP else None)
            )
            if boost_key:
                # If token was elongated, also inject the canonical form so TF-IDF sees it
                if squashed != clean and squashed in INTENSITY_BOOST_MAP:
                    result.append(squashed)
                result.extend(INTENSITY_BOOST_MAP[boost_key])

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
        return self  # no internal state to learn

    def transform(self, X, y=None):
        return [mark_negations(text) for text in X]
