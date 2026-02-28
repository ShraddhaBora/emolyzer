"""
test_preprocessing.py
~~~~~~~~~~~~~~~~~~~~~
Unit tests for src/preprocessing.py
"""

import pytest
from src.preprocessing import (
    expand_contractions,
    mark_negations,
    NegationPreprocessor,
    CONTRACTION_MAP,
    NEGATION_TRIGGERS,
)


# ─── expand_contractions ──────────────────────────────────────────────────────

class TestExpandContractions:

    def test_dont_expanded(self):
        assert expand_contractions("I don't like this") == "i do not like this"

    def test_cant_expanded(self):
        assert expand_contractions("I can't believe it") == "i cannot believe it"

    def test_wont_expanded(self):
        assert expand_contractions("She won't come") == "she will not come"

    def test_isnt_expanded(self):
        assert expand_contractions("This isn't right") == "this is not right"

    def test_hasnt_expanded(self):
        assert expand_contractions("He hasn't done it") == "he has not done it"

    def test_im_expanded(self):
        assert expand_contractions("I'm happy") == "i am happy"

    def test_youre_expanded(self):
        assert expand_contractions("You're amazing") == "you are amazing"

    def test_lowercase_applied(self):
        result = expand_contractions("I Don't Like This")
        assert result == result.lower()

    def test_no_contraction_unchanged(self):
        result = expand_contractions("I love you")
        assert result == "i love you"

    def test_multiple_contractions(self):
        result = expand_contractions("I don't like you and I can't forgive you")
        assert "do not" in result
        assert "cannot" in result
        assert "don't" not in result
        assert "can't" not in result

    def test_all_entries_in_map_are_lowercase(self):
        for contraction in CONTRACTION_MAP:
            assert contraction == contraction.lower(), (
                f"Contraction key '{contraction}' should be lowercase"
            )


# ─── mark_negations ───────────────────────────────────────────────────────────

class TestMarkNegations:

    def test_dont_like_marked(self):
        result = mark_negations("I don't like you")
        assert "NOT_like" in result
        assert "NOT_you" in result

    def test_not_happy_marked(self):
        result = mark_negations("I am not happy")
        assert "NOT_happy" in result

    def test_never_trust_marked(self):
        result = mark_negations("I never trust anyone")
        assert "NOT_trust" in result
        assert "NOT_anyone" in result

    def test_punctuation_resets_scope(self):
        # After the period, negation should not apply to "love"
        result = mark_negations("I don't like you. I love this.")
        assert "NOT_like" in result
        assert "NOT_love" not in result

    def test_positive_statement_unchanged(self):
        result = mark_negations("I am so happy today!")
        assert "NOT_" not in result

    def test_can_not_love_marked(self):
        result = mark_negations("I can't love this")
        assert "NOT_love" in result

    def test_no_good_marked(self):
        result = mark_negations("This is no good")
        assert "NOT_good" in result

    def test_output_is_string(self):
        assert isinstance(mark_negations("hello world"), str)

    def test_empty_string(self):
        result = mark_negations("")
        assert result == ""

    def test_only_negation_word(self):
        result = mark_negations("not")
        # "not" itself should not be prefixed, just kept
        assert result.strip() == "not"

    def test_multiple_sentences(self):
        result = mark_negations("I don't want this. I am happy.")
        # "want" should be prefixed, "happy" should not be
        assert "NOT_want" in result
        assert "NOT_happy" not in result


# ─── NegationPreprocessor (sklearn transformer) ───────────────────────────────

class TestNegationPreprocessor:

    @pytest.fixture
    def preprocessor(self):
        return NegationPreprocessor()

    def test_fit_returns_self(self, preprocessor):
        result = preprocessor.fit(["hello", "world"])
        assert result is preprocessor

    def test_transform_returns_list(self, preprocessor):
        result = preprocessor.transform(["I don't like you"])
        assert isinstance(result, list)

    def test_transform_applies_negation(self, preprocessor):
        result = preprocessor.transform(["I don't like you"])
        assert any("NOT_like" in s for s in result)

    def test_transform_multiple_documents(self, preprocessor):
        docs = [
            "I don't like this at all",
            "I am so happy today",
            "She can't stop crying",
        ]
        result = preprocessor.transform(docs)
        assert len(result) == 3
        assert "NOT_like" in result[0]
        assert "NOT_" not in result[1]   # positive sentence untouched
        assert "NOT_stop" in result[2]

    def test_fit_transform_equivalent(self, preprocessor):
        docs = ["I don't like you", "I love this"]
        fitted = preprocessor.fit(docs)
        transformed = fitted.transform(docs)
        assert len(transformed) == 2

    def test_get_params(self, preprocessor):
        # BaseEstimator provides get_params()
        params = preprocessor.get_params()
        assert isinstance(params, dict)

    def test_set_params(self, preprocessor):
        # Should not raise (stateless, no params to set but contract is honoured)
        preprocessor.set_params()
