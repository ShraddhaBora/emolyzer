"""
test_model_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for src/model_pipeline.py
"""

import pytest
import numpy as np
import pandas as pd

from src.model_pipeline import (
    build_pipeline,
    train_and_cross_validate,
    evaluate_model,
    get_feature_importance,
    predict_emotion,
    EMOTION_MAP,
    EMOTION_LABELS,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_df():
    """
    Synthetic dataset covering all 6 emotion classes with enough samples
    per class so that stratified splitting (test_size=0.20) works correctly.
    Minimum needed: ceil(total * 0.20) >= num_classes (6).
    With 36 total: test ≈ 8, train ≈ 28 — well above the 6-class threshold.
    """
    rows = [
        # Sadness (0)
        ("I feel so sad and lonely today", 0),
        ("deep sadness fills my broken heart", 0),
        ("everything is dark and hopeless now", 0),
        ("tears run down my face again tonight", 0),
        ("I cannot stop crying it hurts so much", 0),
        ("the grief is overwhelming and heavy", 0),
        # Joy (1)
        ("I am extremely happy and full of joy", 1),
        ("what a wonderful and beautiful day this is", 1),
        ("so much happiness and joy in my heart", 1),
        ("laughing and smiling all day long today", 1),
        ("life is beautiful and I feel great", 1),
        ("this is the best day of my entire life", 1),
        # Love (2)
        ("I love you more than words can express", 2),
        ("you are the love and light of my life", 2),
        ("my heart is full of love and warmth", 2),
        ("being with you makes me feel complete", 2),
        ("I cherish every single moment with you", 2),
        ("you are my everything and I adore you", 2),
        # Anger (3)
        ("I am absolutely furious and cannot stop", 3),
        ("this makes me so angry and frustrated", 3),
        ("I want to scream I am so mad right now", 3),
        ("how dare they do this it is outrageous", 3),
        ("my blood is boiling I am so angry", 3),
        ("this injustice fills me with rage today", 3),
        # Fear (4)
        ("something terrible frightened me so much", 4),
        ("I am scared and afraid of what might happen", 4),
        ("this is so terrifying and makes me shake", 4),
        ("I cannot sleep because I am so anxious", 4),
        ("every shadow scares me in the dark night", 4),
        ("my heart races with dread and pure fear", 4),
        # Surprise (5)
        ("wow I never expected that to happen at all", 5),
        ("what a complete surprise that was incredible", 5),
        ("I am shocked and astonished beyond belief", 5),
        ("I cannot believe what just happened today", 5),
        ("that was the most unexpected thing ever", 5),
        ("I never saw that coming what a revelation", 5),
    ]
    text, label = zip(*rows)
    return pd.DataFrame({"text": list(text), "label": list(label)})


@pytest.fixture
def trained_artifacts(dummy_df):
    pipeline, best_model_name, cv_results, X_test, y_test = train_and_cross_validate(
        dummy_df, max_features=500, C=1.0
    )
    return pipeline, X_test, y_test


# ─── build_pipeline ───────────────────────────────────────────────────────────

class TestBuildPipeline:

    def test_pipeline_has_three_steps(self):
        p = build_pipeline()
        assert len(p.steps) == 3

    def test_step_names(self):
        p = build_pipeline()
        step_names = [name for name, _ in p.steps]
        assert "negation" in step_names
        assert "tfidf" in step_names
        assert "clf" in step_names

    def test_negation_step_is_first(self):
        p = build_pipeline()
        assert p.steps[0][0] == "negation"

    def test_custom_max_features(self):
        p = build_pipeline(max_features=100)
        assert p.named_steps["tfidf"].max_features == 100

    def test_custom_C(self):
        p = build_pipeline(C=2.5)
        assert p.named_steps["clf"].C == 2.5


# ─── train_and_cross_validate ──────────────────────────────────────────────────────────────

class TestTrainAndCrossValidate:

    def test_returns_five_items(self, dummy_df):
        result = train_and_cross_validate(dummy_df, max_features=500)
        assert len(result) == 5

    def test_pipeline_is_fitted(self, dummy_df):
        pipeline, *_ = train_and_cross_validate(dummy_df, max_features=500)
        # A fitted pipeline can transform without raising NotFittedError
        from sklearn.exceptions import NotFittedError
        try:
            pipeline.predict(["test input"])
        except NotFittedError:
            pytest.fail("Pipeline is not fitted after train_and_cross_validate()")

    def test_test_set_is_non_empty(self, dummy_df):
        _, _, _, X_test, y_test = train_and_cross_validate(dummy_df, max_features=500)
        assert len(X_test) > 0
        assert len(y_test) > 0

    def test_cv_results_is_dict(self, dummy_df):
        _, best_model_name, cv_results, _, _ = train_and_cross_validate(dummy_df, max_features=500)
        assert isinstance(cv_results, dict)
        assert isinstance(best_model_name, str)


# ─── evaluate_model ───────────────────────────────────────────────────────────

class TestEvaluateModel:

    def test_returns_expected_keys(self, trained_artifacts):
        pipeline, X_test, y_test = trained_artifacts
        result = evaluate_model(pipeline, X_test, y_test)
        expected_keys = {
            "accuracy", "macro_f1", "report_dict",
            "report_str", "confusion_matrix", "class_names",
        }
        assert expected_keys.issubset(result.keys())

    def test_accuracy_is_in_range(self, trained_artifacts):
        pipeline, X_test, y_test = trained_artifacts
        result = evaluate_model(pipeline, X_test, y_test)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_macro_f1_is_in_range(self, trained_artifacts):
        pipeline, X_test, y_test = trained_artifacts
        result = evaluate_model(pipeline, X_test, y_test)
        assert 0.0 <= result["macro_f1"] <= 1.0

    def test_confusion_matrix_shape(self, trained_artifacts):
        pipeline, X_test, y_test = trained_artifacts
        result = evaluate_model(pipeline, X_test, y_test)
        cm = result["confusion_matrix"]
        n = len(result["class_names"])
        assert cm.shape == (n, n)

    def test_class_names_are_strings(self, trained_artifacts):
        pipeline, X_test, y_test = trained_artifacts
        result = evaluate_model(pipeline, X_test, y_test)
        assert all(isinstance(c, str) for c in result["class_names"])


# ─── get_feature_importance ───────────────────────────────────────────────────

class TestGetFeatureImportance:

    def test_returns_dict(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = get_feature_importance(pipeline, top_n=5)
        assert isinstance(result, dict)

    def test_covers_all_classes(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = get_feature_importance(pipeline, top_n=5)
        # Should have an entry for each fitted class
        expected_emotions = {EMOTION_MAP[c] for c in pipeline.named_steps["clf"].classes_}
        assert set(result.keys()) == expected_emotions

    def test_top_n_features_per_class(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = get_feature_importance(pipeline, top_n=5)
        for emotion, features in result.items():
            assert len(features) == 5, f"{emotion} should have 5 features"

    def test_feature_weights_are_floats(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = get_feature_importance(pipeline, top_n=3)
        for emotion, features in result.items():
            for name, weight in features:
                assert isinstance(weight, float)


# ─── predict_emotion ──────────────────────────────────────────────────────────

class TestPredictEmotion:

    def test_returns_expected_keys(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = predict_emotion(pipeline, "I feel so happy today")
        expected = {"predicted_emotion", "confidence", "probabilities", "is_oov"}
        assert expected.issubset(result.keys())

    def test_predicted_emotion_is_known(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = predict_emotion(pipeline, "I feel so happy today")
        assert result["predicted_emotion"] in EMOTION_MAP.values() or result["predicted_emotion"] == "Unknown (Out of Vocabulary)"

    def test_confidence_in_range(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = predict_emotion(pipeline, "I feel so happy today")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = predict_emotion(pipeline, "I feel so happy today")
        if not result["is_oov"]:
            total = sum(result["probabilities"].values())
            assert abs(total - 1.0) < 1e-4

    def test_oov_detection_empty_string(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = predict_emotion(pipeline, "xyzqwerty zzzblorp foobar")
        assert result["is_oov"] is True
        assert result["predicted_emotion"] == "Unknown (Out of Vocabulary)"

    def test_whitespace_only_input(self, trained_artifacts):
        pipeline, _, _ = trained_artifacts
        result = predict_emotion(pipeline, "   ")
        assert result["is_oov"] is True
        assert result["predicted_emotion"] == "Unknown (Out of Vocabulary)"
