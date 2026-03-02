"""
api.py
~~~~~~
FastAPI backend for the Emolyzer React frontend.
Exposes prediction, class distribution, model metrics,
dataset samples, and CSV upload/retrain endpoints.

Start with:
    python -m uvicorn api:app --reload --port 8000

 Model persistence
 -----------------
 After the first training run the pipeline + metadata are saved to
   models/champion_pipeline.joblib
   models/model_metadata.joblib
 On every subsequent startup the cached files are loaded instead of
 retraining, so the server is ready in seconds.
 Deleting those two files forces a full retrain on next startup.
"""

import sys, os, io, threading, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Persistence Paths ────────────────────────────────────────────────────────

_BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR      = os.path.join(_BASE_DIR, "models")
_PIPELINE_PATH   = os.path.join(_MODELS_DIR, "champion_pipeline.joblib")
_METADATA_PATH   = os.path.join(_MODELS_DIR, "model_metadata.joblib")
os.makedirs(_MODELS_DIR, exist_ok=True)

from src.data_utils import (
    load_and_validate,
    class_distribution,
    EMOTION_COLORS,
    EMOTION_MAP,
)
from src.model_pipeline import (
    train_and_cross_validate,
    evaluate_model,
    predict_emotion,
)

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emolyzer API",
    description="Emotion classification API powered by multi-model TF-IDF pipelines.",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Startup: Train model ─────────────────────────────────────────────────────

_state = {}
_retrain_lock = threading.Lock()

@app.on_event("startup")
def startup_event():
    if _load_cached_model():
        print("Cached model loaded successfully — skipping training.")
    else:
        print("No cached model found — training from scratch…")
        _train_from_disk()


def _save_model():
    """Persist the current champion pipeline and its metadata to disk."""
    try:
        joblib.dump(_state["champion_pipeline"], _PIPELINE_PATH)
        joblib.dump(
            {
                "champion_name": _state["champion_name"],
                "cv_results":    _state["cv_results"],
                "eval_result":   _state["eval_result"],
                "df":            _state["df"],
                "metadata":      _state["metadata"],
            },
            _METADATA_PATH,
        )
        print(f"Model saved to {_MODELS_DIR}")
    except Exception as exc:
        print(f"Warning: could not save model to disk — {exc}")


def _load_cached_model() -> bool:
    """Try to load a previously saved model. Returns True on success."""
    if not (os.path.exists(_PIPELINE_PATH) and os.path.exists(_METADATA_PATH)):
        return False
    try:
        pipeline = joblib.load(_PIPELINE_PATH)
        meta     = joblib.load(_METADATA_PATH)
        _state["champion_pipeline"] = pipeline
        _state["champion_name"]     = meta["champion_name"]
        _state["cv_results"]        = meta["cv_results"]
        _state["eval_result"]       = meta["eval_result"]
        _state["df"]                = meta["df"]
        _state["metadata"]          = meta["metadata"]
        return True
    except Exception as exc:
        print(f"Warning: cached model could not be loaded ({exc}) — retraining…")
        return False


def _train_from_disk():
    """Load the default dataset, train models, then persist the result."""
    print("Loading dataset and training models…")
    df, metadata = load_and_validate()
    _state["df"] = df
    _state["metadata"] = metadata
    _state["champion_name"] = "not trained"

    best_pipeline, best_model_name, cv_results, X_test, y_test = train_and_cross_validate(
        df, max_features=30000, C=3.0,
    )
    _state["cv_results"] = cv_results
    _state["champion_name"] = best_model_name
    _state["champion_pipeline"] = best_pipeline
    eval_result = evaluate_model(best_pipeline, X_test, y_test)
    _state["eval_result"] = eval_result
    print(f"Ready! Champion: {best_model_name}")
    _save_model()


# ─── Request Models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str


# ─── Prediction Cache ─────────────────────────────────────────────────────────
# lru_cache cannot hold the pipeline object itself (unhashable), so we wrap
# predict_emotion in a module-level function keyed on (text, champion_name).
# When the model is retrained the champion_name changes, making old entries
# effectively dead (they will be evicted by LRU over time).

@functools.lru_cache(maxsize=1000)
def _cached_predict(pipeline, text: str, champion_name: str) -> dict:
    """Cached wrapper around predict_emotion."""
    return predict_emotion(pipeline, text)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "champion": _state.get("champion_name", "not trained"),
        "retraining": _state.get("retraining", False),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    pipeline = _state.get("champion_pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not yet trained.")
    champion_name = _state.get("champion_name", "")
    result = _cached_predict(pipeline, req.text.strip(), champion_name)
    probs_with_color = {
        emotion: {
            "probability": prob,
            "color": EMOTION_COLORS.get(emotion, "#A9A9A9"),
        }
        for emotion, prob in result["probabilities"].items()
    }
    return {
        "predicted_emotion": result["predicted_emotion"],
        "confidence": result["confidence"],
        "is_oov": result.get("is_oov", False),
        "probabilities": probs_with_color,
    }


@app.get("/classes")
def get_classes():
    return {
        "emotions": [
            {"id": k, "name": v, "color": EMOTION_COLORS.get(v, "#A9A9A9")}
            for k, v in EMOTION_MAP.items()
        ]
    }


@app.get("/distribution")
def get_distribution():
    df = _state.get("df")
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")
    dist = class_distribution(df)
    return {
        "distribution": dist.to_dict(orient="records"),
        "total": int(df.shape[0]),
        "num_classes": int(df["label"].nunique()),
    }


@app.get("/samples")
def get_samples(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100),
    emotion: str = Query(None),
    search: str = Query(None),
):
    """
    Return paginated sample rows from the loaded dataset.
    Optionally filter by emotion label or text search.
    """
    df = _state.get("df")
    if df is None:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    subset = df[["text", "emotion"]].copy()

    if emotion and emotion != "All":
        subset = subset[subset["emotion"] == emotion]

    if search and search.strip():
        term = search.strip().lower()
        subset = subset[subset["text"].str.lower().str.contains(term, na=False)]

    total = len(subset)
    start = (page - 1) * page_size
    end = start + page_size
    page_data = subset.iloc[start:end]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "samples": page_data.to_dict(orient="records"),
    }


@app.get("/metrics")
def get_metrics():
    eval_result = _state.get("eval_result")
    cv_results = _state.get("cv_results")
    champion_name = _state.get("champion_name")
    if eval_result is None:
        raise HTTPException(status_code=503, detail="Model not evaluated yet.")
    return {
        "champion_model": champion_name,
        "holdout_accuracy": round(eval_result["accuracy"], 4),
        "macro_f1": round(eval_result["macro_f1"], 4),
        "weighted_f1": round(float(eval_result.get("weighted_f1", eval_result["macro_f1"])), 4),
        "test_samples": eval_result.get("test_samples", "N/A"),
        "cross_validation": [
            {
                "model": name,
                "mean_f1": round(metrics["mean_f1"], 4),
                "std_f1": round(metrics["std_f1"], 4),
            }
            for name, metrics in cv_results.items()
        ],
    }


@app.post("/upload/analyse")
async def analyse_upload(file: UploadFile = File(...)):
    """
    Accept an uploaded CSV file, validate it, and return:
    - Class distribution of the uploaded data
    - Sample rows preview
    - Row count and column info
    Does NOT retrain the model.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df_new = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # Normalise column names
    df_new.columns = df_new.columns.str.strip().str.lower()

    # Must have text and label columns
    if "text" not in df_new.columns or "label" not in df_new.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain 'text' and 'label' columns."
        )

    df_new = df_new.dropna(subset=["text", "label"])
    df_new["text"] = df_new["text"].astype(str).str.strip()
    df_new["label"] = df_new["label"].astype(int)

    # Map labels to emotion names (use current EMOTION_MAP)
    df_new["emotion"] = df_new["label"].map(EMOTION_MAP)
    unmapped = df_new["emotion"].isna().sum()

    dist = (
        df_new["emotion"].value_counts()
        .reset_index()
        .rename(columns={"emotion": "Emotion", "count": "Count"})
    )
    dist["Color"] = dist["Emotion"].map(EMOTION_COLORS)
    dist["Percentage"] = (dist["Count"] / len(df_new) * 100).round(2)

    sample = df_new[["text", "emotion"]].sample(min(10, len(df_new)), random_state=42)

    return {
        "filename": file.filename,
        "total_rows": len(df_new),
        "num_classes": int(df_new["label"].nunique()),
        "unmapped_labels": int(unmapped),
        "distribution": dist.to_dict(orient="records"),
        "preview": sample.to_dict(orient="records"),
    }


@app.post("/upload/retrain")
async def retrain_on_upload(file: UploadFile = File(...)):
    """
    Accept a CSV, merge it with the existing dataset, and retrain the model.
    Training happens in a background thread so the API stays responsive.
    Poll /health for retraining=True to know when it's done.
    """
    if _state.get("retraining"):
        raise HTTPException(status_code=409, detail="A retrain is already in progress.")
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df_new = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    df_new.columns = df_new.columns.str.strip().str.lower()

    if "text" not in df_new.columns or "label" not in df_new.columns:
        raise HTTPException(
            status_code=400,
            detail="CSV must contain 'text' and 'label' columns."
        )

    df_new = df_new.dropna(subset=["text", "label"])
    df_new["text"] = df_new["text"].astype(str).str.strip()
    df_new["label"] = df_new["label"].astype(int)

    df_new["emotion"] = df_new["label"].map(EMOTION_MAP)
    df_new = df_new.dropna(subset=["emotion"])

    new_rows = len(df_new)

    def _retrain():
        with _retrain_lock:
            _state["retraining"] = True
            _state["champion_name"] = "not trained"
            print(f"Retraining exclusively on custom dataset with {new_rows} total rows…")
            try:
                best_pipeline, best_model_name, cv_results, X_test, y_test = train_and_cross_validate(
                    df_new, max_features=30000, C=3.0,
                )
                _state["df"] = df_new
                _state["cv_results"] = cv_results
                _state["champion_name"] = best_model_name
                _state["champion_pipeline"] = best_pipeline
                eval_result = evaluate_model(best_pipeline, X_test, y_test)
                _state["eval_result"] = eval_result
                print(f"Retrain complete! Champion: {best_model_name}")
                _save_model()
            except Exception as e:
                print(f"Retrain failed: {e}")
            finally:
                _state["retraining"] = False

    thread = threading.Thread(target=_retrain, daemon=True)
    thread.start()

    return {
        "status": "retraining_started",
        "new_rows": new_rows,
        "total_rows": new_rows,
        "message": "Retraining custom dataset has started. Poll /health for status.",
    }

# Trigger reload

# Trigger reload 2

# Trigger reload 3

# Force reload for dict color changes
