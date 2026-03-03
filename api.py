"""
api.py
~~~~~~
FastAPI backend for the Emolyzer React frontend.
Exposes prediction, class distribution, model metrics,
dataset samples, and CSV upload/retrain endpoints.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import nltk
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from routers import predict, dataset, models
from routers.predict import limiter
from src.state import app_state
from src.data_utils import load_and_validate
from src.model_pipeline import train_and_cross_validate, evaluate_model
from src.database import engine
from src.models_db import Base, DatasetRow

# ─── Persistence Paths
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")
_PIPELINE_PATH = os.path.join(_MODELS_DIR, "champion_pipeline.joblib")
_METADATA_PATH = os.path.join(_MODELS_DIR, "model_metadata.joblib")
os.makedirs(_MODELS_DIR, exist_ok=True)

# ─── App Setup
app = FastAPI(
    title="Emolyzer API",
    description="Emotion classification API powered by multi-model TF-IDF pipelines.",
    version="2.1.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(dataset.router)
app.include_router(models.router)

# ─── Startup Logic


def _populate_db(df):
    DatasetRow.__table__.drop(engine, checkfirst=True)
    Base.metadata.create_all(bind=engine)
    df[["text", "label", "emotion"]].to_sql(
        "dataset_rows", engine, if_exists="append", index=False
    )
    print("SQLite database populated.")


def _load_cached_model() -> bool:
    if not (os.path.exists(_PIPELINE_PATH) and os.path.exists(_METADATA_PATH)):
        return False
    try:
        pipeline = joblib.load(_PIPELINE_PATH)
        meta = joblib.load(_METADATA_PATH)

        df, dataset_meta = load_and_validate()
        app_state["df"] = df
        app_state["metadata"] = dataset_meta

        app_state["champion_pipeline"] = pipeline
        app_state["champion_name"] = meta["champion_name"]
        app_state["cv_results"] = meta["cv_results"]
        app_state["eval_result"] = meta["eval_result"]
        return True
    except Exception as exc:
        print(f"Warning: cached model could not be loaded ({exc}) — retraining…")
        return False


def _train_from_disk():
    print("Loading dataset and training models…")
    df, dataset_meta = load_and_validate()
    app_state["df"] = df
    app_state["metadata"] = dataset_meta
    app_state["champion_name"] = "not trained"

    best_pipeline, best_model_name, cv_results, X_test, y_test = (
        train_and_cross_validate(
            df,
            max_features=30000,
            C=3.0,
        )
    )
    app_state["cv_results"] = cv_results
    app_state["champion_name"] = best_model_name
    app_state["champion_pipeline"] = best_pipeline
    app_state["eval_result"] = evaluate_model(best_pipeline, X_test, y_test)
    print(f"Ready! Champion: {best_model_name}")
    models._save_model_versioned()


@app.on_event("startup")
def startup_event():
    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception as e:
        print(f"Warning: NLTK download failed ({e})")

    Base.metadata.create_all(bind=engine)

    if _load_cached_model():
        print("Cached model loaded successfully.")
    else:
        print("No cached model found — training from scratch…")
        _train_from_disk()

    _populate_db(app_state["df"])


# ─── Static Files & SPA Routing
_FRONTEND_DIST = os.path.join(_BASE_DIR, "frontend", "dist")
if os.path.exists(_FRONTEND_DIST):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")),
        name="assets",
    )

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API Route Not Found")
        index_file = os.path.join(_FRONTEND_DIST, "index.html")
        return FileResponse(index_file)
else:

    @app.get("/")
    def root_no_frontend():
        return {
            "status": "online",
            "message": "Backend is running, but frontend/dist was not found. Build the frontend first.",
        }
