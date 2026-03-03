from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Request
import pandas as pd
import io
from datetime import datetime
import os
import joblib
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.data_utils import EMOTION_MAP
from src.model_pipeline import train_and_cross_validate, evaluate_model
from src.state import app_state, retrain_lock
from src.database import engine
from src.models_db import Base, DatasetRow

router = APIRouter(tags=["models"])
limiter = Limiter(key_func=get_remote_address)

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_BASE_DIR, "models")
os.makedirs(_MODELS_DIR, exist_ok=True)


def _save_model_versioned():
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pipeline_path = os.path.join(_MODELS_DIR, f"champion_{timestamp}.joblib")
        metadata_path = os.path.join(_MODELS_DIR, f"metadata_{timestamp}.joblib")

        joblib.dump(app_state["champion_pipeline"], pipeline_path)
        joblib.dump(
            {
                "champion_name": app_state["champion_name"],
                "cv_results": app_state["cv_results"],
                "eval_result": app_state["eval_result"],
                "metadata": app_state["metadata"],
            },
            metadata_path,
        )

        # update generic latest paths
        joblib.dump(
            app_state["champion_pipeline"],
            os.path.join(_MODELS_DIR, "champion_pipeline.joblib"),
        )
        joblib.dump(
            {
                "champion_name": app_state["champion_name"],
                "cv_results": app_state["cv_results"],
                "eval_result": app_state["eval_result"],
                "metadata": app_state["metadata"],
            },
            os.path.join(_MODELS_DIR, "model_metadata.joblib"),
        )

        print(f"Model saved and versioned at {timestamp}")
    except Exception as exc:
        print(f"Warning: could not save model to disk — {exc}")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "champion": app_state.get("champion_name", "not trained"),
        "retraining": app_state.get("retraining", False),
    }


@router.get("/metrics")
def get_metrics():
    eval_result = app_state.get("eval_result")
    cv_results = app_state.get("cv_results")
    champion_name = app_state.get("champion_name")
    if eval_result is None:
        raise HTTPException(status_code=503, detail="Model not evaluated yet.")
    return {
        "champion_model": champion_name,
        "holdout_accuracy": round(eval_result["accuracy"], 4),
        "macro_f1": round(eval_result["macro_f1"], 4),
        "weighted_f1": round(
            float(eval_result.get("weighted_f1", eval_result["macro_f1"])), 4
        ),
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


def _retrain_background(df_new: pd.DataFrame):
    with retrain_lock:
        if app_state["retraining"]:
            return
        app_state["retraining"] = True
        app_state["champion_name"] = "not trained"
        print(
            f"Retraining exclusively on custom dataset with {len(df_new)} total rows…"
        )
        try:
            best_pipeline, best_model_name, cv_results, X_test, y_test = (
                train_and_cross_validate(
                    df_new,
                    max_features=30000,
                    C=3.0,
                )
            )
            app_state["df"] = df_new
            app_state["cv_results"] = cv_results
            app_state["champion_name"] = best_model_name
            app_state["champion_pipeline"] = best_pipeline
            app_state["eval_result"] = evaluate_model(best_pipeline, X_test, y_test)

            DatasetRow.__table__.drop(engine, checkfirst=True)
            Base.metadata.create_all(bind=engine)
            df_new[["text", "label", "emotion"]].to_sql(
                "dataset_rows", engine, if_exists="append", index=False
            )

            print(f"Retrain complete! Champion: {best_model_name}")
            _save_model_versioned()
        except Exception as e:
            print(f"Retrain failed: {e}")
        finally:
            app_state["retraining"] = False


@router.post("/upload/retrain")
@limiter.limit("2/minute")
async def retrain_on_upload(
    request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    if app_state.get("retraining"):
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
            status_code=400, detail="CSV must contain 'text' and 'label' columns."
        )

    df_new = df_new.dropna(subset=["text", "label"])
    df_new["text"] = df_new["text"].astype(str).str.strip()
    df_new["label"] = df_new["label"].astype(int)
    df_new["emotion"] = df_new["label"].map(EMOTION_MAP)
    df_new = df_new.dropna(subset=["emotion"])
    new_rows = len(df_new)

    background_tasks.add_task(_retrain_background, df_new)

    return {
        "status": "retraining_started",
        "new_rows": new_rows,
        "total_rows": new_rows,
        "message": "Retraining custom dataset has started. Poll /health for status.",
    }
