import functools
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.data_utils import EMOTION_COLORS, EMOTION_MAP
from src.model_pipeline import predict_emotion
from src.state import app_state

router = APIRouter(tags=["prediction"])
limiter = Limiter(key_func=get_remote_address)


class PredictRequest(BaseModel):
    text: str


@functools.lru_cache(maxsize=1000)
def _cached_predict(pipeline, text: str, champion_name: str) -> dict:
    return predict_emotion(pipeline, text)


@router.post("/predict")
@limiter.limit("10/second")
def predict(request: Request, req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    pipeline = app_state.get("champion_pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not yet trained.")

    champion_name = app_state.get("champion_name", "")
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


@router.get("/classes")
def get_classes():
    return {
        "emotions": [
            {"id": k, "name": v, "color": EMOTION_COLORS.get(v, "#A9A9A9")}
            for k, v in EMOTION_MAP.items()
        ]
    }
