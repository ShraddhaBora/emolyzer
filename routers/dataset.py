from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File
import pandas as pd
import io
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.data_utils import EMOTION_COLORS, EMOTION_MAP
from src.database import get_db
from src.models_db import DatasetRow

router = APIRouter(tags=["dataset"])


@router.get("/distribution")
def get_distribution(db: Session = Depends(get_db)):
    counts = (
        db.query(DatasetRow.emotion, func.count(DatasetRow.id).label("count"))
        .group_by(DatasetRow.emotion)
        .all()
    )
    if not counts:
        raise HTTPException(status_code=503, detail="Dataset not loaded.")

    total = sum([c.count for c in counts])
    num_classes = len(counts)

    order = list(EMOTION_MAP.values())
    order_dict = {e: i for i, e in enumerate(order)}

    res = []
    for emotion, count in counts:
        res.append(
            {
                "Emotion": emotion,
                "Count": count,
                "Percentage": round((count / total * 100), 2) if total > 0 else 0,
                "Color": EMOTION_COLORS.get(emotion, "#A9A9A9"),
                "_order": order_dict.get(emotion, 999),
            }
        )
    res.sort(key=lambda x: x["_order"])
    for r in res:
        del r["_order"]

    return {
        "distribution": res,
        "total": total,
        "num_classes": num_classes,
    }


@router.get("/samples")
def get_samples(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=5, le=100),
    emotion: str = Query(None),
    search: str = Query(None),
    db: Session = Depends(get_db),
):
    query = db.query(DatasetRow.text, DatasetRow.emotion)

    if emotion and emotion != "All":
        query = query.filter(DatasetRow.emotion == emotion)
    if search and search.strip():
        term = search.strip().lower()
        query = query.filter(func.lower(DatasetRow.text).contains(term))

    total = query.count()
    start = (page - 1) * page_size
    query = query.offset(start).limit(page_size)
    samples = [{"text": r.text, "emotion": r.emotion} for r in query.all()]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
        "samples": samples,
    }


@router.post("/upload/analyse")
async def analyse_upload(file: UploadFile = File(...)):
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
    unmapped = df_new["emotion"].isna().sum()

    dist = (
        df_new["emotion"]
        .value_counts()
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
