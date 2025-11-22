from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .model import FinScribeEngine
from .config_loader import CategoryTaxonomy

BASE_DIR = Path(__file__).resolve().parent.parent
CATEGORIES_PATH = BASE_DIR / "config" / "categories.yaml"
EXEMPLARS_PATH = BASE_DIR / "config" / "exemplars.json"

app = FastAPI(title="FinScribe API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine: FinScribeEngine | None = None
taxonomy: CategoryTaxonomy | None = None


class PredictRequest(BaseModel):
    text: str


class BatchPredictRequest(BaseModel):
    texts: List[str]


class FeedbackRequest(BaseModel):
    text: str
    category_id: str


class ReloadTaxonomyResponse(BaseModel):
    num_categories: int


class CategoriesResponse(BaseModel):
    categories: Dict[str, str]


@app.on_event("startup")
def startup_event():
    global engine, taxonomy
    taxonomy = CategoryTaxonomy(CATEGORIES_PATH)
    engine = FinScribeEngine(CATEGORIES_PATH, EXEMPLARS_PATH)
    print("FinScribe engine initialized")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/categories", response_model=CategoriesResponse)
def get_categories():
    if taxonomy is None:
        raise HTTPException(status_code=503, detail="Taxonomy not initialized")
    return CategoriesResponse(categories=taxonomy.get_all_labels())


@app.post("/predict")
def predict(req: PredictRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return engine.predict(req.text)


@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    outputs = [engine.predict(t) for t in req.texts]
    return {"results": outputs}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if engine is None or taxonomy is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    if req.category_id not in taxonomy.get_all_labels():
        raise HTTPException(status_code=400, detail="Unknown category_id")

    engine.add_feedback(req.text, req.category_id)
    return {"status": "ok", "message": "Feedback stored and index updated."}


@app.post("/reload_taxonomy", response_model=ReloadTaxonomyResponse)
def reload_taxonomy():
    global engine, taxonomy
    if engine is None or taxonomy is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    engine.reload_taxonomy()
    taxonomy.reload()
    return ReloadTaxonomyResponse(num_categories=len(taxonomy.get_all_labels()))
