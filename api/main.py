from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pathlib import Path
import pickle
import re
import time
from functools import lru_cache
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------------------
# App
# ----------------------------
app = FastAPI(
    title="Fake News Detection API",
    version="1.0.0",
    description="Predict whether a news text is Reliable or Unreliable using a TF-IDF + Logistic Regression model."
)

# ----------------------------
# CORS (GitHub Pages + local dev)
# Origin NEVER includes path, only scheme+domain(+port).
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ishaq019.github.io",
        "https://syedishaq.me",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],   # includes OPTIONS preflight
    allow_headers=["*"],
)

# ----------------------------
# Paths & NLTK data (Build-time + runtime fallback)
# - Build-time: bin/post_compile runs download_nltk.py
# - Runtime: if still missing, download once and continue
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

os.environ["NLTK_DATA"] = str(NLTK_DATA_DIR)
nltk.data.path.append(str(NLTK_DATA_DIR))

def ensure_stopwords():
    try:
        stopwords.words("english")
    except LookupError:
        NLTK_DATA_DIR.mkdir(parents=True, exist_ok=True)
        nltk.download("stopwords", download_dir=str(NLTK_DATA_DIR))

ensure_stopwords()

# ----------------------------
# Preprocessing (must match training)
# ----------------------------
STEMMER = PorterStemmer()

@lru_cache(maxsize=1)
def stopwords_set():
    return set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    words = text.lower().split()
    sw = stopwords_set()
    words = [STEMMER.stem(w) for w in words if w not in sw]
    return " ".join(words)

# ----------------------------
# Load model artifacts
# ----------------------------
def load_pickle(path: Path, label: str):
    if not path.exists():
        files = [p.name for p in BASE_DIR.iterdir()]
        raise RuntimeError(
            f"Missing {label}. Expected: {path}\n"
            f"Files present in {BASE_DIR}: {files}"
        )
    with open(path, "rb") as f:
        return pickle.load(f)

VECTORIZER = load_pickle(BASE_DIR / "vector.pkl", "vector.pkl")
MODEL = load_pickle(BASE_DIR / "model.pkl", "model.pkl")

LABELS = {0: "Reliable", 1: "Unreliable"}

# ----------------------------
# Schemas
# ----------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=20000)

class PredictResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    ms: int

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def root():
    return {"message": "API running. See /docs. Use POST /predict."}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/meta")
def meta():
    return {
        "model": type(MODEL).__name__,
        "vectorizer": type(VECTORIZER).__name__,
        "labels": LABELS,
        "nltk_data_dir": str(NLTK_DATA_DIR),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()

    cleaned = preprocess(req.text)
    if not cleaned.strip():
        raise HTTPException(status_code=400, detail="Text became empty after preprocessing.")

    X = VECTORIZER.transform([cleaned])
    pred = int(MODEL.predict(X)[0])

    if hasattr(MODEL, "predict_proba"):
        proba = MODEL.predict_proba(X)[0]
        conf = float(max(proba))
    else:
        conf = 0.5

    ms = int((time.perf_counter() - t0) * 1000)
    return PredictResponse(prediction=pred, label=LABELS[pred], confidence=conf, ms=ms)
