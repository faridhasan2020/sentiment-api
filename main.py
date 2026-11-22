"""FastAPI app serving the Class 23 GRU sentiment model."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predictor import SentimentPredictor

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["predictor"] = SentimentPredictor()   # load the model ONCE at startup
    yield
    state.clear()


app = FastAPI(
    title="GRU Sentiment Analysis API",
    description="Predicts whether an IMDB-style movie review is POSITIVE or NEGATIVE.",
    version="1.0.0",
    lifespan=lifespan,
)


class ReviewRequest(BaseModel):
    text: str = Field(..., min_length=1, examples=["This movie was absolutely amazing!"])


class PredictionResponse(BaseModel):
    label: str
    probability: float


@app.get("/")
def root():
    return {
        "name": "GRU Sentiment Analysis API",
        "description": "GRU model trained on the IMDB 50K reviews dataset. "
                       "POST a review to /predict to get a POSITIVE/NEGATIVE label and probability.",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "predictor" in state}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    try:
        return state["predictor"].predict(request.text)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
