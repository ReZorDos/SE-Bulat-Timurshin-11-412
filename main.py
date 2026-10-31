
from collections.abc import Callable

from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field


app = FastAPI(
    title="Spam Detection API",
    description="API для определения спама в русскоязычных сообщениях",
    version="1.0.0"
)


class SpamRequest(BaseModel):
    text: str = Field(min_length=1)


class SpamResponse(BaseModel):
    is_spam: bool
    confidence: float


def get_predictor() -> Callable:
    from task1_text.spam_detector import detect_spam
    return detect_spam


@app.get("/")
def read_root():
    return {"message": "Spam Detection API is running"}


@app.post("/predict", response_model=SpamResponse)
def predict_spam(
    request: SpamRequest,
    predictor: Callable = Depends(get_predictor)
):
    return predictor(request.text)
