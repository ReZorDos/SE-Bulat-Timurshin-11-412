
import pytest
from fastapi.testclient import TestClient

from main import app, get_predictor


client = TestClient(app)


def fake_detect_spam(text: str) -> dict:
    return {
        "is_spam": True,
        "confidence": 0.98
    }


@pytest.fixture(autouse=True)
def override_predictor():
    app.dependency_overrides[get_predictor] = (
        lambda: fake_detect_spam
    )

    yield

    app.dependency_overrides.clear()


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Spam Detection API is running"
    }


def test_predict_spam():
    response = client.post(
        "/predict",
        json={"text": "Вы выиграли миллион рублей!"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "is_spam": True,
        "confidence": 0.98
    }


def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )

    assert response.status_code == 422


def test_missing_text():
    response = client.post(
        "/predict",
        json={"message": "Салам папалам"}
    )

    assert response.status_code == 422


def test_wrong_method():
    response = client.get("/predict")

    assert response.status_code == 405
