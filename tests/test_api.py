
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Zimbabwe Weather Intelligence API",
        "status": "running",
        "version": "1.0.0"
    }

    
def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model": "XGBoost",
        "prediction": "1-hour-ahead temperature"
    }

    
def test_cities_endpoint():
    response = client.get("/cities")

    assert response.status_code == 200

    data = response.json()

    assert "cities" in data
    assert data["cities"] == [
        "Bulawayo",
        "Gweru",
        "Harare",
        "Mutare"
    ]



def test_predict_endpoint():
    payload = {
        "city": "Harare",
        "prediction_datetime": "2025-12-31 12:00",
        "temperature": 22.0,
        "humidity": 55.0,
        "precipitation": 0.0,
        "pressure": 1015.0,
        "wind_speed": 10.0,
        "wind_direction": 180.0,
        "cloud_cover": 20.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["city"] == "Harare"
    assert data["input_datetime"] == "2025-12-31 12:00"
    assert data["forecast_datetime"] == "2025-12-31 13:00"
    assert data["forecast_horizon"] == "1 hour ahead"
    assert isinstance(data["predicted_temperature_c"], (int, float))
    assert data["model"] == "XGBoost"
    assert data["test_mae_c"] == 0.3448
    assert data["test_rmse_c"] == 0.5210
    assert data["test_r2"] == 0.9882



def test_predict_rejects_unknown_city():
    payload = {
        "city": "Cape Town",
        "prediction_datetime": "2025-12-31 12:00",
        "temperature": 22.0,
        "humidity": 55.0,
        "precipitation": 0.0,
        "pressure": 1015.0,
        "wind_speed": 10.0,
        "wind_direction": 180.0,
        "cloud_cover": 20.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "Unknown city" in response.json()["detail"]


def test_predict_rejects_invalid_datetime():
    payload = {
        "city": "Harare",
        "prediction_datetime": "not-a-valid-date",
        "temperature": 22.0,
        "humidity": 55.0,
        "precipitation": 0.0,
        "pressure": 1015.0,
        "wind_speed": 10.0,
        "wind_direction": 180.0,
        "cloud_cover": 20.0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "Invalid prediction_datetime" in response.json()["detail"]


def test_predict_rejects_missing_required_field():
    payload = {
        "city": "Harare",
        "prediction_datetime": "2025-12-31 12:00",
        "temperature": 22.0,
        "humidity": 55.0,
        "precipitation": 0.0,
        "pressure": 1015.0,
        "wind_speed": 10.0,
        "wind_direction": 180.0
    }

    # cloud_cover is intentionally missing
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert any(
        error["loc"][-1] == "cloud_cover"
        for error in response.json()["detail"]
    )