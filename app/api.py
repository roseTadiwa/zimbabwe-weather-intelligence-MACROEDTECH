from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Zimbabwe Weather Intelligence API",
    description=(
        "AI-based 1-hour-ahead temperature prediction "
        "API using an XGBoost model."
    ),
    version="1.0.0"
)


# =========================================================
# FILE PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "zimbabwe_weather_2015_2025.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "xgboost_weather_model.json"
)


# =========================================================
# LOAD WEATHER DATA
# =========================================================

weather_data = pd.read_csv(DATA_FILE)

weather_data["time"] = pd.to_datetime(
    weather_data["time"]
)


# =========================================================
# LOAD XGBOOST MODEL
# =========================================================

model = xgb.Booster()

model.load_model(
    str(MODEL_FILE)
)


# =========================================================
# REQUEST DATA MODEL
# =========================================================

class WeatherInput(BaseModel):

    city: str

    prediction_datetime: str

    temperature: float
    humidity: float
    precipitation: float
    pressure: float
    wind_speed: float
    wind_direction: float
    cloud_cover: float


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Zimbabwe Weather Intelligence API",
        "status": "running",
        "version": "1.0.0"
    }


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "XGBoost",
        "prediction": "1-hour-ahead temperature"
    }


# =========================================================
# CITY ENDPOINT
# =========================================================

@app.get("/cities")
def get_cities():

    cities = sorted(
        weather_data["city"].unique().tolist()
    )

    return {
        "cities": cities
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict_weather(weather: WeatherInput):

    # -----------------------------------------------------
    # VALIDATE CITY
    # -----------------------------------------------------

    valid_cities = weather_data[
        "city"
    ].unique().tolist()

    if weather.city not in valid_cities:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unknown city. Available cities: "
                f"{valid_cities}"
            )
        )


    # -----------------------------------------------------
    # CONVERT DATETIME
    # -----------------------------------------------------

    try:

        prediction_datetime = pd.Timestamp(
            weather.prediction_datetime
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid prediction_datetime. "
                "Use format YYYY-MM-DD HH:MM."
            )
        )


    # -----------------------------------------------------
    # CITY DATA
    # -----------------------------------------------------

    city_data = weather_data[
        weather_data["city"] == weather.city
    ].sort_values("time")


    # -----------------------------------------------------
    # TIME FEATURES
    # -----------------------------------------------------

    year = prediction_datetime.year

    month = prediction_datetime.month

    day = prediction_datetime.day

    hour = prediction_datetime.hour

    day_of_year = prediction_datetime.dayofyear


    # -----------------------------------------------------
    # CYCLICAL FEATURES
    # -----------------------------------------------------

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )


    # -----------------------------------------------------
    # PREVIOUS HOUR
    # -----------------------------------------------------

    previous_hour = (
        prediction_datetime
        - pd.Timedelta(hours=1)
    )

    previous_hour_data = city_data[
        city_data["time"] == previous_hour
    ]


    if not previous_hour_data.empty:

        historical_lag_1h = float(
            previous_hour_data.iloc[0][
                "temperature_2m"
            ]
        )

        historical_precipitation_lag = float(
            previous_hour_data.iloc[0][
                "precipitation"
            ]
        )

    else:

        historical_lag_1h = weather.temperature

        historical_precipitation_lag = (
            weather.precipitation
        )


    # -----------------------------------------------------
    # PREVIOUS 24 HOURS
    # -----------------------------------------------------

    previous_24_hour = (
        prediction_datetime
        - pd.Timedelta(hours=24)
    )

    previous_24_hour_data = city_data[
        city_data["time"] == previous_24_hour
    ]


    if not previous_24_hour_data.empty:

        historical_lag_24h = float(
            previous_24_hour_data.iloc[0][
                "temperature_2m"
            ]
        )

    else:

        historical_lag_24h = weather.temperature


    # -----------------------------------------------------
    # 24-HOUR ROLLING TEMPERATURE
    # -----------------------------------------------------

    rolling_start = (
        prediction_datetime
        - pd.Timedelta(hours=24)
    )

    rolling_data = city_data[
        (city_data["time"] >= rolling_start)
        &
        (city_data["time"] < prediction_datetime)
    ]


    if not rolling_data.empty:

        historical_rolling_24h = float(
            rolling_data[
                "temperature_2m"
            ].mean()
        )

    else:

        historical_rolling_24h = weather.temperature


    # -----------------------------------------------------
    # CREATE MODEL INPUT
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

        "temperature_2m":
            weather.temperature,

        "relative_humidity_2m":
            weather.humidity,

        "precipitation":
            weather.precipitation,

        "pressure_msl":
            weather.pressure,

        "wind_speed_10m":
            weather.wind_speed,

        "wind_direction_10m":
            weather.wind_direction,

        "cloud_cover":
            weather.cloud_cover,

        "year":
            year,

        "month":
            month,

        "day":
            day,

        "hour":
            hour,

        "day_of_year":
            day_of_year,

        "month_sin":
            month_sin,

        "month_cos":
            month_cos,

        "hour_sin":
            hour_sin,

        "hour_cos":
            hour_cos,

        "temperature_lag_1h":
            historical_lag_1h,

        "temperature_lag_24h":
            historical_lag_24h,

        "temperature_rolling_24h":
            historical_rolling_24h,

        "precipitation_lag_1h":
            historical_precipitation_lag

    }])


    # -----------------------------------------------------
    # MODEL FEATURES
    # -----------------------------------------------------

    features = [

        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "pressure_msl",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
        "year",
        "month",
        "day",
        "hour",
        "day_of_year",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "temperature_lag_1h",
        "temperature_lag_24h",
        "temperature_rolling_24h",
        "precipitation_lag_1h"

    ]


    X = input_data[features]


    # -----------------------------------------------------
    # XGBOOST PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        xgb.DMatrix(X)
    )[0]


    # -----------------------------------------------------
    # API RESPONSE
    # -----------------------------------------------------

    return {

        "city":
            weather.city,

        "input_datetime":
            prediction_datetime.strftime(
                "%Y-%m-%d %H:%M"
            ),

        "forecast_datetime":
            (
                prediction_datetime
                + pd.Timedelta(hours=1)
            ).strftime(
                "%Y-%m-%d %H:%M"
            ),

        "forecast_horizon":
            "1 hour ahead",

        "predicted_temperature_c":
            round(
                float(prediction),
                2
            ),

        "model":
            "XGBoost",

        "test_mae_c":
            0.3448,

        "test_rmse_c":
            0.5210,

        "test_r2":
            0.9882

    }