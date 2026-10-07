import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Zimbabwe Weather Intelligence",
    page_icon="🌦️",
    layout="wide"
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

@st.cache_data
def load_weather_data():

    df = pd.read_csv(DATA_FILE)

    df["time"] = pd.to_datetime(df["time"])

    return df


# =========================================================
# LOAD XGBOOST MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = xgb.Booster()

    model.load_model(str(MODEL_FILE))

    return model


# =========================================================
# LOAD RESOURCES
# =========================================================

try:

    weather_data = load_weather_data()

    model = load_model()

except Exception as e:

    st.error(
        "The application could not load the required files."
    )

    st.exception(e)

    st.stop()


# =========================================================
# TITLE
# =========================================================

st.title("🌦️ Zimbabwe Weather Intelligence")

st.subheader(
    "1-Hour-Ahead Temperature Prediction"
)

st.write(
    "An XGBoost-based weather prediction application "
    "using hourly weather observations from four "
    "Zimbabwean cities."
)


# =========================================================
# CITY
# =========================================================

st.markdown("### 📍 Select Location")

city = st.selectbox(
    "City",
    [
        "Harare",
        "Bulawayo",
        "Mutare",
        "Gweru"
    ]
)


# =========================================================
# CITY DATA
# =========================================================

city_data = weather_data[
    weather_data["city"] == city
].sort_values("time")


min_date = city_data["time"].min().date()

max_date = city_data["time"].max().date()


# =========================================================
# DATE AND TIME
# =========================================================

st.markdown("### 🕐 Prediction Time")

date_col, time_col = st.columns(2)

with date_col:

    prediction_date = st.date_input(
        "Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

with time_col:

    prediction_time = st.time_input(
        "Time",
        value=pd.Timestamp("12:00").time()
    )


prediction_datetime = pd.Timestamp(
    f"{prediction_date} {prediction_time}"
)


# =========================================================
# FIND HISTORICAL RECORD
# =========================================================

selected_record = city_data[
    city_data["time"] == prediction_datetime
]


# =========================================================
# DEFAULT WEATHER VALUES
# =========================================================

if not selected_record.empty:

    record = selected_record.iloc[0]

    default_temperature = float(
        record["temperature_2m"]
    )

    default_humidity = float(
        record["relative_humidity_2m"]
    )

    default_precipitation = float(
        record["precipitation"]
    )

    default_pressure = float(
        record["pressure_msl"]
    )

    default_wind_speed = float(
        record["wind_speed_10m"]
    )

    default_wind_direction = float(
        record["wind_direction_10m"]
    )

    default_cloud_cover = float(
        record["cloud_cover"]
    )

else:

    default_temperature = 20.0

    default_humidity = 60.0

    default_precipitation = 0.0

    default_pressure = 1013.0

    default_wind_speed = 10.0

    default_wind_direction = 180.0

    default_cloud_cover = 50.0


# =========================================================
# WEATHER INPUTS
# =========================================================

st.markdown(
    "### 🌤️ Current Weather Conditions"
)

st.caption(
    "Values are automatically populated from the "
    "historical dataset when available. You can "
    "change them before making a prediction."
)


col1, col2 = st.columns(2)


with col1:

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-20.0,
        max_value=50.0,
        value=default_temperature,
        step=0.1
    )

    humidity = st.number_input(
        "Relative Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=default_humidity,
        step=1.0
    )

    precipitation = st.number_input(
        "Precipitation (mm)",
        min_value=0.0,
        max_value=500.0,
        value=default_precipitation,
        step=0.1
    )

    pressure = st.number_input(
        "Pressure (hPa)",
        min_value=850.0,
        max_value=1100.0,
        value=default_pressure,
        step=0.1
    )


with col2:

    wind_speed = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        max_value=150.0,
        value=default_wind_speed,
        step=0.1
    )

    wind_direction = st.number_input(
        "Wind Direction (°)",
        min_value=0.0,
        max_value=360.0,
        value=default_wind_direction,
        step=1.0
    )

    cloud_cover = st.number_input(
        "Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=default_cloud_cover,
        step=1.0
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

if st.button(
    "🔮 Predict Temperature One Hour Ahead",
    type="primary"
):

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
    # PREVIOUS HOUR DATA
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

        historical_lag_1h = temperature

        historical_precipitation_lag = precipitation


    # -----------------------------------------------------
    # PREVIOUS 24 HOURS
    # -----------------------------------------------------

    previous_24_hours = (
        prediction_datetime
        - pd.Timedelta(hours=24)
    )

    previous_24_hour_data = city_data[
        city_data["time"] == previous_24_hours
    ]


    if not previous_24_hour_data.empty:

        historical_lag_24h = float(
            previous_24_hour_data.iloc[0][
                "temperature_2m"
            ]
        )

    else:

        historical_lag_24h = temperature


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

        historical_rolling_24h = temperature


    # -----------------------------------------------------
    # MODEL INPUT
    # -----------------------------------------------------

    input_data = pd.DataFrame([{

        "temperature_2m":
            temperature,

        "relative_humidity_2m":
            humidity,

        "precipitation":
            precipitation,

        "pressure_msl":
            pressure,

        "wind_speed_10m":
            wind_speed,

        "wind_direction_10m":
            wind_direction,

        "cloud_cover":
            cloud_cover,

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
    # PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        xgb.DMatrix(X)
    )[0]


    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🌡️ Prediction Result"
    )


    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Predicted Temperature",
            f"{prediction:.2f} °C"
        )


    with result_col2:

        st.metric(
            "Forecast Horizon",
            "1 Hour Ahead"
        )


    st.success(
        f"For **{city}**, the predicted temperature "
        f"one hour after "
        f"{prediction_datetime.strftime('%Y-%m-%d %H:%M')} "
        f"is **{prediction:.2f} °C**."
    )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    with st.expander(
        "📊 View Model Information"
    ):

        st.write(
            "**Model:** XGBoost Regressor"
        )

        st.write(
            "**Prediction target:** "
            "Temperature one hour ahead"
        )

        st.write(
            "**Training period:** 2015–2022"
        )

        st.write(
            "**Validation period:** 2023–2024"
        )

        st.write(
            "**Test period:** 2025"
        )

        st.write(
            "**Test MAE:** 0.3448 °C"
        )

        st.write(
            "**Test RMSE:** 0.5210 °C"
        )

        st.write(
            "**Test R²:** 0.9882"
        )