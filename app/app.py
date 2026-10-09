
import os
import streamlit as st
import pandas as pd
import requests
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
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "zimbabwe_weather_2015_2025.csv"
)

# Use the deployed API URL when configured.
# Otherwise, use the local API during development.
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

CITIES = ["Harare", "Bulawayo", "Mutare", "Gweru"]

# =========================================================
# LOAD HISTORICAL WEATHER DATA
# =========================================================

@st.cache_data
def load_weather_data():
    df = pd.read_csv(DATA_FILE)
    df["time"] = pd.to_datetime(df["time"])
    return df


try:
    weather_data = load_weather_data()
except Exception as e:
    st.error("The historical weather dataset could not be loaded.")
    st.exception(e)
    st.stop()

# =========================================================
# CHECK API CONNECTION
# =========================================================

@st.cache_data(ttl=10)
def check_api_health():
    response = requests.get(
        f"{API_BASE_URL}/health",
        timeout=5
    )
    response.raise_for_status()
    return response.json()


try:
    api_health = check_api_health()

    if api_health.get("status") != "healthy":
        st.warning("The prediction API is reporting an unhealthy status.")
    else:
        st.sidebar.success("Prediction API connected")

except requests.RequestException:
    st.sidebar.error("Prediction API unavailable")
    st.error(
        "Cannot connect to the FastAPI backend. "
        "Please ensure the API is running and API_BASE_URL "
        "points to the correct address."
    )
    st.stop()

# =========================================================
# TITLE
# =========================================================

st.title("🌦️ Zimbabwe Weather Intelligence")

st.subheader("1-Hour-Ahead Temperature Prediction")

st.write(
    "An XGBoost-based weather prediction application using "
    "historical hourly weather observations from four Zimbabwean cities. "
    "Streamlit provides the interface, while FastAPI serves predictions."
)

st.caption(
    "This application uses historical weather data and a prediction API. "
    "It is not a live weather forecast."
)

# =========================================================
# CITY SELECTION
# =========================================================

st.markdown("### 📍 Select Location")

city = st.selectbox("City", CITIES)

city_data = (
    weather_data[weather_data["city"] == city]
    .sort_values("time")
)

if city_data.empty:
    st.error(f"No historical weather records were found for {city}.")
    st.stop()

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
# HISTORICAL RECORD
# =========================================================

selected_record = city_data[
    city_data["time"] == prediction_datetime
]

if not selected_record.empty:
    record = selected_record.iloc[0]

    default_temperature = float(record["temperature_2m"])
    default_humidity = float(record["relative_humidity_2m"])
    default_precipitation = float(record["precipitation"])
    default_pressure = float(record["pressure_msl"])
    default_wind_speed = float(record["wind_speed_10m"])
    default_wind_direction = float(record["wind_direction_10m"])
    default_cloud_cover = float(record["cloud_cover"])

else:
    default_temperature = 20.0
    default_humidity = 60.0
    default_precipitation = 0.0
    default_pressure = 1013.0
    default_wind_speed = 10.0
    default_wind_direction = 180.0
    default_cloud_cover = 50.0

    st.info(
        "No exact historical record exists for the selected time. "
        "Default weather values will be used. Check the inputs before "
        "requesting a prediction."
    )

# =========================================================
# WEATHER INPUTS
# =========================================================

st.markdown("### 🌤️ Weather Conditions")

st.caption(
    "Values are populated from the historical dataset when available. "
    "You can edit them before requesting a prediction."
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
# PREDICTION REQUEST
# =========================================================

if st.button(
    "🔮 Predict Temperature One Hour Ahead",
    type="primary"
):
    request_payload = {
        "city": city,
        "prediction_datetime": prediction_datetime.isoformat(),
        "temperature": temperature,
        "humidity": humidity,
        "precipitation": precipitation,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "cloud_cover": cloud_cover
    }

    with st.spinner("Requesting prediction from the FastAPI backend..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=request_payload,
                timeout=30
            )

            if response.status_code == 422:
                st.error("The API rejected the input values.")
                st.json(response.json())
                st.stop()

            response.raise_for_status()
            result = response.json()

        except requests.Timeout:
            st.error("The prediction request timed out. Please try again.")
            st.stop()

        except requests.ConnectionError:
            st.error(
                "The FastAPI backend could not be reached. "
                "Check that it is running and API_BASE_URL is correct."
            )
            st.stop()

        except requests.RequestException as e:
            st.error(f"The prediction request failed: {e}")

            if getattr(e, "response", None) is not None:
                st.code(e.response.text)

            st.stop()

        except ValueError:
            st.error("The API returned a response that could not be read.")
            st.stop()

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    st.markdown("---")
    st.subheader("🌡️ Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted Temperature",
            f"{result['predicted_temperature_c']:.2f} °C"
        )

    with result_col2:
        st.metric(
            "Forecast Horizon",
            result["forecast_horizon"]
        )

    st.success(
        f"For **{result['city']}**, the predicted temperature "
        f"at **{result['forecast_datetime']}** is "
        f"**{result['predicted_temperature_c']:.2f} °C**."
    )

    st.caption(
        f"Input time: {result['input_datetime']} | "
        f"Model: {result['model']}"
    )

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    with st.expander("📊 View Model Information"):
        st.write("**Model:** XGBoost Regressor")
        st.write("**Prediction target:** Temperature one hour ahead")
        st.write("**Training period:** 2015–2022")
        st.write("**Validation period:** 2023–2024")
        st.write("**Test period:** 2025")

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Test MAE",
            f"{result['test_mae_c']:.4f} °C"
        )

        metric_col2.metric(
            "Test RMSE",
            f"{result['test_rmse_c']:.4f} °C"
        )

        metric_col3.metric(
            "Test R²",
            f"{result['test_r2']:.4f}"
        )

        st.caption(
            "These are previously measured test-set metrics, not "
            "a guarantee of accuracy for this individual prediction."
        )

    # =====================================================
    # API RESPONSE
    # =====================================================

    with st.expander("🔧 View API Response"):
        st.json(result)
