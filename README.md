# Phase 02 – Part 01: Zimbabwe Weather Intelligence

An end-to-end AI-based weather intelligence project that predicts temperature one hour ahead across four major cities in Zimbabwe: Harare, Bulawayo, Mutare, and Gweru.

## 1. Project Overview

This project develops a weather prediction pipeline using historical hourly weather data covering 1 January 2015 to 31 December 2025.

The project includes data acquisition, preprocessing, exploratory analysis, feature engineering, machine learning, deep learning experiments, model evaluation, feature importance analysis, a prediction API, and an interactive web application.

**Geographical scope:** Zimbabwe. This implementation is localized to Zimbabwean weather data and should not be confused with a model trained on Indian Southwest Monsoon data.

## 2. Live Application

The project is deployed online using Render and Streamlit Community Cloud.

* **Live Dashboard:** [Open Zimbabwe Weather Intelligence](https://zimbabwe-weather-intelligence-macroedtech.streamlit.app/)
* **API Service:** [Open the FastAPI backend](https://zimbabwe-weather-intelligence-macroedtech.onrender.com/)
* **API Health Check:** [Check API status](https://zimbabwe-weather-intelligence-macroedtech.onrender.com/health)
* **Interactive API Documentation:** [Explore the API](https://zimbabwe-weather-intelligence-macroedtech.onrender.com/docs)

The dashboard communicates with the FastAPI backend, which loads the trained XGBoost model and generates one-hour-ahead temperature predictions.

The application has been tested with all four supported cities: Harare, Bulawayo, Mutare, and Gweru.

**Deployment note:** The API is hosted on Render and may take some time to respond after a period of inactivity on the free tier.

## 3. Study Area and Data

The study covers four cities:

* Harare
* Bulawayo
* Mutare
* Gweru

The historical dataset contains hourly observations covering 2015–2025, with 385,728 records across the four cities.

Weather variables include:

* Air temperature
* Relative humidity
* Precipitation
* Mean sea-level pressure
* Wind speed
* Wind direction
* Cloud cover

Additional temporal, lag-based, and rolling features were engineered to support model development.

The dataset was checked for missing values and duplicate records during preprocessing.

## 4. Machine Learning Models

The following approaches were evaluated:

1. Persistence baseline
2. Linear Regression
3. Random Forest
4. XGBoost
5. Baseline Long Short-Term Memory (LSTM)
6. Improved LSTM v2

The final XGBoost model was trained using historical weather observations and engineered features.

The time-based evaluation strategy used data through 2022 for training, 2023–2024 for validation, and 2025 for testing.

## 5. Model Evaluation

The following results were obtained on the 2025 test dataset.

| Model             | Test MAE (°C) | Test RMSE (°C) | Test R² |
| ----------------- | ------------: | -------------: | ------: |
| Persistence       |        0.9090 |         1.2067 |  0.9367 |
| Linear Regression |        0.4745 |         0.7038 |  0.9785 |
| Random Forest     |        0.3477 |         0.5349 |  0.9876 |
| XGBoost           |        0.3448 |         0.5210 |  0.9882 |
| Baseline LSTM     |        0.8016 |         1.0334 |  0.9536 |
| Improved LSTM v2  |        0.7771 |         1.0002 |  0.9565 |

XGBoost achieved the strongest reported test performance, with:

* **Mean Absolute Error (MAE):** 0.3448°C
* **Root Mean Squared Error (RMSE):** 0.5210°C
* **Coefficient of Determination (R²):** 0.9882

MAE measures the average absolute difference between predictions and observed temperatures. RMSE gives greater weight to larger prediction errors, while R² describes how much of the variation in the test outcomes is explained by the model.

These results describe performance on the evaluated 2025 test data. They do not guarantee equivalent accuracy for future forecasts, extreme weather conditions, or other locations.

## 6. Model Explainability

Feature importance analysis identified current temperature as the most influential XGBoost feature, followed by the 24-hour temperature lag and time-of-day features.

This indicates that recent temperature conditions and temporal patterns contribute substantially to the model's predictions.

Feature importance describes how the trained model uses its inputs; it does not establish causal relationships.

## 7. Application Architecture

The application consists of four main components.

1. **Streamlit frontend:** Provides the user interface for selecting a city, date, time, and weather inputs.
2. **FastAPI backend:** Validates prediction requests, prepares model inputs, and exposes API endpoints.
3. **XGBoost model:** Generates one-hour-ahead temperature predictions.
4. **Historical weather dataset:** Provides the observations used for model development and historical-data-based inputs.

### Prediction request flow

`User → Streamlit → FastAPI → XGBoost → FastAPI → Streamlit`

The frontend and backend are deployed separately. The Streamlit application communicates with the API through the `API_BASE_URL` environment variable, which is configured in Streamlit Community Cloud secrets.

The deployed API provides the following endpoints:

| Endpoint   | Purpose                                |
| ---------- | -------------------------------------- |
| `/`        | Returns basic API information          |
| `/health`  | Reports the API and model status       |
| `/cities`  | Provides the supported city options    |
| `/predict` | Accepts prediction requests            |
| `/docs`    | Provides interactive API documentation |

The application supports historical-data-based predictions and what-if analysis. It is not a live weather service and does not automatically obtain current weather observations.

## 8. Running the Application Locally

### Prerequisites

* Python 3.12 for the API environment
* Python installed for the Streamlit environment
* Git, if cloning the repository
* The historical dataset and trained XGBoost model

### Step 1: Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH.git
cd zimbabwe-weather-intelligence-MACROEDTECH
```

### Step 2: Set up the API environment

On Windows Command Prompt:

```cmd
py -3.12 -m venv api_venv
api_venv\Scripts\activate
python -m pip install -r api_requirements.txt
```

### Step 3: Start the FastAPI backend

From the project root, with `api_venv` activated:

```cmd
uvicorn app.api:app --reload
```

The API will be available at:

* API root: http://127.0.0.1:8000/
* Health check: http://127.0.0.1:8000/health
* Available cities: http://127.0.0.1:8000/cities
* Interactive API documentation: http://127.0.0.1:8000/docs

Keep this terminal running.

### Step 4: Set up the Streamlit environment

Open a second terminal in the project root.

Activate the existing Streamlit virtual environment if available:

```cmd
venv\Scripts\activate
```

Install the Streamlit application's direct dependencies:

```cmd
python -m pip install -r requirements.txt
```

### Step 5: Configure the API URL

For local development, the dashboard defaults to:

```text
http://127.0.0.1:8000
```

If needed, set the environment variable in Windows Command Prompt:

```cmd
set API_BASE_URL=http://127.0.0.1:8000
```

Set this in the same terminal from which you will start Streamlit.

### Step 6: Start Streamlit

```cmd
streamlit run app/app.py
```

Open http://localhost:8501 in your browser.

Both the FastAPI backend and Streamlit frontend must be running for predictions to work locally.

## 9. Project Structure

The following is a simplified overview of the repository:

```text
zimbabwe-weather-intelligence-MACROEDTECH/
├── app/
│   ├── api.py
│   └── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── zimbabwe_weather_2015_2025.csv
│   └── satellite/
├── models/
│   └── xgboost_weather_model.json
├── notebooks/
├── reports/
│   └── satellite/
├── srs/
├── tests/
│   └── test_api.py
├── .gitignore
├── api_requirements.txt
├── requirements.txt
└── README.md
```

This is a simplified representation. Additional notebooks, scripts, data files, and reports may be present in the repository.

## 10. Reproducibility and Limitations

* The API dependencies are recorded in `api_requirements.txt`.
* The Streamlit application's dependencies are recorded in `requirements.txt`.
* Virtual environments and selected large data files are excluded from Git.
* A clean setup requires the historical CSV and trained model files to be available at the paths expected by the application.
* The deployed prediction endpoint uses XGBoost. LSTM experiments are documented separately.
* Prediction quality depends on the relevance and quality of the supplied inputs and the historical data used to train the model.
* The model's reported test performance does not guarantee the same accuracy in future periods.
* The application does not independently retrieve current observations or provide official weather warnings.
* Free-tier hosting services may experience cold starts, temporary delays, or usage limitations.

## 11. Automated Testing

The FastAPI backend has an automated test suite in `tests/test_api.py`.

The tests cover core API behaviour, including health checks, city selection, and prediction requests.

The test suite previously completed with seven passing tests in the development environment. Tests should be rerun after significant changes to the API or model integration.

To run the tests locally, activate the API environment and execute:

```cmd
python -m pytest -v
```

If pytest is not installed in the active environment, install it before running the command.

## 12. Technologies Used

* **Programming language:** Python
* **Data processing:** Pandas, NumPy
* **Machine learning:** Scikit-learn, XGBoost
* **Deep learning experiments:** TensorFlow/Keras
* **API development:** FastAPI, Uvicorn
* **Interactive dashboard:** Streamlit
* **Development and version control:** VS Code, Git, GitHub
* **Deployment:** Render, Streamlit Community Cloud

## 13. Project Status and Future Work

The project has progressed from model development to a deployed end-to-end application.

### Completed milestones

* Historical hourly weather data collection and preprocessing for four Zimbabwean cities, covering 2015–2025.
* Feature engineering and evaluation of baseline machine learning, ensemble learning, and LSTM approaches.
* Selection of XGBoost as the best-performing evaluated model.
* Development of a FastAPI prediction backend and an interactive Streamlit dashboard.
* Deployment of the API on Render and the dashboard on Streamlit Community Cloud.
* Verification of the API health endpoint.
* Successful prediction tests for Harare, Bulawayo, Mutare, and Gweru.
* Automated API testing during development.

### Potential future improvements

* More comprehensive automated and integration testing.
* Monitoring API availability and prediction failures.
* Additional model validation across cities, seasons, and extreme weather conditions.
* Improved prediction visualizations and uncertainty estimates.
* Further analysis of satellite imagery and environmental change.
* Expanded documentation and deployment monitoring.

## 14. Repository

**GitHub:** [roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH](https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH)

This project demonstrates an end-to-end data science workflow, from historical data preparation and model evaluation to API development, web application integration, testing, and cloud deployment.
