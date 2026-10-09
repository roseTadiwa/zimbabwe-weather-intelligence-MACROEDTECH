# Phase 02 – Part 01: Zimbabwe Weather Intelligence

An end-to-end AI-based weather intelligence project that predicts temperature one hour ahead across four major cities in Zimbabwe: Harare, Bulawayo, Mutare, and Gweru.

## 1. Project Overview

This project develops a weather prediction pipeline using historical hourly weather data covering 1 January 2015 to 31 December 2025.

The project includes data acquisition, preprocessing, exploratory analysis, feature engineering, machine learning, deep learning experiments, model evaluation, explainability, an API, and an interactive web application.

**Geographical scope:** Zimbabwe. This implementation is localized to Zimbabwean weather data and should not be confused with a model trained on Indian Southwest Monsoon data.

## 2. Study Area and Data

The study covers four cities:

* Harare
* Bulawayo
* Mutare
* Gweru

The historical dataset contains hourly observations for:

* Air temperature
* Relative humidity
* Precipitation
* Mean sea-level pressure
* Wind speed
* Wind direction
* Cloud cover

Additional temporal, lag-based, and rolling features were engineered for model development.

## 3. Machine Learning Models

The following approaches were evaluated:

1. Persistence baseline
2. Linear Regression
3. Random Forest
4. XGBoost
5. Baseline LSTM
6. Improved LSTM v2

The final XGBoost model was trained using historical weather and engineered features. The time-based evaluation used data through 2022 for training, 2023–2024 for validation, and 2025 for testing.

## 4. Model Evaluation

| Model             | Test MAE (°C) | Test RMSE (°C) | Test R² |
| ----------------- | ------------: | -------------: | ------: |
| Persistence       |        0.9090 |         1.2067 |  0.9367 |
| Linear Regression |        0.4745 |         0.7038 |  0.9785 |
| Random Forest     |        0.3477 |         0.5349 |  0.9876 |
| XGBoost           |        0.3448 |         0.5210 |  0.9882 |
| Baseline LSTM     |        0.8016 |         1.0334 |  0.9536 |
| Improved LSTM v2  |        0.7771 |         1.0002 |  0.9565 |

XGBoost achieved the strongest reported test performance, with an MAE of **0.3448°C**, RMSE of **0.5210°C**, and R² of **0.9882**.

These metrics describe performance on the evaluated 2025 test data and do not guarantee equivalent accuracy for future forecasts or other locations.

## 5. Model Explainability

Feature importance analysis identified current temperature as the most influential XGBoost feature, followed by the 24-hour temperature lag and time-of-day features.

Feature importance describes how the trained model uses its inputs. It does not establish causal relationships.

## 6. Application Architecture

The project uses separate API and web application components:

1. **Streamlit frontend:** provides the interface for selecting a city, date, time, and weather inputs.
2. **FastAPI backend:** validates prediction requests, prepares model inputs, and exposes prediction endpoints.
3. **XGBoost model:** generates the one-hour-ahead temperature prediction.
4. **Historical dataset:** provides weather observations used by the application.

The request flow is:

`User → Streamlit → FastAPI → XGBoost → FastAPI → Streamlit`

The application supports historical-data-based predictions and what-if analysis. It is not a live weather service and does not automatically obtain current weather observations.

## 7. Running the Application Locally

### Prerequisites

* Python 3.12 for the API environment
* Python installed for the Streamlit environment
* Git, if cloning the repository
* The project dataset and trained XGBoost model

### Step 1: Clone the repository

```bash
git clone https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH.git
cd zimbabwe-weather-intelligence
```

If the repository folder has a different name after cloning, enter the folder that Git created.

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

Open a second terminal in the project root. Activate the project's existing Streamlit environment, if available:

```cmd
venv\Scripts\activate
```

Install the Streamlit application's dependencies:

```cmd
python -m pip install streamlit pandas requests
```

### Step 5: Start Streamlit

```cmd
streamlit run app/app.py
```

Open http://localhost:8501 in your browser.

Both the FastAPI backend and Streamlit frontend must be running for predictions to work.

## 8. Project Structure

```text
zimbabwe-weather-intelligence/
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
├── .gitignore
├── api_requirements.txt
├── requirements.txt
└── README.md
```

This is a simplified overview; additional notebooks, scripts, data files, and reports may be present in the working repository.

## 9. Reproducibility and Limitations

* The API dependencies are recorded in `api_requirements.txt`.
* The Streamlit application's direct dependencies are installed separately.
* Virtual environments and selected large data files are excluded from Git.
* A clean setup requires the historical CSV and trained model files to be available at the paths expected by the application.
* LSTM experiments are documented separately; the deployed prediction endpoint uses XGBoost.
* Predictions depend on the quality and relevance of the input observations and the model's historical training data.

## 10. Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn
* XGBoost
* TensorFlow/Keras for LSTM experiments
* FastAPI and Uvicorn
* Streamlit
* Git and GitHub

## 11. Project Status

The XGBoost model has been evaluated, and the Streamlit frontend has been successfully integrated with the FastAPI prediction backend in the local development environment.

Further work may include deployment, automated testing, monitoring, and additional validation.
