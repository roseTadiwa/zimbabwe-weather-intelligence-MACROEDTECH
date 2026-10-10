# Phase 02 – Part 01: Zimbabwe Weather Intelligence

An end-to-end AI-based weather intelligence project that predicts temperature one hour ahead across four major cities in Zimbabwe: Harare, Bulawayo, Mutare, and Gweru.

## 1. Project Overview

This project develops a weather prediction pipeline using historical hourly weather data covering 1 January 2015 to 31 December 2025.

The project includes data acquisition, preprocessing, exploratory data analysis, feature engineering, machine learning, deep learning experiments, model evaluation, feature importance analysis, a prediction API, automated API testing, and an interactive web application.

**Geographical scope:** Zimbabwe. This implementation uses Zimbabwean weather data and should not be confused with a model trained on Indian Southwest Monsoon data.

## 2. Live Application

The project is deployed using Render and Streamlit Community Cloud.

* **Live Dashboard:** https://zimbabwe-weather-intelligence-macroedtech-nv4fr3n3byeb8vtgxfza.streamlit.app/
* **API Service:** https://zimbabwe-weather-intelligence-macroedtech.onrender.com/
* **API Health Check:** https://zimbabwe-weather-intelligence-macroedtech.onrender.com/health
* **Interactive API Documentation:** https://zimbabwe-weather-intelligence-macroedtech.onrender.com/docs

The Streamlit dashboard communicates with the FastAPI backend. The backend loads the trained XGBoost model and generates one-hour-ahead temperature predictions.

The application has been tested with all four supported cities: Harare, Bulawayo, Mutare, and Gweru.

**Deployment note:** The API is hosted on Render and may take some time to respond after a period of inactivity on the free tier.

## 3. Study Area and Data

The study covers four cities in Zimbabwe:

* Harare
* Bulawayo
* Mutare
* Gweru

The historical dataset contains 385,728 hourly records across the four cities, covering 2015–2025.

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

Six modelling approaches were evaluated:

1. Persistence baseline
2. Linear Regression
3. Random Forest
4. XGBoost
5. Baseline Long Short-Term Memory (LSTM)
6. Improved LSTM v2

The final deployed model is XGBoost. It was selected based on its performance on the evaluated test dataset.

A time-based evaluation strategy was used:

* **Training:** Data through 2022
* **Validation:** 2023–2024
* **Testing:** 2025

This approach separates the final test period from the earlier training and validation periods.

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

XGBoost achieved the strongest reported test performance among the evaluated models:

* **Mean Absolute Error (MAE):** 0.3448°C
* **Root Mean Squared Error (RMSE):** 0.5210°C
* **Coefficient of Determination (R²):** 0.9882

MAE measures the average absolute difference between predicted and observed temperatures. RMSE gives greater weight to larger errors. R² measures how much of the variation in the test outcomes is explained by the model.

These metrics describe performance on the evaluated 2025 test data. They do not guarantee equivalent accuracy in future periods, extreme weather conditions, or other locations.

## 6. Model Explainability

Feature importance analysis identified current temperature as the most influential XGBoost feature, followed by the 24-hour temperature lag and time-of-day features.

This suggests that recent temperature conditions and temporal patterns contribute substantially to the model's predictions.

Feature importance describes how the trained model uses its inputs. It does not establish causal relationships.

## 7. Application Architecture

The application consists of four main components.

1. **Streamlit frontend:** Provides the interface for selecting a city, prediction date and time, and weather input values.
2. **FastAPI backend:** Validates prediction requests, prepares model inputs, and exposes API endpoints.
3. **XGBoost model:** Generates one-hour-ahead temperature predictions.
4. **Historical weather dataset:** Provides the observations used during model development.

### Prediction request flow

`User → Streamlit → FastAPI → XGBoost → FastAPI → Streamlit`

The frontend and backend are deployed separately. The Streamlit application communicates with the API through the `API_BASE_URL` environment variable, configured in Streamlit Community Cloud secrets.

The deployed API provides the following endpoints:

| Endpoint   | Purpose                                |
| ---------- | -------------------------------------- |
| `/`        | Returns basic API information          |
| `/health`  | Reports API and model status           |
| `/cities`  | Provides the supported city options    |
| `/predict` | Accepts prediction requests            |
| `/docs`    | Provides interactive API documentation |

### Prediction behaviour

The application predicts temperature one hour ahead using the supplied weather input values and the trained XGBoost model.

Users can explore different input conditions through what-if analysis. The application does not independently retrieve current weather observations and should not be interpreted as an official forecast service or weather-warning system.

Prediction quality depends on the relevance and quality of the input values and on how well the evaluated historical data represents the conditions being modelled.

## 8. Running the Application Locally

### Prerequisites

* Python 3.12 for the API environment
* Python compatible with the dashboard dependencies
* Git, if cloning the repository
* The historical dataset and trained XGBoost model files

### Step 1: Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH.git
cd zimbabwe-weather-intelligence-MACROEDTECH
```

### Step 2: Set up the API environment

On Windows Command Prompt, run:

```bat
py -3.12 -m venv api_venv
api_venv\Scripts\activate
python -m pip install -r api_requirements.txt
```

### Step 3: Verify the required files

Before starting the application, confirm that the following files are present:

```text
data/processed/zimbabwe_weather_2015_2025.csv
models/xgboost_weather_model.json
```

The API and dashboard must be able to access files at the paths expected by the application.

If either required file is absent from a fresh clone, obtain it from the project's authorised data or model storage before running the application.

### Step 4: Start the FastAPI backend

From the project root, with `api_venv` activated, run:

```bash
uvicorn app.api:app --reload
```

The API will be available at:

* API root: http://127.0.0.1:8000/
* Health check: http://127.0.0.1:8000/health
* Available cities: http://127.0.0.1:8000/cities
* Interactive API documentation: http://127.0.0.1:8000/docs

Keep this terminal running.

### Step 5: Set up the Streamlit environment

Open a second terminal in the project root.

If you already have the Streamlit virtual environment, activate it:

```bat
venv\Scripts\activate
```

Install the dashboard dependencies:

```bash
python -m pip install -r requirements.txt
```

Use a Python version compatible with the packages specified in `requirements.txt`.

### Step 6: Configure the API URL

For local development, the dashboard defaults to:

```text
http://127.0.0.1:8000
```

If necessary, configure the API URL in Windows Command Prompt:

```bat
set API_BASE_URL=http://127.0.0.1:8000
```

Run this command in the same terminal from which you will start Streamlit.

### Step 7: Start Streamlit

Run:

```bash
streamlit run app/app.py
```

Open the dashboard at:

http://localhost:8501

Both the FastAPI backend and Streamlit frontend must be running for local predictions to work.

## 9. Project Structure

The following is a simplified overview of the repository:

```text
zimbabwe-weather-intelligence-MACROEDTECH/
├── app/
│   ├── api.py
│   ├── app.py
│   └── zimbabwe_flag.png
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

This is a simplified representation. Additional notebooks, scripts, reports, and supporting files may be present in the repository.

## 10. Reproducibility and Limitations

* API dependencies are recorded in `api_requirements.txt`.
* Dashboard dependencies are recorded in `requirements.txt`.
* Virtual environments and selected large data files are excluded from Git.
* A clean setup requires the historical CSV and trained model to be available at the paths expected by the application.
* The deployed prediction endpoint uses XGBoost. LSTM experiments are documented separately.
* Predictions depend on the quality and relevance of the supplied weather inputs.
* Reported test performance does not guarantee the same accuracy in future periods.
* The application does not independently retrieve current weather observations or issue official weather warnings.
* Performance during extreme weather conditions requires further evaluation.
* Free-tier hosting services may experience cold starts, temporary delays, or usage limitations.

## 11. Automated Testing

The FastAPI backend has an automated test suite in `tests/test_api.py`.

The tests cover core API behaviour, including:

* Root endpoint response
* API health check
* Supported city selection
* Successful prediction requests
* Rejection of unknown cities
* Rejection of invalid date-time values
* Rejection of requests with missing required fields

The test suite completed with seven passing tests in the development environment.

To run the tests locally, activate the API environment and execute:

```bash
python -m pytest -v
```

If pytest is not installed, install it in the API environment before running the command.

Tests should be rerun after significant changes to the API, model, or prediction logic.

## 12. Technologies Used

* **Programming language:** Python
* **Data processing:** Pandas, NumPy
* **Machine learning:** Scikit-learn, XGBoost
* **Deep learning experiments:** TensorFlow/Keras
* **API development:** FastAPI, Uvicorn
* **Interactive dashboard:** Streamlit
* **Development:** Visual Studio Code
* **Version control:** Git, GitHub
* **Deployment:** Render, Streamlit Community Cloud

## 13. Project Status and Future Work

The project has progressed from historical weather data processing and model evaluation to a deployed end-to-end application.

### Completed milestones

* Collected and preprocessed historical hourly weather data for four Zimbabwean cities covering 2015–2025.
* Performed feature engineering and evaluated baseline, regression, ensemble, and LSTM models.
* Selected XGBoost as the best-performing evaluated model.
* Developed a FastAPI prediction backend.
* Developed an interactive Streamlit dashboard.
* Deployed the API on Render and the dashboard on Streamlit Community Cloud.
* Verified the public API health endpoint and interactive API documentation.
* Tested predictions for Harare, Bulawayo, Mutare, and Gweru.
* Developed and ran an automated API test suite, with seven tests passing.
* Documented model evaluation results, system architecture, setup instructions, and limitations.

### Potential future improvements

* Expand automated testing to include frontend-backend integration and deployment checks.
* Monitor API availability, response times, and prediction failures.
* Evaluate model performance separately across cities, seasons, and extreme weather conditions.
* Introduce prediction uncertainty estimates where appropriate.
* Improve prediction visualisations and user guidance.
* Continue satellite imagery and environmental change analysis.
* Improve deployment monitoring and reproducibility documentation.

## 14. Repository

**GitHub Repository:**

https://github.com/roseTadiwa/zimbabwe-weather-intelligence-MACROEDTECH

This project demonstrates an end-to-end data science workflow, from historical data preparation and model evaluation to API development, application integration, automated testing, and cloud deployment.
