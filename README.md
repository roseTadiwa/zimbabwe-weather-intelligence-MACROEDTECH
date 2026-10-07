\# Zimbabwe Weather Intelligence



An AI-based weather prediction system for one-hour-ahead temperature forecasting across four major cities in Zimbabwe: Harare, Bulawayo, Mutare, and Gweru.



\## Project Overview



This project develops an end-to-end weather intelligence and temperature prediction system using historical weather data from 2015 to 2025.



The system covers:



\- Historical weather data acquisition

\- Data validation and preprocessing

\- Exploratory data analysis

\- Feature engineering

\- Statistical and machine learning modelling

\- Deep learning with LSTM

\- Model evaluation and comparison

\- Model explainability

\- One-hour-ahead temperature prediction

\- Interactive Streamlit application



The project compares persistence, Linear Regression, Random Forest, XGBoost, and LSTM-based approaches.



\## Study Area



The system focuses on four Zimbabwean cities:



\- Harare

\- Bulawayo

\- Mutare

\- Gweru



Hourly historical weather observations were used for the period:



\*\*1 January 2015 to 31 December 2025\*\*



\## Weather Variables



The dataset contains:



\- Temperature

\- Relative humidity

\- Precipitation

\- Mean sea-level pressure

\- Wind speed

\- Wind direction

\- Cloud cover



Additional temporal and lag-based features were engineered for model development.



\## Machine Learning Models



The project evaluated the following approaches:



1\. Persistence baseline

2\. Linear Regression

3\. Random Forest

4\. XGBoost

5\. Baseline LSTM

6\. Improved LSTM v2



The final tree-based models were trained using engineered weather and temporal features.



\## Final Model Results



| Model | Test MAE (°C) | Test RMSE (°C) | Test R² |

|---|---:|---:|---:|

| Persistence | 0.9090 | 1.2067 | 0.9367 |

| Linear Regression | 0.4745 | 0.7038 | 0.9785 |

| Random Forest | 0.3477 | 0.5349 | 0.9876 |

| XGBoost | 0.3448 | 0.5210 | 0.9882 |

| Baseline LSTM | 0.8016 | 1.0334 | 0.9536 |

| LSTM v2 | 0.7771 | 1.0002 | 0.9565 |



XGBoost achieved the strongest overall test performance, with a test MAE of \*\*0.3448°C\*\*, RMSE of \*\*0.5210°C\*\*, and R² of \*\*0.9882\*\*.



\## XGBoost Explainability



Feature importance analysis showed that the current temperature was the most influential feature for the XGBoost model, followed by the 24-hour temperature lag and time-of-day features.



Feature importance indicates the features the model relies on for prediction and should not be interpreted as causal relationships.



\## Streamlit Application



The project includes an interactive Streamlit application for one-hour-ahead temperature prediction.



The application allows the user to:



\- Select a Zimbabwean city

\- Select a historical date

\- Select a time

\- Automatically load historical weather observations

\- Modify weather inputs for what-if analysis

\- Generate a one-hour-ahead temperature prediction

\- View the model performance information



The application uses the trained XGBoost model.



\## Project Structure



```text

zimbabwe-weather-intelligence/

│

├── app/

│   └── app.py

│

├── data/

│   ├── raw/

│   └── processed/

│       └── zimbabwe\_weather\_2015\_2025.csv

│

├── models/

│   ├── lstm\_weather\_model.keras

│   ├── lstm\_weather\_model\_v2.keras

│   └── xgboost\_weather\_model.json

│

├── notebooks/

│

├── reports/

│   ├── figures/

│   └── model evaluation results

│

├── srs/

│   ├── data acquisition

│   ├── preprocessing

│   ├── feature engineering

│   ├── model development

│   ├── model evaluation

│   └── explainability scripts

│

├── .gitignore

├── requirements.txt

└── README.md

