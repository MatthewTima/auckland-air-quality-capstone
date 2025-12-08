# Auckland Air Quality Capstone Portfolio

## 1. Project Overview

This capstone project investigates exceedances of the National Environmental Standards for Air Quality (NESAQ) in Auckland. Using long-term monitoring data from Auckland Council, we analyse when and where air pollutants exceed thresholds and build a hybrid forecasting model (Random Forest, Gradient Boosting, LSTM) deployed via a Streamlit dashboard.

The goal is to help Auckland Council move from descriptive reporting to predictive, decision-support tools for PM₂.₅ and NO₂ exceedance risks.

---

## 2. Team Members & Roles

- **Christian Tolentino** – Team Leader, Modelling & Streamlit Lead  
  Responsible for dashboard design, model integration, and overall technical direction.

- **Matthew Tima** – Co-Leader & Project Manager, Data Processing & Research Lead  
  Led data cleaning, data integrity checks, and LSTM component of the hybrid model. Supported project management and timeline tracking.

- **Juthaphat Apinanpisut** – Assistant Researcher & Documentation Lead  
  Led literature review, report writing, and supported exploratory data analysis and model interpretation.

---

## 3. Data Engineering Summary

- Merged all 1C and 2C datasets across 10 monitoring sites.
- Cleaned invalid values (negative pollutants, incorrect traffic counts).
- Standardised timestamps and parameter names across all CSVs.
- Resampled data to hourly/daily frequency as needed.
- Created engineered features:
  - Lag variables (Lag1–LagN)
  - Rolling means (e.g., 24-hour averages)
  - Weather and traffic interaction terms
- Labeled exceedances based on NESAQ thresholds.

---

## 4. Modelling Summary

We implemented:

- **Baseline seasonal models:** ARIMA/SARIMA for trend and seasonality.  
- **Hybrid ensemble model:**  
  - Random Forest  
  - Gradient Boosting  
  - LSTM

The ensemble forecast is computed as the average of model predictions (RF + GB + LSTM when available). Evaluation used MAE, RMSE, MAPE, and R² for PM₂.₅ and NO₂.

---

## 5. Streamlit Dashboard Features

The dashboard contains five main tabs:

1. **Overview** – Dataset summary, number of sites, pollutants, records, NESAQ thresholds.  
2. **Temporal Analysis** – Annual, monthly, daily exceedance plots, with breaches highlighted.  
3. **Spatial Comparison** – Compare multiple sites to identify exceedance hotspots.  
4. **Weather & Traffic Insights** – Visualise relationships between pollutants, weather, and traffic variables.  
5. **Forecasting** – Hybrid model forecasts for PM₂.₅ and NO₂, user-controlled lags and horizon, and metrics (MAE, RMSE) for transparency.

Screenshots of each tab can be added here once finalised.

---

## 6. How to Run the Project

### Requirements

- Python 3.10+
- pip
- Recommended: virtual environment
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import r2_score
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler


### Setup

```bash
git clone https://github.com/YOUR-ORG/auckland-air-quality-capstone.git
cd auckland-air-quality-capstone
pip install -r requirements.txt
