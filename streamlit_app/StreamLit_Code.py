# -------------------------------
# auckland_aq_dashboard_v27_full_upgraded_filtered.py
# -------------------------------

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

# TensorFlow LSTM included in ensemble
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# ---------------------------
# Forecasting Abbreviation Mapping
# ---------------------------
forecast_abbr = {
    'Temp': 'Temperature',
    'TEMP': 'Temperature',
    'RH': 'Relative Humidity',
    'WS': 'Wind Speed',
    'WD': 'Wind Direction'
}

traffic_abbr = {
    'TrafficV': 'Traffic Volume'
}

# ---------------------------
# STREAMLIT CONFIG
# ---------------------------
st.set_page_config(
    page_title="Auckland Air Quality Dashboard (v27 Upgraded)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌏 Auckland Council — Air Quality Dashboard (v27 Upgraded)")
st.markdown("_Pre-cleaned or raw uploads → improved forecasting → daily-aggregated pollutant-weather/traffic correlations._")

# ---------------------------
# NESAQ THRESHOLDS
# ---------------------------
nesaq_thresholds = {
    'PM2.5': {'Annual': 10, '24-hour': 25},
    'PM10': {'Annual': 20, '24-hour': 50},
    'O3': {'8-hour': 100, '1-hour': 150},
    'NO2': {'Annual': 40, '24-hour': 100, '1-hour': 200},
    'SO2': {'24-hour': 350},
    'CO': {'1-hour': 30, '8-hour': 10, '24-hour': 4}
}
nesaq_pollutants_in_data = list(nesaq_thresholds.keys())

# ---------------------------
# Helpers / Data Loading
# ---------------------------
@st.cache_data
def load_combined_dataset(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df = df.dropna(subset=['Site','Parameter','Value','Datetime']).sort_values('Datetime')
    return df

@st.cache_data
def load_raw_dataset(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df.columns = [c.strip() for c in df.columns]
    return df

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def clean_raw_dataset(df, nesaq_pollutants_in_data=None, ffill_limit=2, knn_neighbors=3):
    """
    Clean and preprocess raw air quality monitoring data.

    Parameters:
        df (pd.DataFrame): Raw dataset with columns like 'Date', 'Time', 'Datetime', 'Site', 'Parameter', 'Value'.
        nesaq_pollutants_in_data (list): List of key pollutants to check for negative values.
        ffill_limit (int): Maximum consecutive missing values to forward/backward fill.
        knn_neighbors (int): Number of neighbors for KNN imputation.

    Returns:
        pd.DataFrame: Cleaned dataset with correct types and imputed missing values.
    """

    # -------------------------------
    # 1. Datetime creation and validation
    # -------------------------------
    if 'Date' in df.columns and 'Time' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    else:
        raise ValueError("Dataset must have 'Datetime' or both 'Date' + 'Time' columns.")

    # -------------------------------
    # 2. Column existence check
    # -------------------------------
    for col in ['Site', 'Parameter', 'Value']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # -------------------------------
    # 3. Enforce correct data types
    # -------------------------------
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df['Site'] = df['Site'].astype('category')
    df['Parameter'] = df['Parameter'].astype('category')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Warn if too many values could not be converted
    if df['Value'].isna().mean() > 0.1:
        print("Warning: More than 10% of 'Value' could not be converted to numeric.")

    # -------------------------------
    # 4. Correct negative values for key pollutants
    # -------------------------------
    if nesaq_pollutants_in_data:
        mask = df['Parameter'].isin(nesaq_pollutants_in_data) & (df['Value'] < 0)
        df.loc[mask, 'Value'] = df.loc[mask, 'Value'].abs()

    # -------------------------------
    # 5. Fill missing categorical columns
    # -------------------------------
    cat_cols = ['Site', 'Parameter']
    df = df.sort_values('Datetime')
    for col in cat_cols:
        df[col] = df[col].ffill().bfill()
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col] = df[col].fillna(mode_value[0])

    # -------------------------------
    # 6. Numeric imputation
    # -------------------------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        if col == 'Value':
            df_grouped = []
            for (site, param), group in df.groupby(['Site', 'Parameter']):
                group = group.sort_values('Datetime').copy()
                group['Value'] = group['Value'].ffill(limit=ffill_limit).bfill(limit=ffill_limit)
                if group['Value'].isna().sum() > 0 and group['Value'].notna().sum() >= 5:
                    imputer = KNNImputer(n_neighbors=knn_neighbors)
                    group[['Value']] = imputer.fit_transform(group[['Value']])
                df_grouped.append(group)
            df = pd.concat(df_grouped, ignore_index=True)
        else:
            imputer = KNNImputer(n_neighbors=knn_neighbors)
            df[[col]] = imputer.fit_transform(df[[col]])

    # -------------------------------
    # 7. Final clean-up
    # -------------------------------
    df = df.dropna(subset=['Site', 'Parameter', 'Value', 'Datetime']).sort_values('Datetime')
    df.reset_index(drop=True, inplace=True)

    return df


@st.cache_data
def pivot_long_to_wide(df):
    df = df.copy()
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    wide = df.pivot_table(index='Datetime', columns='Parameter', values='Value', aggfunc='mean')
    wide = wide.sort_index()
    return wide

# ---------------------------
# Sidebar - Uploads + Controls
# ---------------------------
st.sidebar.header("Dataset Upload Options")
dataset_mode = st.sidebar.radio("Choose dataset type:", ['Pre-cleaned combined dataset', 'Raw dataset(s)'])

uploaded_file = None
uploaded_files = None
df = None

if dataset_mode == 'Pre-cleaned combined dataset':
    uploaded_file = st.file_uploader("Upload Combined Dataset (CSV/Excel)", type=["csv","xlsx"])
    if uploaded_file:
        try:
            df = load_combined_dataset(uploaded_file)
            st.success(f"✅ Combined dataset loaded: {len(df)} records, {df['Site'].nunique()} sites")
        except Exception as e:
            st.error(f"Error loading combined dataset: {e}")
else:
    uploaded_files = st.file_uploader("Upload One or More Raw Datasets (CSV/Excel)",
                                      type=["csv","xlsx"], accept_multiple_files=True)
    if uploaded_files:
        all_cleaned = []
        for f in uploaded_files:
            try:
                tmp = load_raw_dataset(f)
                tmp_clean = clean_raw_dataset(tmp)
                all_cleaned.append(tmp_clean)
                st.success(f"✅ {f.name} cleaned: {len(tmp_clean)} records, {tmp_clean['Site'].nunique()} sites")
            except Exception as e:
                st.error(f"Error processing {f.name}: {e}")
        if all_cleaned:
            df = pd.concat(all_cleaned, ignore_index=True)
            st.subheader("📊 Comparison Dataset Summary")
            st.dataframe(df.groupby(['Site','Parameter']).agg({'Value':'mean'}).reset_index())

# ---------------------------
# Global Detected Variables (Sidebar)
# ---------------------------
if df is not None:
    wide_df = pivot_long_to_wide(df)
    all_columns = wide_df.columns.tolist()

    nesaq_pollutants_in_data_detected = [c for c in all_columns if c in nesaq_pollutants_in_data]
    aux_pollutants = ['BC(370)','BC(880)','NO','NOx','AQI','TEMP']
    non_pollutants = [c for c in all_columns if c not in nesaq_pollutants_in_data + aux_pollutants]

    traffic_vars_in_data = [c for c in non_pollutants if 'traffic' in c.lower()]
    weather_vars_in_data = [c for c in non_pollutants if c not in traffic_vars_in_data]

    st.sidebar.header("Detected Variables")
    st.sidebar.write(f"✅ Key Pollutants Detected: {nesaq_pollutants_in_data_detected}")
    st.sidebar.write(f"🌦 Weather Variables Detected: {weather_vars_in_data}")
    st.sidebar.write(f"🛣 Traffic Variables Detected: {traffic_vars_in_data}")
    st.sidebar.write(f"ℹ️ Other Pollutants/Variables: {aux_pollutants}")  

    st.sidebar.markdown("---")
    st.sidebar.header("Forecasting Controls")
    do_forecast = st.sidebar.checkbox("Enable forecasting computations", value=True)
    log_adjust = st.sidebar.selectbox("Lag days (temporal feature)", list(range(1,15)), index=0)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Days)", min_value=1, max_value=7, value=3)

# ---------------------------
# Site selector
# ---------------------------
if df is not None:
    available_sites = sorted(df['Site'].unique())
    selected_site = st.sidebar.selectbox("Select Site (Global Filter)", ["All Sites"] + available_sites)
    if selected_site != "All Sites":
        df_filtered = df[df['Site'] == selected_site].copy()
    else:
        df_filtered = df.copy()

# ---------------------------
# Pivot wide for weather/traffic correlation
# ---------------------------
if df is not None:
    wide_df = pivot_long_to_wide(df_filtered)

# ---------------------------
# Dashboard Tabs
# ---------------------------
if df is not None:
    tabs = st.tabs(["Overview","Temporal","Spatial","Weather & Traffic","Forecasting","Insights & Policy"])

    # ---------------------------
    # Overview Tab
    # ---------------------------
    with tabs[0]:
        st.subheader("📄 Project Overview & NESAQ Thresholds")
        threshold_rows = []
        for pollutant, periods in nesaq_thresholds.items():
            for period, value in periods.items():
                threshold_rows.append([pollutant, period, value])
        st.dataframe(pd.DataFrame(threshold_rows, columns=['Pollutant', 'Period', 'NESAQ Threshold']))

        st.subheader("📊 Dataset Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sites", df['Site'].nunique())
        col2.metric("Parameters", df['Parameter'].nunique())
        col3.metric("Records", len(df))

    # ---------------------------
    # Temporal Tab
    # ---------------------------
    with tabs[1]:
        st.subheader("⏱ Exceedances & Compliance")
        for pollutant in nesaq_pollutants_in_data_detected:
            df_param = df_filtered[df_filtered['Parameter']==pollutant].copy()
            thresholds = nesaq_thresholds.get(pollutant, {})
            st.markdown(f"### {pollutant} Exceedances")
            for period, threshold_value in thresholds.items():
                if period.lower() == 'annual':
                    annual_avg = df_param.resample('Y', on='Datetime')['Value'].mean().reset_index()
                    # filter out years with no records
                    annual_avg = annual_avg[~annual_avg['Value'].isna()]
                    annual_avg['Exceedance'] = annual_avg['Value'] > threshold_value
                    fig = px.bar(annual_avg, x='Datetime', y='Value', color='Exceedance',
                                 color_discrete_map={True:'red', False:'green'},
                                 title=f"{pollutant} Annual Exceedances")
                    fig.add_hline(y=threshold_value, line_dash='dash', annotation_text=f"Threshold: {threshold_value}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    df_param = df_param.sort_values('Datetime').copy()
                    if '24-hour' in period.lower():
                        agg = df_param.resample('D', on='Datetime')['Value'].mean().reset_index()
                        agg.rename(columns={'Value':'Value','Datetime':'Date'}, inplace=True)
                    elif '8-hour' in period.lower():
                        df_param['8hr_mean'] = df_param['Value'].rolling(window=8, min_periods=1).mean()
                        agg = df_param.groupby(df_param['Datetime'].dt.date)['8hr_mean'].max().reset_index()
                        agg.rename(columns={'8hr_mean':'Value','Datetime':'Date'}, inplace=True)
                    else:
                        agg = df_param.groupby(df_param['Datetime'].dt.date)['Value'].max().reset_index()
                        agg.rename(columns={'Datetime':'Date'}, inplace=True)

                    # filter out dates with no recorded data
                    agg = agg[~agg['Value'].isna()]
                    agg['Exceedance'] = agg['Value'] > threshold_value
                    fig = px.bar(agg, x='Date', y='Value', color='Exceedance',
                                 color_discrete_map={True:'red', False:'green'},
                                 title=f"{pollutant} {period} Exceedances")
                    fig.add_hline(y=threshold_value, line_dash='dash', annotation_text=f"Threshold: {threshold_value}")
                    st.plotly_chart(fig, use_container_width=True)


    # ---------------------------
    # Spatial Tab
    # ---------------------------
    with tabs[2]:
        st.subheader("📍 Spatial Comparison Across Sites")
        spatial_param = st.selectbox("Select Pollutant for Spatial Comparison", nesaq_pollutants_in_data_detected)
        df_spatial = df[df['Parameter']==spatial_param]
        site_avg = df_spatial.groupby('Site')['Value'].mean().reset_index()
        fig_bar = px.bar(site_avg, x='Site', y='Value', color='Site',
                         title=f"Average {spatial_param} by Site")
        thr = nesaq_thresholds.get(spatial_param, {}).get('Annual') or nesaq_thresholds.get(spatial_param, {}).get('24-hour')
        if thr is not None:
            fig_bar.add_hline(y=thr, line_dash='dash', annotation_text=f"Threshold: {thr}")
        st.plotly_chart(fig_bar, use_container_width=True)
	
    # ---------------------------
    # Weather & Traffic Tab
    # ---------------------------
    with tabs[3]:
        st.subheader("🌦 Weather & 🛣 Traffic Insights (Daily Aggregated)")
    
        all_columns = wide_df.columns.tolist()
        pollutants_in_data = [c for c in all_columns if c in nesaq_pollutants_in_data_detected]
        non_pollutants = [c for c in all_columns if c not in nesaq_pollutants_in_data_detected + aux_pollutants]
    
        traffic_vars_in_data = [c for c in non_pollutants if 'traffic' in c.lower()]
        weather_vars_in_data = [c for c in non_pollutants if c not in traffic_vars_in_data]
    
        friendly_map = {**forecast_abbr, **traffic_abbr}
        pollutants_friendly = [friendly_map.get(p, p) for p in pollutants_in_data]
        weather_vars_friendly = [friendly_map.get(w, w) for w in weather_vars_in_data]
        traffic_vars_friendly = [friendly_map.get(t, t) for t in traffic_vars_in_data]
    
        st.write(f"✅ Key Pollutants: {pollutants_friendly}")
        st.write(f"🌦 Weather Variables: {weather_vars_friendly}")
        st.write(f"🛣 Traffic Variables: {traffic_vars_friendly}")
    
        if not pollutants_in_data:
            st.info("No key pollutants detected in dataset.")
        elif not weather_vars_in_data and not traffic_vars_in_data:
            st.info("No weather or traffic variables detected in dataset.")
        else:
            wide_daily = df_filtered.pivot_table(index='Datetime', columns='Parameter', values='Value')
            wide_daily = wide_daily.apply(pd.to_numeric, errors='coerce').resample('D').mean()
            wide_daily = wide_daily.dropna(how='all')
    
            corr_columns = weather_vars_in_data + traffic_vars_in_data
            corr_data = pd.DataFrame(index=pollutants_in_data, columns=corr_columns, dtype=float)
    
            for pollutant in pollutants_in_data:
                for var in corr_columns:
                    if pollutant in wide_daily.columns and var in wide_daily.columns:
                        merged = pd.concat([wide_daily[pollutant], wide_daily[var]], axis=1).dropna()
                        corr_val = merged.iloc[:, 0].corr(merged.iloc[:, 1]) if len(merged) > 1 else np.nan
                        corr_data.loc[pollutant, var] = corr_val
                    else:
                        corr_data.loc[pollutant, var] = np.nan
    
            corr_data = corr_data.astype(float)
    
            # ---------------------------
            # Highlight function for dataframe
            # ---------------------------
            def highlight_corr(val):
                if pd.isna(val):
                    return 'background-color: #f0f0f0;'
                if val >= 0.6:
                    return 'background-color: green; color: white; font-weight:bold;'
                if val <= -0.6:
                    return 'background-color: red; color: white; font-weight:bold;'
                return 'background-color: yellow; color: black;'
    
            st.write("Correlation matrix (daily-aggregated) between pollutants and weather/traffic variables:")
            st.dataframe(corr_data.style.applymap(highlight_corr))
    
            # ---------------------------
            # Plotly Heatmap with data points (fixed colorscale)
            # ---------------------------
            heatmap_z = corr_data.values
            annot_text = np.empty_like(heatmap_z, dtype=object)
            for i in range(heatmap_z.shape[0]):
                for j in range(heatmap_z.shape[1]):
                    val = heatmap_z[i, j]
                    annot_text[i, j] = f"{val:.2f}" if not np.isnan(val) else ""
    
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_z,
                x=corr_data.columns.tolist(),
                y=corr_data.index.tolist(),
                text=annot_text,
                texttemplate="%{text}",
                textfont={"size":12, "color":"black"},
                hovertemplate="Pollutant: %{y}<br>Variable: %{x}<br>Correlation: %{text}<extra></extra>",
                colorscale=[
                    [0.0, 'red'],      #  -1
                    [0.5, 'yellow'],   #   0
                    [1, 'green'],      #  +1
                ],
                zmin=-1, zmax=1,
                colorbar=dict(title="Correlation")
            ))
    
            fig_heatmap.update_layout(
                title="🌡️ Pollutant vs Weather & Traffic Correlation (Daily Aggregated)",
                xaxis_title="Weather & Traffic Variables",
                yaxis_title="Pollutants",
                autosize=True,
                height=520
            )
    
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
            # ---------------------------
            # Strong correlations table
            # ---------------------------
            strong_corrs = corr_data.stack(dropna=True).reset_index()
            strong_corrs.columns = ['Pollutant', 'Variable', 'Correlation']
            strong_corrs = strong_corrs[(strong_corrs['Correlation'] >= 0.6) | (strong_corrs['Correlation'] <= -0.6)]
            if not strong_corrs.empty:
                st.write("Strong correlations (>=0.6 or <=-0.6):")
                st.dataframe(strong_corrs.sort_values('Correlation', ascending=False))
            else:
                st.info("No strong correlations detected (>=0.6 or <=-0.6).")


import scipy.stats as stats
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import r2_score
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


if df is not None:
    with tabs[4]:
        st.subheader("📅 Forecasting — PM2.5 & NO₂ Adaptive Ensemble Forecast (Auckland Council Ready)")

        forecast_pollutants = ['PM2.5', 'NO2']

        if not do_forecast:
            st.info("Forecasting computations are disabled. Enable in sidebar to run models.")
        else:
            for selected_pollutant in forecast_pollutants:

                # -----------------------------
                # Filter Data
                # -----------------------------
                df_poll = df_filtered[df_filtered['Parameter'] == selected_pollutant].sort_values('Datetime').dropna(subset=['Value']).reset_index(drop=True)
                if df_poll.empty or len(df_poll) < 20:
                    st.warning(f"Not enough data to forecast {selected_pollutant}.")
                    continue

                # -----------------------------
                # Wide format + features
                # -----------------------------
                df_wide = pivot_long_to_wide(df_filtered)
                features = weather_vars_in_data + traffic_vars_in_data
                model_cols = [selected_pollutant] + features
                df_model = df_wide[model_cols].copy()
                df_model[features] = df_model[features].fillna(method='ffill').fillna(method='bfill')

                # -----------------------------
                # Lag features
                # -----------------------------
                for lag in range(1, log_adjust + 1):
                    df_model[f'{selected_pollutant}_Lag{lag}'] = df_model[selected_pollutant].shift(lag)
                df_model = df_model.dropna().reset_index(drop=True)

                # -----------------------------
                # Training-Validation Split
                # -----------------------------
                split_idx = int(0.8 * len(df_model))
                X = df_model[[f'{selected_pollutant}_Lag{i}' for i in range(1, log_adjust+1)] + features]
                y = df_model[selected_pollutant]
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                # -----------------------------
                # Random Forest (RF)
                # -----------------------------
                rf_param_grid = {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
                rf = RandomForestRegressor(random_state=42)
                rf_random = RandomizedSearchCV(rf, rf_param_grid, n_iter=5, cv=KFold(3), scoring='neg_mean_squared_error', n_jobs=-1)
                rf_random.fit(X_train, y_train)
                rf_best = rf_random.best_estimator_
               

                # -----------------------------
                # Gradient Boosting (GB)
                # -----------------------------
                gb_param_grid = {
                    'n_estimators': [150, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5],
                    'subsample': [0.7, 0.8, 1.0]
                }
                gb = GradientBoostingRegressor(random_state=42)
                gb_random = RandomizedSearchCV(gb, gb_param_grid, n_iter=5, cv=KFold(3), scoring='neg_mean_squared_error', n_jobs=-1)
                gb_random.fit(X_train, y_train)
                gb_best = gb_random.best_estimator_
                

                # -----------------------------
                # LSTM (SciKeras + Random Search + TimeSeriesSplit)
                # -----------------------------
                y_pred_lstm_full = np.full(len(df_model), np.nan)  # FULL length placeholder
                lstm_future_series = []
                sigma_lstm = 0

                if LSTM_AVAILABLE and len(y_train) > log_adjust:
                    scaler = MinMaxScaler()
                    y_scaled = scaler.fit_transform(y_train.values.reshape(-1,1))

                    X_lstm_train, y_lstm_train = [], []
                    for i in range(log_adjust, len(y_scaled)):
                        X_lstm_train.append(y_scaled[i-log_adjust:i,0])
                        y_lstm_train.append(y_scaled[i,0])
                    X_lstm_train = np.array(X_lstm_train).reshape(-1, log_adjust,1)
                    y_lstm_train = np.array(y_lstm_train)

                    def create_lstm_model(units=32, lr=0.001):
                        model = Sequential()
                        model.add(LSTM(units, activation="tanh", input_shape=(log_adjust,1)))
                        model.add(Dense(1))
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
                        return model

                    lstm_model = KerasRegressor(
                        model=create_lstm_model,
                        units=32,
                        lr=0.001,
                        epochs=12,
                        batch_size=8,
                        verbose=0
                    )

                    param_dist = {
                        "model__units": [32, 64, 128],
                        "model__lr": [0.001, 0.005, 0.01],
                        "batch_size": [8, 16],
                        "epochs": [20, 30, 40]
                    }

                    tscv = TimeSeriesSplit(n_splits=3)

                    random_search = RandomizedSearchCV(
                        estimator=lstm_model,
                        param_distributions=param_dist,
                        n_iter=5,
                        cv=tscv,
                        scoring="neg_mean_squared_error",
                        random_state=42
                    )

                    random_search.fit(X_lstm_train, y_lstm_train)
                    best_lstm = random_search.best_estimator_

                    y_pred_lstm_scaled = best_lstm.predict(X_lstm_train)
                    y_pred_lstm_train = scaler.inverse_transform(y_pred_lstm_scaled.reshape(-1,1)).flatten()

                    # fill LSTM predictions into full-length array
                    y_pred_lstm_full[:len(y_pred_lstm_train)+log_adjust] = np.concatenate([np.full(log_adjust, np.nan), y_pred_lstm_train])

                    # Multi-step future
                    last_input = y_scaled[-log_adjust:].reshape(1, log_adjust,1)
                    for _ in range(forecast_horizon):
                        pred_scaled = best_lstm.predict(last_input)
                        pred_val = scaler.inverse_transform(pred_scaled.reshape(-1,1))[0,0]
                        lstm_future_series.append(pred_val)
                        last_input = np.append(last_input[:,1:,:], pred_scaled.reshape(1,1,1), axis=1)

                    sigma_lstm = np.std(y_train[log_adjust:] - y_pred_lstm_train)

                # -----------------------------
                # Ensemble Predictions
                # -----------------------------
                y_pred_rf = rf_final.predict(X)
                y_pred_gb = gb_final.predict(X)

                # Standard deviations for CI
                sigma_rf = np.std(y_train - y_pred_rf[:len(y_train)])
                sigma_gb = np.std(y_train - y_pred_gb[:len(y_train)])

                df_model['RF_Pred'] = y_pred_rf
                df_model['GB_Pred'] = y_pred_gb
                df_model['LSTM_Pred'] = y_pred_lstm_full
                df_model['Ensemble'] = df_model[['RF_Pred','GB_Pred','LSTM_Pred']].mean(axis=1)

                # -----------------------------
                # Evaluation Metrics (per-model + ensemble)
                # -----------------------------
                # RF metrics (validation slice)
                rf_val = y_pred_rf[split_idx:]
                rf_mae = mean_absolute_error(y_val, rf_val)
                rf_rmse = mean_squared_error(y_val, rf_val, squared=False)
                rf_r2 = r2_score(y_val, rf_val)

                # GB metrics (validation slice)
                gb_val = y_pred_gb[split_idx:]
                gb_mae = mean_absolute_error(y_val, gb_val)
                gb_rmse = mean_squared_error(y_val, gb_val, squared=False)
                gb_r2 = r2_score(y_val, gb_val)

                # LSTM metrics (only where LSTM produced predictions on the validation set)
                lstm_segment = y_pred_lstm_full[split_idx:]
                lstm_mask = ~np.isnan(lstm_segment)
                if LSTM_AVAILABLE and lstm_mask.any():
                    # compare only where LSTM has predictions
                    lstm_val = lstm_segment[lstm_mask]
                    y_val_for_lstm = y_val.values[lstm_mask]
                    lstm_mae = mean_absolute_error(y_val_for_lstm, lstm_val)
                    lstm_rmse = mean_squared_error(y_val_for_lstm, lstm_val, squared=False)
                    lstm_r2 = r2_score(y_val_for_lstm, lstm_val)
                else:
                    lstm_mae = lstm_rmse = lstm_r2 = np.nan

                # Ensemble metrics (validation slice)
                ens_val = df_model['Ensemble'].values[split_idx:]
                ens_mae = mean_absolute_error(y_val, ens_val)
                ens_rmse = mean_squared_error(y_val, ens_val, squared=False)
                ens_r2 = r2_score(y_val, ens_val)

                # -----------------------------
                # Future Forecast (unchanged)
                # -----------------------------
                last_features = df_model.iloc[-log_adjust:][[f'{selected_pollutant}_Lag{j}' for j in range(1,log_adjust+1)] + features].values
                future_forecast_vals = []

                for i in range(forecast_horizon):
                    rf_pred = rf_final.predict(last_features)[0]
                    gb_pred = gb_final.predict(last_features)[0]
                    lstm_pred = lstm_future_series[i] if lstm_future_series else 0
                    ensemble_pred = (rf_pred + gb_pred + lstm_pred) / 3
                    future_forecast_vals.append(ensemble_pred)

                    # Shift lag features
                    new_lag_row = np.append(last_features[0,1:log_adjust], ensemble_pred)
                    if features:
                        new_lag_row = np.append(new_lag_row, last_features[0,log_adjust:])
                    last_features = new_lag_row.reshape(1,-1)

                sigma_ensemble = np.sqrt((sigma_rf**2 + sigma_gb**2 + sigma_lstm**2) / 3)
                ci_mult = stats.norm.ppf(0.975)
                ci_upper = [v + ci_mult * sigma_ensemble for v in future_forecast_vals]
                ci_lower = [v - ci_mult * sigma_ensemble for v in future_forecast_vals]

                future_dates = pd.date_range(df_poll['Datetime'].max() + pd.Timedelta(days=1), periods=forecast_horizon)
                threshold_24h = nesaq_thresholds[selected_pollutant].get('24-hour', np.inf)

                df_forecast = pd.DataFrame({
                    'Date': future_dates,
                    'Forecast': future_forecast_vals,
                    'CI_upper': ci_upper,
                    'CI_lower': ci_lower,
                    'Exceedance': [v > threshold_24h for v in future_forecast_vals]
                })

                # -----------------------------
                # Plot forecast (keeps forecast + CI + exceedance)
                # -----------------------------
                fig_f = go.Figure()

                fig_f.add_trace(go.Scatter(
                    x=df_forecast['Date'],
                    y=df_forecast['Forecast'],
                    mode='lines+markers+text',
                    text=[f"{v:.1f}" for v in df_forecast['Forecast']],
                    textposition="top center",
                    name='Forecast'
                ))

                fig_f.add_trace(go.Scatter(
                    x=df_forecast['Date'],
                    y=df_forecast['CI_upper'],
                    mode='lines+markers+text',
                    text=[f"{v:.1f}" for v in df_forecast['CI_upper']],
                    textposition="bottom center",
                    line=dict(dash='dash', color='blue'),
                    name='95% CI Upper'
                ))

                fig_f.add_trace(go.Scatter(
                    x=df_forecast['Date'],
                    y=df_forecast['CI_lower'],
                    mode='lines+markers+text',
                    text=[f"{v:.1f}" for v in df_forecast['CI_lower']],
                    textposition="top center",
                    line=dict(dash='dash', color='blue'),
                    name='95% CI Lower'
                ))

                if any(df_forecast['Exceedance']):
                    fig_f.add_trace(go.Scatter(
                        x=df_forecast[df_forecast['Exceedance']]['Date'],
                        y=df_forecast[df_forecast['Exceedance']]['Forecast'],
                        mode='markers+text',
                        text=[f"{v:.1f}" for v in df_forecast[df_forecast['Exceedance']]['Forecast']],
                        textposition="top center",
                        marker=dict(color='red', size=10),
                        name='Exceedance'
                    ))

                if threshold_24h:
                    fig_f.add_hline(
                        y=threshold_24h,
                        line_dash='dash',
                        line_color='red',
                        annotation_text=f"Threshold: {threshold_24h}"
                    )

                fig_f.update_layout(
                    title=f"{selected_pollutant} — {forecast_horizon}-Day Forecast (Ensemble RF+GB+LSTM)",
                    xaxis_title="Date",
                    yaxis_title=selected_pollutant
                )

                st.plotly_chart(fig_f, use_container_width=True)

                # -----------------------------
                # Display metrics table
                # -----------------------------
                # helper to format numbers or show 'N/A'
                def _fmt(x):
                    return f"{x:.2f}" if (isinstance(x, (int, float, np.floating)) and not np.isnan(x)) else "N/A"

                st.markdown(f"""
### ✅ **Model Accuracy Comparison — {selected_pollutant}**

| Model | MAE | RMSE | R² |
|------:|:----:|:----:|:---:|
| Random Forest | {_fmt(rf_mae)} | {_fmt(rf_rmse)} | {_fmt(rf_r2)} |
| Gradient Boosting | {_fmt(gb_mae)} | {_fmt(gb_rmse)} | {_fmt(gb_r2)} |
| LSTM | {_fmt(lstm_mae)} | {_fmt(lstm_rmse)} | {_fmt(lstm_r2)} |
| **Ensemble (RF + GB + LSTM)** | **{_fmt(ens_mae)}** | **{_fmt(ens_rmse)}** | **{_fmt(ens_r2)}** |
""")

