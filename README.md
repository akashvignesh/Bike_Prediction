# 🚴 Bike Ride Demand Prediction

An end-to-end machine learning system that forecasts 6-hourly bike ride demand for NYC Citi Bike stations, comparing 2024 historical actuals against 2025 model predictions via an interactive Streamlit dashboard.

---

## 📌 Project Overview

This project builds a production-grade ML pipeline that:

1. **Ingests** raw Citi Bike trip data from the public S3 bucket.
2. **Engineers features** — aggregates rides into 6-hour bins, extracts time-based features, and generates 112 lag features (28 days × 4 bins/day) for the top 3 busiest stations.
3. **Stores features** in [Hopsworks](https://www.hopsworks.ai/) Feature Store for reproducibility and versioning.
4. **Trains** a LightGBM regression model evaluated with time-series cross-validation.
5. **Generates** rolling 2025 predictions using the trained model and stores them back in Hopsworks.
6. **Visualises** 2024 actual vs 2025 predicted demand in an interactive Streamlit dashboard.
7. **Automates** the feature and inference pipelines on a schedule using GitHub Actions.

---

## 🏗️ Architecture

```
 Citi Bike S3 (raw CSVs)
          │
          ▼
 ┌─────────────────────┐
 │  Feature Pipeline   │  (GitHub Actions — hourly cron)
 │  feature_engineering│
 │  .py                │
 └────────┬────────────┘
          │  Writes features
          ▼
 ┌─────────────────────┐
 │   Hopsworks         │
 │   Feature Store     │◄──── Model Registry
 └────────┬────────────┘
          │  Reads features + model
          ▼
 ┌─────────────────────┐
 │  Inference Pipeline │  (GitHub Actions — triggered after feature pipeline)
 │  interface.py       │
 └────────┬────────────┘
          │  Writes predictions
          ▼
 ┌─────────────────────┐
 │   Hopsworks         │
 │   Predictions FG    │
 └────────┬────────────┘
          │
          ▼
 ┌─────────────────────┐
 │  Streamlit Dashboard│  (front-end/app3.py)
 │  2024 Actual vs     │
 │  2025 Predicted     │
 └─────────────────────┘
```

---

## 🗂️ Repository Structure

```
Bike_Prediction/
│
├── Back-end/                        # Model development notebooks
│   ├── lightgbmModel.ipynb          # LightGBM model training
│   ├── lightgbmTuned.ipynb          # Hyperparameter tuning
│   ├── lightgbmModelwithFI.ipynb    # Feature importance analysis
│   ├── data_collect.ipynb           # Raw data collection
│   ├── pipeline_utils.py            # Shared utilities
│   └── *.joblib                     # Serialised model artefacts
│
├── front-end/                       # Streamlit dashboard
│   ├── app.py                       # Basic dashboard
│   ├── app2.py                      # Enhanced dashboard
│   └── app3.py                      # Production dashboard (tabbed UI)
│
├── workflows/                       # Automated pipeline scripts
│   ├── feature_engineering.py       # Data fetch, transform & upload to Hopsworks
│   ├── interface.py                 # Rolling inference & prediction storage
│   └── utils.py                     # Shared helpers
│
├── models/                          # Saved model files
│   └── lightgbm_bikeride_model.joblib
│
├── .github/workflows/
│   ├── feature_pipeline.yaml        # Hourly feature pipeline (cron)
│   └── inference_pipeline.yaml      # Inference pipeline (post-feature trigger)
│
├── *.parquet                        # Cached data snapshots
├── requirements.txt                 # Python dependencies
└── runtime.txt                      # Python version pin (3.11)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **ML Model** | [LightGBM](https://lightgbm.readthedocs.io/) — gradient-boosted trees for regression |
| **Feature Store & Model Registry** | [Hopsworks](https://www.hopsworks.ai/) |
| **Dashboard / UI** | [Streamlit](https://streamlit.io/) |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Visualisation** | [Altair](https://altair-viz.github.io/) |
| **Model Serialisation** | [Joblib](https://joblib.readthedocs.io/) |
| **Data Format** | Apache Parquet (via [PyArrow](https://arrow.apache.org/docs/python/)) |
| **ML Evaluation** | [Scikit-learn](https://scikit-learn.org/) — TimeSeriesSplit, RMSE, MAE, R² |
| **CI/CD Automation** | [GitHub Actions](https://docs.github.com/en/actions) |
| **Environment Config** | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## ⚙️ ML Pipeline Details

### Feature Engineering

- Downloads monthly Citi Bike trip CSVs from the public AWS S3 bucket.
- Identifies the **top 3 busiest stations** by ride count.
- Aggregates trip records into **6-hour time bins**.
- Creates **112 lag features** (28 days of history × 4 bins per day).
- Extracts time-based features: `hour`, `day_of_week`, `month`, `is_weekend`.
- Uploads the transformed feature group to Hopsworks.

### Model Training

- Algorithm: **LightGBM Regressor**
- Validation: **TimeSeriesSplit** (5 folds) to prevent data leakage.
- Metrics: RMSE, MAE, R², MAPE.
- Best model is registered in the Hopsworks Model Registry.

### Inference (Rolling Prediction)

- Loads the latest model from the Hopsworks Model Registry.
- Generates rolling 6-hour predictions across the full 2025 calendar year.
- Appends each prediction back to history so it can be used as lag input for the next step.
- Stores predictions in a dedicated Hopsworks feature group.

---

## 🖥️ Streamlit Dashboard

The production dashboard (`front-end/app3.py`) provides:

- **Tab 1 — 2024 Actual Demand**: Interactive time-series chart of real ride counts per station.
- **Tab 2 — 2025 Predicted Demand**: Interactive forecast chart using model output.
- **Location filter**: Sidebar dropdown to select individual stations.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- A [Hopsworks](https://www.hopsworks.ai/) account with a project and API key.

### Installation

```bash
git clone https://github.com/akashvignesh/Bike_Prediction.git
cd Bike_Prediction
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
HOPSWORKS_API_KEY=<your_hopsworks_api_key>
HOPSWORKS_PROJECT_NAME=<your_project_name>
FEATURE_GROUP_NAME=time_series_six_hourly_feature_group_bike
FEATURE_GROUP_VERSION=1
FEATURE_VIEW_NAME=time_series_six_hourly_feature_view_bike
FEATURE_VIEW_VERSION=1
```

### Run the Dashboard

```bash
streamlit run front-end/app3.py
```

### Run Pipelines Manually

```bash
# Feature pipeline
python -m workflows.feature_engineering

# Inference pipeline
python -m workflows.interface
```

---

## 🔄 Automated Pipelines (GitHub Actions)

| Workflow | Trigger | Script |
|---|---|---|
| `bike_rides_hourly_features_pipeline` | Every hour (cron) | `workflows/feature_engineering.py` |
| `Bike_rides_hourly_inference_pipeline` | After feature pipeline completes | `workflows/interface.py` |

Set `HOPSWORKS_API_KEY` as a GitHub Actions secret in your repository settings.

---

## 📊 Data Source

Trip data is sourced from the [NYC Citi Bike public dataset](https://citibikenyc.com/system-data), hosted on AWS S3:

```
https://s3.amazonaws.com/tripdata/{YYYYMM}-citibike-tripdata.zip
```
