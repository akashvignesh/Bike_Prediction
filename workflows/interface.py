import os
import logging
from pathlib import Path
import hopsworks
from dotenv import load_dotenv
import joblib
import pandas as pd
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_hopsworks_project():
    try:
        return hopsworks.login(
            project=os.getenv("HOPSWORKS_PROJECT_NAME"),
            api_key_value=os.getenv("HOPSWORKS_API_KEY")
        )
    except Exception as e:
        logger.error(f"Failed to login to Hopsworks: {e}")
        raise

# Connect to Hopsworks
logger.info("🔗 Connecting to Hopsworks...")
try:
    project = get_hopsworks_project()
    feature_store = project.get_feature_store()
except Exception as e:
    logger.error(f"Failed to connect to Hopsworks or get feature store: {e}")
    raise

# Get or create feature group
feature_group_name = "bike_prediction_flow_202504"
feature_group_version = int(os.getenv("FEATURE_GROUP_VERSION", 1))
try:
    feature_group = feature_store.get_feature_group(
        name=feature_group_name,
        version=feature_group_version
    )
    logger.info(f"Feature group '{feature_group_name}' v{feature_group_version} retrieved successfully.")
except:
    logger.info(f"Feature group '{feature_group_name}' v{feature_group_version} not found, creating...")
    try:
        feature_group = feature_store.create_feature_group(
            name=feature_group_name,
            version=feature_group_version,
            description="time-series data for bike at six hour frequency for April 2025",
            primary_key=["location_id", "pickup_hour"],
            event_time="pickup_hour"
        )
        logger.info(f"Feature group '{feature_group_name}' v{feature_group_version} created successfully.")
    except Exception as e:
        logger.error(f"Failed to create or retrieve feature group: {e}")
        raise

# Check if feature group has data
try:
    feature_group_stats = feature_group.read().shape
    if feature_group_stats[0] == 0:
        logger.error(f"Feature group '{feature_group_name}' v{feature_group_version} is empty. Please populate it with data.")
        raise ValueError("Empty feature group")
    logger.info(f"Feature group contains {feature_group_stats[0]} rows.")
except Exception as e:
    logger.error(f"Failed to read feature group data: {e}")
    raise

# Create or retrieve feature view
feature_view_name = "bike_prediction_view_flow_202504"
feature_view_version = int(os.getenv("FEATURE_VIEW_VERSION", 1))

try:
    feature_view = feature_store.get_feature_view(
        name=feature_view_name,
        version=feature_view_version
    )
    logger.info(f"Feature view '{feature_view_name}' v{feature_view_version} retrieved successfully.")
except:
    logger.info(f"Feature view '{feature_view_name}' v{feature_view_version} not found, creating...")
    try:
        feature_view = feature_store.create_feature_view(
            name=feature_view_name,
            version=feature_view_version,
            query=feature_group.select_all()
        )
        logger.info(f"Feature view '{feature_view_name}' v{feature_view_version} created successfully.")
    except Exception as e:
        logger.error(f"Failed to create feature view: {e}")
        raise

# Verify feature view is not None
if feature_view is None:
    logger.error(f"Feature view '{feature_view_name}' v{feature_view_version} could not be created or retrieved.")
    raise ValueError("Feature view is None")

# Load model from model registry
logger.info("📦 Loading model from registry...")
model_registry = project.get_model_registry()
try:
    models = model_registry.get_models(name="bike_demand_predictor_next_hour")
    if not models:
        raise ValueError("No models found with name 'bike_demand_predictor_next_hour'")
    model = max(models, key=lambda model: model.version)
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / "lightgbm_bikeride_model.joblib")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Load feature view data
logger.info("📥 Loading feature view data...")
try:
    ts_data, _ = feature_view.training_data(
        description="time_series_six_hourly_bike_ride_202504"
    )
    if ts_data.empty:
        logger.error("Feature view returned empty training data.")
        raise ValueError("Empty training data")
except Exception as e:
    logger.error(f"Failed to load training data: {e}")
    raise

# Preprocess data
logger.info("🧹 Preprocessing data...")
ts_data["location_id"] = ts_data["location_id"].astype(str).str.replace('.', '', regex=False)
ts_data["pickup_hour"] = pd.to_datetime(ts_data["pickup_hour"])
valid_ids = {"614005", "590514", "532903"}
ts_data = ts_data[ts_data["location_id"].isin(valid_ids)]

# Setup for prediction
full_df = ts_data.copy()
predictions = []
future_dates = pd.date_range(
    start="2025-04-01 00:00:00",
    end="2025-04-30 18:00:00",
    freq="6H",
    tz="UTC"
)
location_ids = sorted(valid_ids)
reg_features = [f"target_lag_{i+1}" for i in range(112)] + ["hour", "day_of_week", "month", "is_weekend", "location_id"]

logger.info("🔮 Generating predictions for April 2025...")

# Rolling prediction loop
for ts in future_dates:
    for loc in location_ids:
        hist = full_df[full_df["location_id"] == loc].sort_values("pickup_hour").tail(112)
        if len(hist) < 112:
            logger.warning(f"Skipping prediction for {loc} at {ts}: insufficient history ({len(hist)} rows)")
            continue

        # Create lag features
        feature_row = {
            f"target_lag_{i+1}": hist.iloc[-(i+1)]["target"] for i in range(112)
        }

        # Add time-based features
        feature_row["hour"] = ts.hour
        feature_row["day_of_week"] = ts.dayofweek
        feature_row["month"] = ts.month
        feature_row["is_weekend"] = int(ts.dayofweek in [5, 6])
        feature_row["pickup_hour"] = ts
        feature_row["location_id"] = loc

        # Prepare DataFrame for prediction
        X_pred = pd.DataFrame([feature_row])[reg_features]
        X_pred["location_id"] = X_pred["location_id"].astype(float)

        # Predict
        try:
            pred = model.predict(X_pred)[0]
        except Exception as e:
            logger.error(f"Prediction failed for {loc} at {ts}: {e}")
            continue

        # Store prediction
        predictions.append({
            "pickup_hour": ts,
            "location_id": loc,
            "predicted_rides": round(pred)
        })

        # Append predicted row to history
        full_df = pd.concat([
            full_df,
            pd.DataFrame([{
                **feature_row,
                "target": pred
            }])
        ], ignore_index=True)

logger.info("✅ April 2025 predictions complete.")

# Save predictions to CSV
pred_df = pd.DataFrame(predictions)
output_csv = "flow/bike_predictions_202504_6hr_flow.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
pred_df.to_csv(output_csv, index=False)
logger.info(f"📁 Saved predictions to {output_csv}")

# Create and save to feature group
try:
    fg_pred = feature_store.get_or_create_feature_group(
        name="bike_demand_predictions_flow_202504",
        version=1,
        description="6-hourly predicted demand for April 2025",
        primary_key=["pickup_hour", "location_id"],
        event_time="pickup_hour"
    )
    fg_pred.insert(pred_df, write_options={"wait_for_job": True})
    logger.info("✅ Predictions uploaded to Hopsworks Feature Group: bike_demand_predictions_flow_202504 v1")
except Exception as e:
    logger.error(f"Failed to upload predictions to feature group: {e}")
    raise