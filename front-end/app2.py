import os
import streamlit as st
import pandas as pd
import plotly.express as px
import hopsworks

# Page config
st.set_page_config(page_title="MAE Viewer", layout="wide")
st.title("📉 Mean Absolute Error (MAE) - Bike Ride Prediction")

# Hopsworks login
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

# Get feature groups
actual_fg = fs.get_feature_group(
    name=os.getenv("FEATURE_GROUP_NAME"),
    version=int(os.getenv("FEATURE_GROUP_VERSION", "1"))
)

pred_fg = fs.get_feature_group(
    name="bike_demand_predictions",
    version=1
)

# Load data
actual_df = actual_fg.read()
actual_df["pickup_hour"] = pd.to_datetime(actual_df["pickup_hour"], utc=True)
pred_df = pred_fg.read()
pred_df["pickup_hour"] = pd.to_datetime(pred_df["pickup_hour"], utc=True)

# Filter for required columns
actual_df = actual_df[["pickup_hour", "location_id", "target"]]
pred_df = pred_df[["pickup_hour", "location_id", "predicted_rides"]]

# Sidebar: Location filter
all_locations = sorted(set(actual_df["location_id"]).union(pred_df["location_id"]))
selected_location = st.sidebar.selectbox("Select Location ID", all_locations)

# Filter by location
actual_filtered = actual_df[actual_df["location_id"] == selected_location]
pred_filtered = pred_df[pred_df["location_id"] == selected_location]

# Merge actual and predicted
merged = pd.merge(actual_filtered, pred_filtered, on=["pickup_hour", "location_id"], how="inner")
merged["absolute_error"] = abs(merged["target"] - merged["predicted_rides"])

# Group by pickup_hour to get MAE over time
mae_by_hour = merged.groupby("pickup_hour")["absolute_error"].mean().reset_index()
mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

# Plot MAE
fig = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"MAE Over Time for Location {selected_location}",
    labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error"},
    markers=True
)

# Display
st.plotly_chart(fig, use_container_width=True)
st.metric("📌 Average MAE", round(mae_by_hour["MAE"].mean(), 2))
