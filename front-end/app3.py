import streamlit as st
import pandas as pd
import os
import hopsworks
import altair as alt
HOPSWORKS_API_KEY="LVmrhMHM87zqUPpc.KSnbzXbEPo0sGiqmKTuKbWtM6dNDJAGRCLURFm8tiJF75xz1ye4kNy6d3zP8mQjR"
HOPSWORKS_PROJECT_NAME="s3akash" 

FEATURE_GROUP_NAME = "time_series_six_hourly_feature_group_bike"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "time_series_six_hourly_feature_view_bike"
FEATURE_VIEW_VERSION = 1
# Set page configuration
st.set_page_config(page_title="Bike Demand Predictions", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stSelectbox {font-weight: bold;}
    .title {color: #2c3e50; font-size: 2.5em;}
    </style>
""", unsafe_allow_html=True)

# Title with emoji
st.markdown('<p class="title">🚴‍♂️ 6-Hourly Bike Ride Demand: 2024 vs 2025 🚴‍♀️</p>', unsafe_allow_html=True)

from dotenv import load_dotenv

# Load environment variables and connect to Hopsworks
load_dotenv()
project = hopsworks.login(
    project= HOPSWORKS_PROJECT_NAME,
    api_key_value= HOPSWORKS_API_KEY
)
fs = project.get_feature_store()

# Load environment variables
FEATURE_GROUP_NAME =  FEATURE_GROUP_NAME
FEATURE_GROUP_VERSION = FEATURE_GROUP_VERSION

# Load actual 2024 data
feature_group = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)
actual_df = feature_group.read()
actual_df["pickup_hour"] = pd.to_datetime(actual_df["pickup_hour"], utc=True)

# Group 2024 actuals to get 'target' count
actual_df_grouped = actual_df.groupby(["pickup_hour", "location_id"])["target"].sum().reset_index()

# Load 2025 predictions
fg_pred = fs.get_feature_group(name="bike_demand_predictions", version=1)
pred_df = fg_pred.read()
pred_df["pickup_hour"] = pd.to_datetime(pred_df["pickup_hour"], utc=True)

# Clean location_id: Remove '.' and convert to integer if possible
def clean_location_id(loc_id):
    # Convert to string and remove '.' and any trailing characters
    loc_str = "".join(str(loc_id).split("."))
    # Convert to integer if possible, otherwise keep as string
    try:
        return int(loc_str)
    except ValueError:
        return loc_str

# Apply cleaning to both dataframes
actual_df_grouped["location_id"] = actual_df_grouped["location_id"].apply(clean_location_id)
pred_df["location_id"] = pred_df["location_id"].apply(clean_location_id)

# Get unique location IDs
location_ids = sorted(set(actual_df_grouped["location_id"]).union(set(pred_df["location_id"])))

# Sidebar with image and filter
st.sidebar.header("Filter Options")
selected_loc = st.sidebar.selectbox("Select Location ID", location_ids)

# Filter both datasets
actual_filtered = actual_df_grouped[actual_df_grouped["location_id"] == selected_loc]
pred_filtered = pred_df[pred_df["location_id"] == selected_loc]

# Rename for chart
actual_filtered = actual_filtered.rename(columns={"target": "ride_count"})
pred_filtered = pred_filtered.rename(columns={"predicted_rides": "ride_count"})

# Create two tabs
tab1, tab2 = st.tabs(["📅 2024 Actual Demand", "🔮 2025 Predicted Demand"])

# 2024 Chart (Tab 1)
with tab1:
    chart_2024 = alt.Chart(actual_filtered).mark_line(point=True).encode(
        x=alt.X("pickup_hour:T", title="Time"),
        y=alt.Y("ride_count:Q", title="Ride Count"),
        color=alt.value("#1f77b4"),  # Blue color for 2024
        tooltip=[
            alt.Tooltip("pickup_hour:T", title="Time"),
            alt.Tooltip("ride_count:Q", title="Ride Count")
        ]
    ).properties(
        title=f"🚲 2024 Ride Demand for Location ID: {selected_loc}",
        width=1000,
        height=400
    ).interactive()

    st.altair_chart(chart_2024, use_container_width=True)

# 2025 Chart (Tab 2)
with tab2:
    chart_2025 = alt.Chart(pred_filtered).mark_line(point=True).encode(
        x=alt.X("pickup_hour:T", title="Time"),
        y=alt.Y("ride_count:Q", title="Ride Count"),
        color=alt.value("#ff7f0e"),  # Orange color for 2025
        tooltip=[
            alt.Tooltip("pickup_hour:T", title="Time"),
            alt.Tooltip("ride_count:Q", title="Ride Count")
        ]
    ).properties(
        title=f"🚲 2025 Predicted Ride Demand for Location ID: {selected_loc}",
        width=1000,
        height=400
    ).interactive()

    st.altair_chart(chart_2025, use_container_width=True)