import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import hopsworks
import pandas as pd
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Step 3: Fetch raw data
logger.info("📥 Fetching raw data...")
import requests
import os
def download_citibike_data(year, month):
    month_str = f"{month:02d}"
    #url = f"https://s3.amazonaws.com/tripdata/JC-{year}{month_str}-citibike-tripdata.csv.zip" 
    url = f"https://s3.amazonaws.com/tripdata/202504-citibike-tripdata.zip" 
    zip_file_path = os.path.join( "raw_data","2024", f"{year}{month_str}-citibike-tripdata.csv.zip")
    
    if os.path.exists(zip_file_path):
        print(f"File for {year}-{month_str} already exists")
        return zip_file_path
    
    print(f"Downloading data for {year}-{month_str}...")
    response = requests.get(url)
    response.raise_for_status()
    with open(zip_file_path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {zip_file_path}")
    return zip_file_path

# Download data for 2024 (12 months)
year = 2025
zip_paths = [download_citibike_data(year, "04")]
print("All data downloaded successfully")

#2
import os
import zipfile
import pandas as pd

# Set paths
folder_path = "raw_data/flow"
output_folder = "filter_data_2024/flow"
os.makedirs(output_folder, exist_ok=True)

# File list
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".zip")])

# Track column structure
column_set = None
matching_files = []
null_counts = {}
row_counts = {}

# Process ZIPs
for file_name in file_list:
    month = file_name.split("-")[0]
    file_path = os.path.join(folder_path, file_name)
    monthly_df_list = []

    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        csv_files = [f for f in zip_ref.namelist() if f.endswith(".csv") and "__MACOSX" not in f]
        if not csv_files:
            print(f"⚠️ No CSVs in {file_name}")
            continue

        for csv_name in csv_files:
            with zip_ref.open(csv_name) as csv_file:
                try:
                    df = pd.read_csv(csv_file, low_memory=False)

                    # Drop index column if present
                    if "Unnamed: 0" in df.columns:
                        df.drop(columns=["Unnamed: 0"], inplace=True)

                    # Drop any column with "duplicate" in the name
                    dup_cols = [col for col in df.columns if "duplicate" in col.lower()]
                    if dup_cols:
                        print(f"⚠️ Dropping duplicate columns in {csv_name}: {dup_cols}")
                        df.drop(columns=dup_cols, inplace=True)

                    # Check column consistency
                    if column_set is None:
                        column_set = set(df.columns)
                        matching_files.append(file_name)
                    elif set(df.columns) == column_set:
                        matching_files.append(file_name)
                    else:
                        print(f"❌ {file_name} → {csv_name} has different columns, skipping.")
                        continue

                    # Save raw stats
                    row_counts[csv_name] = len(df)
                    null_counts[csv_name] = df.isnull().sum().to_dict()

                    monthly_df_list.append(df)

                except Exception as e:
                    print(f"❌ Failed to process {csv_name} in {file_name}: {e}")

    # Combine and save cleaned data
    if monthly_df_list:
        combined_df = pd.concat(monthly_df_list, ignore_index=True)
        cleaned_df = combined_df.dropna()

        output_csv_name = f"{month}_filtered.csv"
        output_path = os.path.join(output_folder, output_csv_name)
        cleaned_df.to_csv(output_path, index=False)

        print(f"✅ Saved: {output_csv_name} with {len(cleaned_df)} cleaned rows")

# Summary
print("\n--- Summary ---")
print("\nFiles with matching columns:")
print(matching_files)

print("\nRow counts per original CSV file:")
for file, count in row_counts.items():
    print(f"{file}: {count} rows")

print("\nNull values per original CSV file:")
for file, counts in null_counts.items():
    print(f"\n{file}:")
    for col, val in counts.items():
        print(f"  {col}: {val}")

#3
import os
import pandas as pd

# Set the folder where your .csv files are located
folder_path = "filter_data_2024"
file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])

column_set = None
matching_files = []
null_counts = {}
row_counts = {}

for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    try:
        df = pd.read_csv(file_path)

        # Initialize or compare column set
        if column_set is None:
            column_set = set(df.columns)
            matching_files.append(file_name)
        elif set(df.columns) == column_set:
            matching_files.append(file_name)
        else:
            print(f"❌ {file_name} has different columns")

        # Count null values per column
        null_counts[file_name] = df.isnull().sum().to_dict()

        # Count number of rows
        row_counts[file_name] = len(df)

    except Exception as e:
        print(f"⚠️ Error reading {file_name}: {e}")

# Display summary
print("\n✅ Files with matching columns:")
print(matching_files)

print("\n📊 Row counts per file:")
for file, count in row_counts.items():
    print(f"{file}: {count} rows")

print("\n🧼 Null values per file:")
for file, counts in null_counts.items():
    print(f"\n{file}:")
    for col, val in counts.items():
        print(f"  {col}: {val}")

#4
import os
import pandas as pd
from collections import Counter

# Path where your 202401_filtered.csv to 202412_filtered.csv files are stored
folder_path = "filter_data_2024/flow"

# To store top 3 from each month
monthly_top_stations = {}
yearly_counter = Counter()

# Loop through each month
for month in range(1, 13):
    file_name = f"2025{month:02d}_filtered.csv"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, usecols=["start_station_id"])
        df = df.dropna(subset=["start_station_id"])
        df["start_station_id"] = df["start_station_id"].astype(str)

        top3 = df["start_station_id"].value_counts().head(3)
        monthly_top_stations[file_name] = top3

        yearly_counter.update(top3.to_dict())

# Convert monthly results to DataFrame
monthly_df = pd.DataFrame(monthly_top_stations).fillna(0).astype(int)

# Get top 3 for the year
top_3_year = pd.DataFrame(yearly_counter.most_common(3), columns=["start_station_id", "total_count"])

# Save or print the result
print("Monthly Top 3 Stations:")
print(monthly_df)

print("\nTop 3 Stations for the Year:")
print(top_3_year)

import os
import pandas as pd

# Define your folder path and top station IDs
folder_path = "filter_data_2024"
top_station_ids = {"6140.05", "5905.14", "5329.03"}

# List to collect filtered DataFrames
filtered_dfs = []

for month in range(1, 13):
    file_name = f"2025{month:02d}_filtered.csv"
    file_path = os.path.join(folder_path, file_name)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        # Convert all ID fields to string to avoid parquet errors
        for col in ["ride_id", "start_station_id", "end_station_id"]:
            df[col] = df[col].astype(str)

        # Filter based on start_station_id
        df_filtered = df[df["start_station_id"].isin(top_station_ids)]
        filtered_dfs.append(df_filtered)

# Combine all filtered data
combined_df = pd.concat(filtered_dfs, ignore_index=True)

# Save as parquet
combined_df.to_parquet("BikeRide2024Top3Locationflow.parquet", index=False)

print("Saved as BikeRide2024Top3Locationflow.parquet")

#6
def load_and_preprocess_data(file_path):
    print("📥 Loading dataset...")
    df = pd.read_parquet(file_path)
    print("✅ Dataset loaded successfully.")

    print("🕒 Converting datetime columns...")
    df['started_at'] = pd.to_datetime(df['started_at'], format='mixed')
    df['ended_at'] = pd.to_datetime(df['ended_at'], format='mixed')  
    df['pickup_hour'] = df['started_at'].dt.floor('6H')
    df['location_id'] = df['start_station_id'].astype(str)

    print("⏱️ Calculating ride duration...")
    df['duration_minutes'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60.0

    print("📊 Aggregating target (trip counts)...")
    ride_counts = df.groupby(['pickup_hour', 'location_id']).size().reset_index(name='target')

    print("🔁 Creating 112 lag features (28 days × 4 bins/day)...")
    lagged_data = []
    for loc in ride_counts['location_id'].unique():
        loc_df = ride_counts[ride_counts['location_id'] == loc].sort_values('pickup_hour')
        for lag in range(1, 113):
            loc_df[f'target_lag_{lag}'] = loc_df['target'].shift(lag)
        lagged_data.append(loc_df)

    df_lagged = pd.concat(lagged_data)

    print("📅 Extracting time-based features...")
    df_lagged['hour'] = df_lagged['pickup_hour'].dt.hour
    df_lagged['day_of_week'] = df_lagged['pickup_hour'].dt.dayofweek
    df_lagged['month'] = df_lagged['pickup_hour'].dt.month
    df_lagged['is_weekend'] = df_lagged['day_of_week'].isin([5, 6]).astype(int)

    print("🧹 Dropping missing values...")
    df_lagged = df_lagged.dropna()

    print("✅ Preprocessing complete.")
    return df_lagged
file_path = "BikeRide2024Top3Locationflow.parquet"
    
df_transformed = load_and_preprocess_data(file_path)
df_transformed.to_parquet("transformeddata2024flow.parquet", index=False)


import os
import config as config
from pathlib import Path
import hopsworks
from dotenv import load_dotenv

load_dotenv()
project = hopsworks.login(
    project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
feature_store = project.get_feature_store()
feature_group=feature_store.get_or_create_feature_group(
    name="Bike prediction from flow",
    version=1,
    description= "Time-series Data for Bike at six hour frequency from flow",
    primary_key=["location_id","pickup_hour"],
    event_time="pickup_hour"
)
import pandas as pd
df = pd.read_parquet("transformeddata2024flow.parquet")
feature_group.insert(df,write_options={"wait_for_job":False})
