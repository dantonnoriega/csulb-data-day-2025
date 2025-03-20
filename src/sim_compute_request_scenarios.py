# %%
import plotnine as p9
import polars as pl
import numpy as np
import json
from datetime import datetime, timedelta
import random
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# %%
instance_profiles_dict = {
 'micro':  {'requests_per_hour': 100,  'cost_per_hour': 0.012},
 'small':  {'requests_per_hour': 250,  'cost_per_hour': 0.024},
 'medium': {'requests_per_hour': 600,  'cost_per_hour': 0.048},
 'large':  {'requests_per_hour': 1500, 'cost_per_hour': 0.096},
 'xlarge': {'requests_per_hour': 4000, 'cost_per_hour': 0.192}
}

# %% Define instance profiles
def create_instance_profiles():
    """Create profiles for 5 different compute instance types"""
    
    instance_profiles = pl.DataFrame({
        "instance_type": ["micro", "small", "medium", "large", "xlarge"],
        "requests_per_hour": [100, 250, 600, 1500, 4000],
        "cost_per_hour": [0.012, 0.024, 0.048, 0.096, 0.192]
    })
    
    return instance_profiles

# Generate datetime series for 6 months with hourly intervals
def generate_datetime_series():
    """Generate hourly timestamps for a 6-month period"""
    
    # Start date: Jan 1, 2023
    start_date = datetime(2023, 1, 1, 0, 0, 0)
    # End date: Jun 30, 2023 (6 months)
    end_date = datetime(2023, 6, 30, 23, 0, 0)
    
    # Generate hourly intervals
    hours = int((end_date - start_date).total_seconds() / 3600) + 1
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    
    return timestamps

# Create base request pattern with time-of-day, day-of-week, and monthly patterns
def generate_base_pattern(timestamps):
    """Generate a realistic base pattern for requests"""
    
    # Initialize request counts
    base_requests = []
    
    # Base hourly pattern (24 hours)
    hourly_pattern = np.array([
        0.3, 0.2, 0.15, 0.1, 0.1, 0.15,              # 00:00 - 05:00
        0.3, 0.6, 0.9, 1.2, 1.3, 1.4,                # 06:00 - 11:00
        1.5, 1.4, 1.3, 1.2, 1.1, 1.0,                # 12:00 - 17:00
        0.9, 0.8, 0.7, 0.6, 0.5, 0.4                 # 18:00 - 23:00
    ])
    
    # Day of week pattern (Mon=0, Sun=6)
    dow_pattern = np.array([1.2, 1.1, 1.0, 1.0, 1.1, 0.7, 0.5])
    
    # Monthly seasonality pattern
    monthly_pattern = np.array([0.8, 0.85, 0.9, 1.0, 1.1, 1.2])
    
    # Base request rate (average requests per hour)
    base_rate = 20000
    
    for ts in timestamps:
        hour_factor = hourly_pattern[ts.hour]
        day_factor = dow_pattern[ts.weekday()]
        month_factor = monthly_pattern[ts.month - 1]  # Jan=0, Jun=5
        
        # Calculate requests for this hour
        hourly_requests = base_rate * hour_factor * day_factor * month_factor
        
        # Add some random noise (±10%)
        noise = np.random.normal(1, 0.1)
        hourly_requests = max(0, hourly_requests * noise)
        
        base_requests.append(int(hourly_requests))
    
    return base_requests

# Generate scenarios: normal, high, and low
def generate_scenarios(timestamps, base_requests):
    """Generate different request scenarios based on the base pattern"""
    
    data = []
    
    # Normal scenario
    for ts, req in zip(timestamps, base_requests):
        data.append({"datetime": ts, "scenario": "normal", "requests": req})
    
    # High scenario: 50% more requests with occasional spikes
    for ts, req in zip(timestamps, base_requests):
        high_req = int(req * 1.5)
        
        # Add random spikes (5% chance of having a traffic spike)
        if random.random() < 0.05:
            high_req = int(high_req * random.uniform(1.5, 2.5))
            
        data.append({"datetime": ts, "scenario": "high", "requests": high_req})
    
    # Low scenario: 30% fewer requests
    for ts, req in zip(timestamps, base_requests):
        low_req = int(req * 0.7)
        data.append({"datetime": ts, "scenario": "low", "requests": low_req})
    
    # Convert to polars DataFrame
    df = pl.DataFrame(data)
    
    return df

# Create plots for visualizing the data
def create_plots(df):
    """Create visualization plots for the simulated data"""
    
    # Filter to just one week for detailed view
    one_week = df.filter(
        (pl.col("datetime") >= datetime(2023, 3, 1)) & 
        (pl.col("datetime") < datetime(2023, 3, 8))
    )
    
    # Convert to pandas for plotnine
    one_week_pd = one_week.to_pandas()
    df_pd = df.to_pandas()
    
    # Create a 1-week detailed plot
    p_week = (
        p9.ggplot(one_week_pd, p9.aes(x="datetime", y="requests", color="scenario")) +
        p9.geom_line() +
        p9.labs(
            title="Request Patterns (1 Week in March)",
            x="Date",
            y="Requests per Hour"
        ) +
        p9.theme_minimal()
    )
    
    # Create a monthly average plot
    monthly_avg = df.with_columns([
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.year().alias("year")
    ]).group_by(["month", "year", "scenario"]).agg(
        pl.col("requests").mean().alias("avg_requests")
    ).sort(["year", "month", "scenario"])
    
    monthly_avg_pd = monthly_avg.to_pandas()
    monthly_avg_pd["month_name"] = monthly_avg_pd["month"].apply(
        lambda m: datetime(2023, m, 1).strftime("%B")
    )
    
    p_monthly = (
        p9.ggplot(monthly_avg_pd, p9.aes(x="month_name", y="avg_requests", fill="scenario")) +
        p9.geom_col(position="dodge") +
        p9.labs(
            title="Average Hourly Requests by Month",
            x="Month",
            y="Average Requests per Hour"
        ) +
        p9.theme_minimal() +
        p9.theme(axis_text_x=p9.element_text(angle=45, hjust=1))
    )
    
    return p_week, p_monthly

# %%
print("Generating compute resource simulation...")

# Create instance profiles
instance_profiles = create_instance_profiles()
print("Instance profiles created:")
print(instance_profiles)

# %% Generate datetime series
timestamps = generate_datetime_series()
print(f"Generated {len(timestamps)} hourly timestamps over 6 months")

# %% Generate base pattern
base_requests = generate_base_pattern(timestamps)

# %% Generate scenarios
df = generate_scenarios(timestamps, base_requests)
print("Generated request scenarios:")
print(df.head())
print(f"Total rows: {df.height}")

# %% Save data to CSV
print("Saving data to CSV files...")
df.write_csv(Path('../data/compute_request_scenarios.csv'))
instance_profiles.write_csv(Path('../data/compute_instance_profiles.csv'))

# Write the JSON data to a file
# with open(Path('../data/compute_instance_profiles.json'), "w") as file:
#     json.dump(instance_profiles_dict, file, indent=4)

# %% create plots
print("Creating visualization plots...")
create_plots(df)

# %% Summary statistics by scenario
summary = df.group_by("scenario").agg([
    pl.col("requests").mean().alias("mean_requests"),
    pl.col("requests").min().alias("min_requests"),
    pl.col("requests").max().alias("max_requests"),
    pl.col("requests").quantile(0.95).alias("p95_requests"),
])

print("\nSummary statistics by scenario:")
print(summary)

print("\nSimulation completed successfully!")
# %%
