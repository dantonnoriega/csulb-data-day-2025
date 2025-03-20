# %%
import plotnine as p9
import polars as pl
import numpy as np
from datetime import datetime, timedelta
import random
import os
from pathlib import Path
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_rows(40)
pl.Config.set_fmt_str_lengths(100)
pl.Config.set_tbl_width_chars(200)
pl.Config(
  tbl_formatting="ASCII_MARKDOWN",
  tbl_hide_column_data_types=True,
  tbl_hide_dataframe_shape=True
)

# %% read data
scenarios_df = pl.read_csv(Path('../data/compute_request_scenarios.csv'), try_parse_dates=True)
instance_profiles = pl.read_csv(Path('../data/compute_instance_profiles.csv'))

# %% Define budget and allocations for each strategy
# Set a single budget constraint for simplicity
budget_hourly = 3.0  # $4 per hour = ~$3,000 per month

print(f"Budget constraint: ${budget_hourly:.2f}/hour (${budget_hourly * 24 * 30:.2f}/month)")

# Define the allocations for each strategy
# These are pre-determined for simplicity

# Strategy 1: Cost Efficiency - Many smaller instances
cost_efficiency_allocation = {
    "micro": 25,
    "small": 20,
    "medium": 17,
    "large": 6, 
    "xlarge": 1 
} 

# Strategy 2: Performance - Fewer, more powerful instances
performance_allocation = {
    "micro": 0, 
    "small": 0, 
    "medium": 0,
    "large": 6, 
    "xlarge": 15
} 

# Strategy 3: Balanced - Mix of instance types
balanced_allocation = {
    "micro": 10,
    "small": 15,
    "medium": 8,
    "large": 5, 
    "xlarge": 7 
} 

# Calculate capacity and cost for each allocation
print("\nCalculating capacity and cost for each strategy:")

strategies = ["cost_efficiency", "performance", "balanced"]
allocations = [cost_efficiency_allocation, performance_allocation, balanced_allocation]

for strategy, allocation in zip(strategies, allocations):
    # Calculate total capacity
    hourly_capacity = round(sum(
        count * instance_profiles.filter(pl.col("instance_type") == instance_type)["requests_per_hour"].item()
        for instance_type, count in allocation.items()
    ), 3)
    
    # Calculate total cost
    total_cost = round(sum(
        count * instance_profiles.filter(pl.col("instance_type") == instance_type)["cost_per_hour"].item()
        for instance_type, count in allocation.items()
    ), 3)
    
    print(f"\nStrategy: {strategy}")
    print(f"  Allocation: {allocation}")
    print(f"  Total capacity: {hourly_capacity} requests/hour")
    print(f"  Hourly cost: ${total_cost:.4f}")
    print(f"  Monthly cost: ${total_cost * 24 * 30:.2f}")
    print(f"  Cost per 1000 requests: ${total_cost / hourly_capacity * 1000:.4f}")

# %% Calculate service metrics for each scenario and strategy
results = []

for scenario in scenarios_df["scenario"].unique():
    scenario_data = scenarios_df.filter(pl.col("scenario") == scenario)
    
    for strategy, allocation in zip(strategies, allocations):
        # Calculate total capacity
        hourly_capacity = round(sum(
            count * instance_profiles.filter(pl.col("instance_type") == instance_type)["requests_per_hour"].item()
            for instance_type, count in allocation.items()
        ), 3)
        
        # Calculate total cost
        total_cost = round(sum(
            count * instance_profiles.filter(pl.col("instance_type") == instance_type)["cost_per_hour"].item()
            for instance_type, count in allocation.items()
        ), 3)
        
        # Calculate unserved requests (where demand > capacity)
        scenario_data_with_metrics = scenario_data.with_columns([
            pl.lit(hourly_capacity).alias("capacity"),
            pl.when(pl.col("requests") > hourly_capacity)
                .then(pl.col("requests") - hourly_capacity)
                .otherwise(0).alias("unserved_requests"),
            pl.col('datetime').dt.strftime("%Y-%m").alias('ym')
        ])
        
        # Calculate service rate
        total_requests = scenario_data_with_metrics["requests"].sum()
        unserved_requests = scenario_data_with_metrics["unserved_requests"].sum()
        service_rate = 1 - (unserved_requests / total_requests)
        
        # Calculate hours over capacity
        hours_over_capacity = scenario_data_with_metrics.filter(pl.col("requests") > pl.col("capacity")).height
        total_hours = scenario_data_with_metrics.height
        over_capacity_percent = hours_over_capacity / total_hours * 100
        
        # Calculate average utilization
        avg_utilization = scenario_data_with_metrics["requests"].sum() / (total_hours * hourly_capacity) * 100
        
        # Monthly cost
        n_months = scenario_data_with_metrics['ym'].unique().len()
        avg_monthly_cost = total_cost * total_hours / n_months
        
        # Add results
        results.append({
            "scenario": scenario,
            "strategy": strategy,
            "hourly_capacity": hourly_capacity,
            "hourly_cost": total_cost,
            "monthly_cost": avg_monthly_cost,
            "service_rate": service_rate,
            "over_capacity_percent": over_capacity_percent,
            "avg_utilization": avg_utilization
        })

# Convert to DataFrame
results_df = pl.DataFrame(results)
print(results_df)

(
  results_df
  .select(['strategy', 'hourly_capacity', 'hourly_cost'])
  .unique()
  .write_csv(Path('../data/compute_allocation_strategies.csv'))
)

# %% Analyze the results
# Print summary table
print("\nSummary of results:")
summary_table = results_df.select([
    "scenario", 
    "strategy", 
    "service_rate", 
    "monthly_cost", 
    "over_capacity_percent", 
    "avg_utilization"
]).sort(['scenario', 'strategy'])

print(summary_table)

# %% Create visualizations
# Plot 1: Service Rate by Strategy and Scenario
p1 = (
    p9.ggplot(results_df, p9.aes(x='strategy', y='service_rate', fill='scenario')) +
    p9.geom_col(position='dodge') +
    p9.labs(
        title="Service Rate by Strategy and Scenario",
        x="Strategy",
        y="Service Rate",
        fill="Scenario"
    ) +
    p9.scale_y_continuous(labels=lambda x: [f"{i*100:.1f}%" for i in x]) +
    p9.theme_light()
)
p1.show()

# %% Plot 2: Cost vs. Service Rate
p2 = (
    p9.ggplot(results_df, p9.aes(x='monthly_cost', y='service_rate', color='strategy', shape='scenario')) +
    p9.geom_point(size=3, alpha = .5) +
    p9.labs(
        title="Monthly Cost vs. Service Rate",
        x="Monthly Cost ($)",
        y="Service Rate",
        color="Strategy",
        shape="Scenario"
    ) +
    p9.scale_y_continuous(labels=lambda x: [f"{i*100:.1f}%" for i in x]) +
    p9.theme_light()
)
p2.show()

# %% Plot 3: Utilization by Strategy and Scenario
p3 = (
    p9.ggplot(results_df, p9.aes(x='strategy', y='avg_utilization', fill='scenario')) +
    p9.geom_col(position='dodge') +
    p9.labs(
        title="Average Utilization by Strategy and Scenario",
        x="Strategy",
        y="Average Utilization",
        fill="Scenario"
    ) +
    p9.scale_y_continuous(labels=lambda x: [f"{i:.1f}%" for i in x]) +
    p9.theme_light()
)
p3.show()

# %% Final insights

# Best strategy for each scenario (by service rate)
print(summary_table)

print("\nObservations:")
print("1. The performance strategy provides the highest service rate across all scenarios")
print("2. The cost efficiency strategy provides the best value (service rate per dollar)")
print("3. No single strategy is optimal for all scenarios and business priorities")
# %%
