#!/usr/bin/env python
"""Build and execute the EDA notebook.

Run from the project root::

    python scripts/build_eda_notebook.py

The notebook is intentionally generated from source so it stays reproducible
and reviewable in Git.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "01_eda.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


CELLS = [
    md(
        """
# Exploratory Data Analysis — Logistics & Fleet Operations

**Project 24 · Comprehensive Logistics and Fleet Management Platform**

This notebook profiles the cleaned 5,000-trip fleet log, checks data quality,
and explores delivery performance, fuel efficiency, driver scores and injected
anomalies. It is reproducible: the dataset is rebuilt by
`scripts/run_pipeline.py` with `seed=42`.
"""
    ),
    md("## 1. Setup and data loading"),
    code(
        """
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from fleet import config

pio.templates.default = "plotly_white"
config.FIGURES.mkdir(parents=True, exist_ok=True)

dataset = pd.read_csv(config.PROCESSED_DATASET)
metadata = pd.read_csv(config.TRIP_METADATA)
anomalies = pd.read_csv(config.ANOMALY_LABELS)

trips = (
    dataset.merge(metadata, on="trip_id", how="left")
    .merge(anomalies, on="trip_id", how="left")
)
trips["is_anomaly"] = trips["anomaly_type"].notna()
trips["kmpl"] = trips["distance_km"] / trips["fuel_L"]
trips["month"] = pd.to_datetime(trips["trip_start_time"]).dt.month
trips["month_name"] = pd.to_datetime(trips["trip_start_time"]).dt.strftime("%b")

print(f"Fleet log: {dataset.shape[0]:,} rows x {dataset.shape[1]} columns")
print(f"Metadata:  {metadata.shape[0]:,} rows")
print(f"Anomalies: {anomalies.shape[0]:,} labelled rows")
dataset.head()
"""
    ),
    md("## 2. Data quality profile"),
    code(
        """
quality = pd.DataFrame(
    {
        "dtype": dataset.dtypes.astype(str),
        "missing": dataset.isna().sum(),
        "unique_values": dataset.nunique(),
        "example": [dataset[column].iloc[0] for column in dataset.columns],
    }
)
quality
"""
    ),
    code(
        """
print(f"Rows:                 {len(dataset):,}")
print(f"Columns:              {len(dataset.columns)}")
print(f"Missing values:       {int(dataset.isna().sum().sum())}")
print(f"Duplicate trip IDs:   {int(dataset.trip_id.duplicated().sum())}")
print(f"Same-city trips:      {int((dataset.origin == dataset.destination).sum())}")
print(f"Vehicles / drivers:   {dataset.vehicle_id.nunique()} / {dataset.driver_id.nunique()}")
print(f"Date range:           {metadata.trip_start_time.min()} to {metadata.trip_start_time.max()}")
print(f"On-time share:        {(dataset.delivery_status == 'On_Time').mean():.2%}")
print(f"Anomaly share:        {len(anomalies) / len(dataset):.2%}")
"""
    ),
    md(
        """
**Quality verdict.** The processed dataset has the exact nine columns from the
brief, no missing values, unique trip IDs, valid city pairs and a chronological
2025 timestamp range. The original 500 rows are retained in
`data/raw/dataset_original.csv`; the remaining 4,500 rows are seeded synthetic
augmentation. Anomaly labels are kept in a separate file to prevent leakage.
"""
    ),
    md("## 3. Delivery performance"),
    code(
        """
status_order = ["On_Time", "Delayed", "Failed"]
status_counts = dataset.delivery_status.value_counts().reindex(status_order)
colors = ["#2E8B57", "#E6A817", "#C0392B"]

fig = px.bar(
    x=status_order,
    y=status_counts.values,
    color=status_order,
    color_discrete_sequence=colors,
    text=[f"{value:,} ({value / len(dataset):.1%})" for value in status_counts.values],
    labels={"x": "Delivery status", "y": "Trips"},
    title="Delivery status distribution (n=5,000)",
)
fig.update_traces(textposition="outside")
fig.update_layout(showlegend=False, yaxis_range=[0, status_counts.max() * 1.18])
fig.write_image(config.FIGURES / "01_delivery_status.png", width=900, height=500, scale=2)
fig.show()
"""
    ),
    code(
        """
monthly = (
    trips.groupby("month")
    .agg(
        trips=("trip_id", "count"),
        on_time=("delivery_status", lambda series: (series == "On_Time").mean()),
        avg_distance=("distance_km", "mean"),
    )
    .reset_index()
)

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_bar(
    x=monthly["month"],
    y=monthly["trips"],
    name="Trips",
    marker_color="#4C78A8",
)
fig.add_trace(
    go.Scatter(
        x=monthly["month"],
        y=monthly["on_time"] * 100,
        name="On-time %",
        mode="lines+markers",
        line=dict(color="#2E8B57", width=3),
    ),
    secondary_y=True,
)
fig.update_layout(
    title="Monthly trip volume and on-time performance",
    xaxis_title="Month",
    yaxis_title="Trips",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
fig.update_yaxes(title_text="On-time %", range=[70, 100], secondary_y=True)
fig.write_image(config.FIGURES / "02_monthly_performance.png", width=1000, height=520, scale=2)
fig.show()
"""
    ),
    md(
        """
**Observation.** On-time performance is calibrated to 88% overall. Volume is
spread across the year, with a modest monsoon dip in on-time performance
(June–September) driven by the synthetic risk model. This gives the delay
classifier a realistic seasonal signal rather than a flat target.
"""
    ),
    code(
        """
pivot = trips.pivot_table(
    index="origin",
    columns="destination",
    values="delivery_status",
    aggfunc=lambda series: (series == "On_Time").mean(),
).reindex(index=config.CITIES, columns=config.CITIES)

z = pivot.to_numpy() * 100
text = np.where(np.isnan(z), "", np.char.add(np.round(np.nan_to_num(z), 1).astype(str), "%"))

fig = go.Figure(
    go.Heatmap(
        z=z,
        x=config.CITIES,
        y=config.CITIES,
        colorscale="RdYlGn",
        zmin=70,
        zmax=100,
        text=text,
        texttemplate="%{text}",
        colorbar=dict(title="On-time %"),
        hovertemplate="%{y} → %{x}<br>On-time: %{z:.1f}%<extra></extra>",
    )
)
fig.update_layout(
    title="On-time rate by origin–destination pair",
    xaxis_title="Destination",
    yaxis_title="Origin",
    height=520,
)
fig.write_image(config.FIGURES / "03_route_heatmap.png", width=850, height=600, scale=2)
fig.show()
"""
    ),
    md("## 4. Fuel efficiency and anomalies"),
    code(
        """
fig = px.scatter(
    trips,
    x="distance_km",
    y="fuel_L",
    color="is_anomaly",
    color_discrete_map={False: "#4C78A8", True: "#C0392B"},
    opacity=0.55,
    hover_data=["trip_id", "vehicle_class", "kmpl"],
    labels={
        "distance_km": "Distance (km)",
        "fuel_L": "Fuel (L)",
        "is_anomaly": "Injected anomaly",
    },
    title="Distance vs fuel consumption (anomalies highlighted)",
)
clean = trips[~trips["is_anomaly"]]
coefficients = np.polyfit(clean["distance_km"], clean["fuel_L"], 1)
x_line = np.linspace(clean["distance_km"].min(), clean["distance_km"].max(), 100)
fig.add_scatter(
    x=x_line,
    y=np.polyval(coefficients, x_line),
    mode="lines",
    name="Clean-data trend",
    line=dict(color="#111111", dash="dash"),
)
fig.write_image(config.FIGURES / "04_distance_vs_fuel.png", width=1000, height=560, scale=2)
fig.show()

print(f"Distance–fuel correlation: {dataset.distance_km.corr(dataset.fuel_L):.3f}")
print(f"Median fuel efficiency:    {(clean.distance_km / clean.fuel_L).median():.2f} km/L")
"""
    ),
    code(
        """
clean = trips[~trips["is_anomaly"]]
fig = px.box(
    clean,
    x="vehicle_class",
    y="kmpl",
    color="vehicle_class",
    points="outliers",
    labels={"kmpl": "km per litre", "vehicle_class": "Vehicle class"},
    title="Fuel efficiency by vehicle class (clean rows)",
)
fig.update_layout(showlegend=False, yaxis_range=[0, 15])
fig.write_image(config.FIGURES / "05_fuel_efficiency.png", width=900, height=520, scale=2)
fig.show()
"""
    ),
    md(
        """
**Observation.** Distance and fuel are strongly correlated (r ≈ 0.84) after
cleaning, and median efficiency is ~4.6 km/L — realistic for a mixed truck
fleet. The highlighted points are deliberately injected fuel spikes, fuel
drops and distance outliers (3% of rows); they are excluded from the efficiency
baseline but retained for anomaly-detection evaluation.
"""
    ),
    md("## 5. Driver behaviour"),
    code(
        """
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Driver score distribution", "On-time rate by score band"),
    specs=[[{"type": "xy"}, {"type": "xy"}]],
)
fig.add_trace(
    go.Histogram(x=trips["driver_score"], nbinsx=25, marker_color="#4C78A8", name="Trips"),
    row=1,
    col=1,
)

trips["score_band"] = pd.cut(
    trips["driver_score"],
    bins=[59, 69, 79, 89, 100],
    labels=["60–69", "70–79", "80–89", "90–100"],
)
band = (
    trips.groupby("score_band", observed=True)
    .agg(on_time=("delivery_status", lambda series: (series == "On_Time").mean()))
    .reset_index()
)
fig.add_trace(
    go.Bar(
        x=band["score_band"].astype(str),
        y=band["on_time"] * 100,
        marker_color="#2E8B57",
        text=[f"{value:.1f}%" for value in band["on_time"] * 100],
        textposition="outside",
        name="On-time %",
    ),
    row=1,
    col=2,
)
fig.update_xaxes(title_text="Driver score", row=1, col=1)
fig.update_yaxes(title_text="Trips", row=1, col=1)
fig.update_xaxes(title_text="Score band", row=1, col=2)
fig.update_yaxes(title_text="On-time %", range=[0, 105], row=1, col=2)
fig.update_layout(showlegend=False, title="Driver score and delivery outcomes", height=480)
fig.write_image(config.FIGURES / "06_driver_score.png", width=1100, height=520, scale=2)
fig.show()
"""
    ),
    code(
        """
correlation = trips[["distance_km", "fuel_L", "driver_score", "kmpl"]].corr()
fig = px.imshow(
    correlation,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation matrix (interpretable numeric features)",
)
fig.write_image(config.FIGURES / "07_correlation.png", width=700, height=600, scale=2)
fig.show()
"""
    ),
    md(
        """
**Observation.** Higher-scoring drivers deliver on time more often, but the
relationship is deliberately not deterministic — route length, season and
operational noise also matter. The score is a behaviour indicator, not a
disciplinary verdict; see `docs/ethics/driver_score_methodology.md`.
"""
    ),
    md("## 6. Injected anomalies"),
    code(
        """
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Anomaly type counts", "Fuel efficiency: clean vs anomalous"),
)
counts = anomalies["anomaly_type"].value_counts()
fig.add_trace(
    go.Bar(
        x=counts.index,
        y=counts.values,
        marker_color=["#C0392B", "#E67E22", "#8E44AD", "#2C3E50"],
        text=counts.values,
        textposition="outside",
        name="Count",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Box(
        x=trips["is_anomaly"].map({False: "Clean", True: "Anomaly"}),
        y=trips["kmpl"],
        marker_color="#4C78A8",
        name="km/L",
    ),
    row=1,
    col=2,
)
fig.update_yaxes(title_text="Rows", row=1, col=1)
fig.update_yaxes(title_text="km per litre", range=[0, 15], row=1, col=2)
fig.update_layout(showlegend=False, title="Injected faults used for anomaly-detection evaluation", height=500)
fig.write_image(config.FIGURES / "08_anomalies.png", width=1100, height=540, scale=2)
fig.show()
"""
    ),
    md(
        """
**Observation.** The anomaly mix covers operational failure modes: fuel spikes
(possible leaks/theft), fuel drops (sensor faults), distance outliers (route
deviations) and duplicate trips (billing errors). Labels are stored separately
from the features, so the anomaly model is evaluated honestly.
"""
    ),
    md("## 7. Route demand"),
    code(
        """
routes = (
    trips.groupby("route")
    .agg(
        trips=("trip_id", "count"),
        on_time=("delivery_status", lambda series: (series == "On_Time").mean()),
        avg_distance=("distance_km", "mean"),
    )
    .sort_values("trips", ascending=False)
    .head(12)
    .reset_index()
)
fig = px.bar(
    routes,
    x="trips",
    y="route",
    orientation="h",
    color="on_time",
    color_continuous_scale="RdYlGn",
    text="trips",
    labels={"trips": "Trips", "route": "Route", "on_time": "On-time rate"},
    title="Top 12 routes by trip volume",
)
fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_colorbar_title="On-time")
fig.write_image(config.FIGURES / "09_route_volume.png", width=950, height=620, scale=2)
fig.show()
"""
    ),
    md(
        """
## 8. Key findings

1. **5,000 trips, 9 canonical columns, 0 missing values.** The cleaned dataset
   meets the brief's schema exactly and covers all 12 months of 2025.
2. **88.0% on-time / 9.0% delayed / 3.0% failed.** The status mix is calibrated
   to the target and is stable across the year.
3. **Fuel efficiency is realistic.** Median efficiency is ~4.6 km/L, and
   distance–fuel correlation is ~0.84; the original file's correlation was
   effectively zero.
4. **Driver score is predictive but not deterministic.** On-time rates rise
   with score bands, leaving room for route and seasonal effects.
5. **3% labelled anomalies** cover fuel spikes/drops, distance outliers and
   duplicate trips, enabling honest anomaly-detection evaluation.
6. **Route demand is balanced across the six-city network**, with long-haul
   corridors showing slightly lower on-time performance.
7. **Modelling implications:** use route, distance, month, vehicle class, load
   and driver score as features; keep anomaly labels out of training; evaluate
   with a time-based split to avoid leakage.
"""
    ),
]


def main() -> None:
    notebook = nbf.v4.new_notebook(cells=CELLS)
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(f"Wrote {NOTEBOOK}")

    command = [
        sys.executable,
        "-m",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(NOTEBOOK),
        "--ExecutePreprocessor.timeout=300",
    ]
    subprocess.run(command, check=True, cwd=ROOT)
    print("Executed notebook successfully")


if __name__ == "__main__":
    main()
