"""Central configuration for the fleet platform.

Keeping the constants in one module makes the synthetic-data pipeline,
the database seed script, the ML training code and the dashboard agree
on the same city network, vehicle classes and file locations.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"
MODELS = ROOT / "models"

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
SEED = 42
TOTAL_TRIPS = 5_000
ORIGINAL_TRIPS = 500
ON_TIME_SHARE = 0.88
DELAYED_SHARE = 0.09
FAILED_SHARE = 0.03
ANOMALY_SHARE = 0.03
YEAR = 2025

# Six-city network used by the provided dataset.
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Chennai", "Hyderabad"]

# Approximate city coordinates for dashboard maps.
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bengaluru": (12.9716, 77.5946),
    "Pune": (18.5204, 73.8567),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
}

# Approximate one-way road distances in km. Symmetric by construction.
CITY_DISTANCES: dict[tuple[str, str], float] = {
    ("Mumbai", "Delhi"): 1420,
    ("Mumbai", "Bengaluru"): 980,
    ("Mumbai", "Pune"): 150,
    ("Mumbai", "Chennai"): 1330,
    ("Mumbai", "Hyderabad"): 710,
    ("Delhi", "Bengaluru"): 2150,
    ("Delhi", "Pune"): 1450,
    ("Delhi", "Chennai"): 2200,
    ("Delhi", "Hyderabad"): 1580,
    ("Pune", "Bengaluru"): 840,
    ("Pune", "Chennai"): 1180,
    ("Pune", "Hyderabad"): 560,
    ("Bengaluru", "Chennai"): 350,
    ("Bengaluru", "Hyderabad"): 570,
    ("Chennai", "Hyderabad"): 630,
}

# Vehicle fleet mix. kmpl = typical loaded fuel efficiency.
VEHICLE_CLASSES: dict[str, dict[str, float]] = {
    "Heavy Truck": {"kmpl": 4.2, "capacity_tonnes": 25.0, "count": 40},
    "Medium Truck": {"kmpl": 6.5, "capacity_tonnes": 12.0, "count": 35},
    "Light Van": {"kmpl": 10.5, "capacity_tonnes": 3.5, "count": 15},
    "Refrigerated Truck": {"kmpl": 4.8, "capacity_tonnes": 18.0, "count": 10},
}

DRIVER_COUNT = 50
VEHICLE_COUNT = 100

# Monsoon months reduce fuel efficiency and on-time performance.
MONSOON_MONTHS = {6, 7, 8, 9}

# File names
RAW_DATASET = DATA_RAW / "dataset_original.csv"
PROCESSED_DATASET = DATA_PROCESSED / "dataset.csv"
TRIP_METADATA = DATA_PROCESSED / "trip_metadata.csv"
ANOMALY_LABELS = DATA_PROCESSED / "anomaly_labels.csv"
VEHICLES_CSV = DATA_PROCESSED / "vehicles.csv"
DRIVERS_CSV = DATA_PROCESSED / "drivers.csv"
QUALITY_REPORT = DATA_PROCESSED / "quality_report.md"
DATA_DICTIONARY = DOCS / "data_dictionary.csv"

# Database
DATABASE_URL = "sqlite:///fleet.db"
