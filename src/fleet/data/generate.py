"""Reproducible data pipeline for the fleet platform.

The pipeline deliberately preserves the 500-row file supplied with the
project brief, repairs its known quality problems, and augments it with
4,500 synthetic trips so that the final dataset has 5,000 rows.

Run with::

    python scripts/run_pipeline.py

Outputs
-------
data/processed/dataset.csv          canonical 9-column fleet log
data/processed/trip_metadata.csv    timestamps + operational context
data/processed/anomaly_labels.csv   ground truth for injected faults
data/processed/vehicles.csv         vehicle dimension table
data/processed/drivers.csv          driver dimension table
data/processed/quality_report.md    data-quality summary
docs/data_dictionary.csv            filled data dictionary
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from fleet import config

CANONICAL_COLUMNS = [
    "trip_id",
    "vehicle_id",
    "driver_id",
    "origin",
    "destination",
    "distance_km",
    "fuel_L",
    "delivery_status",
    "driver_score",
]

FEATURE_COLUMNS = [
    "vehicle_id",
    "driver_id",
    "origin",
    "destination",
    "distance_km",
    "fuel_L",
    "delivery_status",
    "driver_score",
    "vehicle_class",
    "load_tonnes",
]


# ---------------------------------------------------------------------------
# Dimension tables
# ---------------------------------------------------------------------------
def build_vehicles(rng: np.random.Generator) -> pd.DataFrame:
    """Create 100 vehicles spread across four realistic vehicle classes."""
    rows: list[dict[str, Any]] = []
    vehicle_number = 1
    for vehicle_class, meta in config.VEHICLE_CLASSES.items():
        for _ in range(int(meta["count"])):
            rows.append(
                {
                    "vehicle_id": f"VH{vehicle_number:03d}",
                    "vehicle_class": vehicle_class,
                    "capacity_tonnes": meta["capacity_tonnes"],
                    "age_years": int(rng.integers(0, 13)),
                    "depot_city": str(rng.choice(config.CITIES)),
                }
            )
            vehicle_number += 1
    return pd.DataFrame(rows)


def build_drivers() -> pd.DataFrame:
    """Create 50 drivers with a stable, explainable base behaviour score."""
    rows: list[dict[str, Any]] = []
    for number in range(1, config.DRIVER_COUNT + 1):
        experience = 1 + ((number * 7) % 15)
        base_score = 62 + ((number * 17) % 37) + min(experience * 0.25, 3)
        rows.append(
            {
                "driver_id": f"DRV{number:03d}",
                "base_score": round(float(np.clip(base_score, 60, 99)), 1),
                "experience_years": experience,
                "depot_city": config.CITIES[number % len(config.CITIES)],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def route_distance(rng: np.random.Generator, origin: str, destination: str) -> float:
    """Return a road distance with a small, reproducible route-level jitter."""
    key = tuple(sorted((origin, destination)))
    base = config.CITY_DISTANCES.get(key) or config.CITY_DISTANCES[tuple(reversed(key))]
    return round(float(base * (1 + rng.uniform(-0.08, 0.08))), 1)


def pick_destination(rng: np.random.Generator, origin: str) -> str:
    choices = [city for city in config.CITIES if city != origin]
    return str(rng.choice(choices))


def assign_timestamps(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Assign unique, chronological timestamps across 2025."""
    n = len(df)
    start = datetime(config.YEAR, 1, 1, 5, 0, 0)
    end = datetime(config.YEAR, 12, 31, 23, 0, 0)
    total_seconds = int((end - start).total_seconds())

    seconds = np.sort(rng.integers(0, total_seconds, size=n)) + np.arange(n)
    seconds = np.clip(seconds, 0, total_seconds)
    stamps = [start + timedelta(seconds=int(value)) for value in seconds]
    df = df.copy()
    df["trip_start_time"] = [stamp.strftime("%Y-%m-%d %H:%M:%S") for stamp in stamps]
    return df


def assign_fuel(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Model fuel from distance, vehicle class, load, age and season."""
    df = df.copy()
    class_efficiency = {name: meta["kmpl"] for name, meta in config.VEHICLE_CLASSES.items()}
    base_efficiency = df["vehicle_class"].map(class_efficiency).to_numpy(dtype=float)

    load_fraction = rng.uniform(0.2, 1.0, size=len(df))
    month = pd.to_datetime(df["trip_start_time"]).dt.month.to_numpy()
    monsoon = np.isin(month, list(config.MONSOON_MONTHS))
    noise = rng.normal(1.0, 0.05, size=len(df))

    kmpl = (
        base_efficiency
        * (1 - 0.12 * load_fraction)
        * (1 - 0.01 * df["age_years"].to_numpy(dtype=float))
        * np.where(monsoon, 0.95, 1.0)
        * noise
    )
    df["load_tonnes"] = (load_fraction * df["capacity_tonnes"].to_numpy(dtype=float)).round(1)
    df["fuel_L"] = np.clip(df["distance_km"].to_numpy(dtype=float) / kmpl, 1.0, None).round(1)
    return df


def inject_measurement_anomalies(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Inject fuel and distance faults into ~2.6% of rows."""
    df = df.copy()
    df["is_anomaly"] = False
    df["anomaly_type"] = ""
    df["anomaly_detail"] = ""

    n = len(df)
    n_anomalies = round(n * config.ANOMALY_SHARE)
    n_duplicates = round(n_anomalies * 0.133)
    n_measurement = n_anomalies - n_duplicates

    n_spike = round(n_measurement * 0.46)
    n_drop = round(n_measurement * 0.31)
    n_distance = n_measurement - n_spike - n_drop
    fault_types = (
        ["fuel_spike"] * n_spike + ["fuel_drop"] * n_drop + ["distance_outlier"] * n_distance
    )
    rng.shuffle(fault_types)

    indices = rng.choice(n, size=n_measurement, replace=False)
    labels: list[dict[str, str]] = []
    for index, fault in zip(indices, fault_types):
        index = int(index)
        if fault == "fuel_spike":
            factor = float(rng.uniform(1.8, 2.5))
            df.loc[index, "fuel_L"] = round(float(df.loc[index, "fuel_L"]) * factor, 1)
            detail = f"fuel_L multiplied by {factor:.2f}"
        elif fault == "fuel_drop":
            factor = float(rng.uniform(0.30, 0.55))
            df.loc[index, "fuel_L"] = round(float(df.loc[index, "fuel_L"]) * factor, 1)
            detail = f"fuel_L multiplied by {factor:.2f}"
        else:
            factor = float(rng.uniform(2.5, 4.0))
            df.loc[index, "distance_km"] = round(float(df.loc[index, "distance_km"]) * factor, 1)
            df.loc[index, "fuel_L"] = round(float(df.loc[index, "fuel_L"]) * factor, 1)
            detail = f"distance_km multiplied by {factor:.2f}"

        df.loc[index, "is_anomaly"] = True
        df.loc[index, "anomaly_type"] = fault
        df.loc[index, "anomaly_detail"] = detail
        labels.append(
            {"trip_id": str(df.loc[index, "trip_id"]), "anomaly_type": fault, "detail": detail}
        )
    return df, labels


def assign_delivery_status(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Assign status with a risk model, calibrated to 88/9/3 exactly."""
    df = df.copy()
    month = pd.to_datetime(df["trip_start_time"]).dt.month.to_numpy()
    monsoon = np.isin(month, list(config.MONSOON_MONTHS)).astype(float)

    risk = (
        0.45 * (100 - df["driver_score"].to_numpy(dtype=float)) / 40
        + 0.25 * (df["distance_km"].to_numpy(dtype=float) / 2200)
        + 0.12 * monsoon
        + 0.10 * df["is_anomaly"].to_numpy(dtype=float)
        + 0.08 * rng.normal(0, 1, size=len(df))
    )

    n = len(df)
    n_on_time = round(n * config.ON_TIME_SHARE)
    n_delayed = round(n * config.DELAYED_SHARE)
    order = np.argsort(risk)

    status = np.empty(n, dtype=object)
    status[order[:n_on_time]] = "On_Time"
    status[order[n_on_time : n_on_time + n_delayed]] = "Delayed"
    status[order[n_on_time + n_delayed :]] = "Failed"
    df["delivery_status"] = status
    return df


def inject_duplicate_trips(
    df: pd.DataFrame, rng: np.random.Generator, labels: list[dict[str, str]]
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Create exact feature duplicates while keeping trip_id/timestamp unique."""
    df = df.copy()
    n_duplicates = round(len(df) * config.ANOMALY_SHARE * 0.133)
    candidates = np.flatnonzero(~df["is_anomaly"].to_numpy())
    if len(candidates) < 2 * n_duplicates:
        return df, labels

    permutation = rng.permutation(candidates)
    targets = permutation[:n_duplicates]
    sources = permutation[n_duplicates : 2 * n_duplicates]

    for target, source in zip(targets, sources):
        target, source = int(target), int(source)
        for column in FEATURE_COLUMNS:
            df.loc[target, column] = df.loc[source, column]
        detail = f"duplicate of {df.loc[source, 'trip_id']}"
        df.loc[target, "is_anomaly"] = True
        df.loc[target, "anomaly_type"] = "duplicate_trip"
        df.loc[target, "anomaly_detail"] = detail
        labels.append(
            {
                "trip_id": str(df.loc[target, "trip_id"]),
                "anomaly_type": "duplicate_trip",
                "detail": detail,
            }
        )
    return df, labels


def rebalance_status(df: pd.DataFrame) -> pd.DataFrame:
    """Restore the exact 88/9/3 target after duplicate rows are created."""
    df = df.copy()
    n = len(df)
    target = {
        "On_Time": round(n * config.ON_TIME_SHARE),
        "Delayed": round(n * config.DELAYED_SHARE),
    }
    target["Failed"] = n - target["On_Time"] - target["Delayed"]

    for _ in range(100):
        counts = df["delivery_status"].value_counts().to_dict()
        over = [status for status in target if counts.get(status, 0) > target[status]]
        under = [status for status in target if counts.get(status, 0) < target[status]]
        if not over or not under:
            break
        source, destination = over[0], under[0]
        candidates = df.index[
            (df["delivery_status"] == source) & (~df["is_anomaly"].astype(bool))
        ]
        if len(candidates) == 0:
            break
        df.loc[candidates[0], "delivery_status"] = destination
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_dataset() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    rng = np.random.default_rng(config.SEED)
    vehicles = build_vehicles(rng)
    drivers = build_drivers()

    # --- 1. Clean the supplied 500-row file -------------------------------
    original = pd.read_csv(config.RAW_DATASET)
    original = original.rename(columns={"score": "driver_score"})
    original = original[CANONICAL_COLUMNS].copy()
    original["data_source"] = "original"

    same_city = original["origin"] == original["destination"]
    original.loc[same_city, "destination"] = [
        pick_destination(rng, str(origin)) for origin in original.loc[same_city, "origin"]
    ]
    original["distance_km"] = [
        route_distance(rng, str(origin), str(destination))
        for origin, destination in zip(original["origin"], original["destination"])
    ]
    original["driver_score"] = (
        original["driver_score"].clip(60, 100).round().astype(int)
    )

    # --- 2. Generate 4,500 additional trips ------------------------------
    driver_base = dict(zip(drivers["driver_id"], drivers["base_score"]))
    vehicle_ids = vehicles["vehicle_id"].tolist()
    driver_ids = drivers["driver_id"].tolist()
    new_rows: list[dict[str, Any]] = []
    for offset in range(config.TOTAL_TRIPS - len(original)):
        vehicle_id = str(rng.choice(vehicle_ids))
        driver_id = str(rng.choice(driver_ids))
        origin = str(rng.choice(config.CITIES))
        destination = pick_destination(rng, origin)
        score = int(np.clip(round(driver_base[driver_id] + rng.normal(0, 4)), 60, 100))
        new_rows.append(
            {
                "trip_id": f"TRP{len(original) + offset + 1:05d}",
                "vehicle_id": vehicle_id,
                "driver_id": driver_id,
                "origin": origin,
                "destination": destination,
                "distance_km": route_distance(rng, origin, destination),
                "fuel_L": 0.0,
                "delivery_status": "",
                "driver_score": score,
                "data_source": "synthetic",
            }
        )

    trips = pd.concat([original, pd.DataFrame(new_rows)], ignore_index=True)
    trips = trips.sort_values("trip_id").reset_index(drop=True)

    # --- 3. Enrich with dimensions, timestamps, fuel ---------------------
    trips = trips.merge(
        vehicles[["vehicle_id", "vehicle_class", "capacity_tonnes", "age_years"]],
        on="vehicle_id",
        how="left",
        validate="many_to_one",
    )
    trips = assign_timestamps(trips, rng)
    trips = assign_fuel(trips, rng)

    # --- 4. Faults and status --------------------------------------------
    trips, labels = inject_measurement_anomalies(trips, rng)
    trips = assign_delivery_status(trips, rng)
    trips, labels = inject_duplicate_trips(trips, rng, labels)
    trips = rebalance_status(trips)

    # Keep the canonical column order for dataset.csv.
    dataset = trips[CANONICAL_COLUMNS].copy()
    dataset["trip_id"] = dataset["trip_id"].astype(str)
    dataset["vehicle_id"] = dataset["vehicle_id"].astype(str)
    dataset["driver_id"] = dataset["driver_id"].astype(str)

    metadata = trips[
        [
            "trip_id",
            "trip_start_time",
            "vehicle_class",
            "load_tonnes",
            "data_source",
        ]
    ].copy()
    parsed = pd.to_datetime(metadata["trip_start_time"])
    metadata["month"] = parsed.dt.month
    metadata["day_of_week"] = parsed.dt.day_name()
    metadata["route"] = trips["origin"] + " → " + trips["destination"]

    anomaly_labels = pd.DataFrame(labels, columns=["trip_id", "anomaly_type", "detail"])
    anomaly_labels = anomaly_labels.sort_values("trip_id").reset_index(drop=True)

    return dataset, metadata, vehicles, drivers, anomaly_labels


def _quality_report(
    dataset: pd.DataFrame,
    metadata: pd.DataFrame,
    anomaly_labels: pd.DataFrame,
) -> str:
    kmpl = dataset["distance_km"] / dataset["fuel_L"]
    status_counts = dataset["delivery_status"].value_counts()
    status_share = dataset["delivery_status"].value_counts(normalize=True).mul(100).round(2)
    duplicate_features = dataset.duplicated(
        subset=[column for column in FEATURE_COLUMNS if column in dataset.columns]
    ).sum()
    monthly = metadata.groupby("month").size()

    lines = [
        "# Data Quality Report",
        "",
        "Generated by `scripts/run_pipeline.py` with `seed=42`.",
        "",
        "## Shape and integrity",
        "",
        f"- Rows: **{len(dataset):,}**",
        f"- Columns: **{len(dataset.columns)}** (canonical schema)",
        f"- Null values: **{int(dataset.isna().sum().sum())}**",
        f"- Duplicate trip IDs: **{int(dataset['trip_id'].duplicated().sum())}**",
        f"- Exact duplicate feature rows: **{int(duplicate_features)}**",
        "",
        "## Delivery performance",
        "",
        "| Status | Count | Share |",
        "|---|---:|---:|",
    ]
    for status in ["On_Time", "Delayed", "Failed"]:
        lines.append(
            f"| {status} | {int(status_counts.get(status, 0)):,} | {status_share.get(status, 0):.2f}% |"
        )

    lines += [
        "",
        "## Fuel efficiency",
        "",
        f"- km/L min: **{kmpl.min():.2f}**",
        f"- km/L median: **{kmpl.median():.2f}**",
        f"- km/L max: **{kmpl.max():.2f}**",
        f"- Correlation between distance and fuel: **{dataset['distance_km'].corr(dataset['fuel_L']):.3f}**",
        "",
        "## Injected anomalies",
        "",
        (
            f"- Labelled anomalies: **{len(anomaly_labels):,}** "
            f"({len(anomaly_labels) / len(dataset) * 100:.2f}% of rows)"
        ),
        "",
        "| Anomaly type | Count |",
        "|---|---:|",
    ]
    for anomaly_type, count in anomaly_labels["anomaly_type"].value_counts().items():
        lines.append(f"| {anomaly_type} | {int(count)} |")

    lines += [
        "",
        "## Monthly trip volume",
        "",
        "| Month | Trips |",
        "|---|---:|",
    ]
    for month, count in monthly.items():
        lines.append(f"| {int(month):02d} | {int(count)} |")

    lines += [
        "",
        "## Cleaning decisions",
        "",
        "1. The original 500 rows are preserved unchanged in `data/raw/dataset_original.csv`.",
        "2. 73 same-city trips in the original file were given a valid destination.",
        "3. Route distances were normalised against a realistic six-city road-distance matrix.",
        "4. Fuel was recomputed from vehicle class, load, vehicle age and monsoon season.",
        "5. Delivery status was recalibrated to the brief's 88% on-time target.",
        "6. 4,500 synthetic trips were appended, keeping the original 9-column schema.",
        "7. Anomalies are labelled in a separate file so models cannot leak the answer.",
        "",
        "## Limitations",
        "",
        "- The dataset is synthetic/derived for academic use and is not real operational data.",
        "- Driver scores are a transparent demonstration model, not an employment decision tool.",
        "- Fuel efficiency values are modelled estimates, not telematics measurements.",
        "",
    ]
    return "\n".join(lines)


def _data_dictionary() -> pd.DataFrame:
    rows = [
        # dataset.csv
        ("trip_id", "string", "TRP00001", "TRP02500", "TRP05000", "No",
         "Unique trip identifier. Sequential and immutable.", "dataset.csv"),
        ("vehicle_id", "string", "VH001", "VH050", "VH100", "No",
         "Fleet vehicle identifier; joins to vehicles.csv.", "dataset.csv"),
        ("driver_id", "string", "DRV001", "DRV025", "DRV050", "No",
         "Anonymised driver identifier; joins to drivers.csv.", "dataset.csv"),
        ("origin", "string", "Mumbai", "Delhi", "Chennai", "No",
         "Origin city. One of Mumbai, Delhi, Bengaluru, Pune, Chennai, Hyderabad.", "dataset.csv"),
        ("destination", "string", "Pune", "Hyderabad", "Bengaluru", "No",
         "Destination city; always different from origin.", "dataset.csv"),
        ("distance_km", "float", "148.3", "776.1", "1429.2", "No",
         "One-way route distance in kilometres.", "dataset.csv"),
        ("fuel_L", "float", "38.4", "77.4", "121.4", "No",
         "Fuel consumed in litres; modelled from distance, vehicle class, load and season.",
         "dataset.csv"),
        ("delivery_status", "string", "On_Time", "Delayed", "Failed", "No",
         "Delivery outcome. Calibrated to 88% On_Time, 9% Delayed, 3% Failed.",
         "dataset.csv"),
        ("driver_score", "integer", "94", "78", "61", "No",
         "Transparent driver behaviour score from 60 to 100; see docs/ethics.",
         "dataset.csv"),
        # trip_metadata.csv
        ("trip_start_time", "datetime", "2025-01-01 05:12:00", "2025-07-14 18:40:00",
         "2025-12-31 22:05:00", "No",
         "Trip start timestamp in YYYY-MM-DD HH:MM:SS format; chronological across 2025.",
         "trip_metadata.csv"),
        ("vehicle_class", "string", "Heavy Truck", "Medium Truck", "Light Van", "No",
         "Vehicle category used for fuel-efficiency baselines.", "trip_metadata.csv"),
        ("load_tonnes", "float", "2.4", "9.8", "21.5", "No",
         "Modelled payload in tonnes.", "trip_metadata.csv"),
        ("month", "integer", "1", "7", "12", "No",
         "Calendar month extracted from trip_start_time.", "trip_metadata.csv"),
        ("day_of_week", "string", "Monday", "Friday", "Sunday", "No",
         "Day name extracted from trip_start_time.", "trip_metadata.csv"),
        ("route", "string", "Mumbai → Pune", "Delhi → Chennai", "Pune → Hyderabad", "No",
         "Human-readable origin → destination pair.", "trip_metadata.csv"),
        ("data_source", "string", "original", "synthetic", "synthetic", "No",
         "Provenance flag: original 500 rows or generated rows.", "trip_metadata.csv"),
        # vehicles.csv
        ("vehicle_id", "string", "VH001", "VH050", "VH100", "No",
         "Primary key for the vehicle dimension.", "vehicles.csv"),
        ("vehicle_class", "string", "Heavy Truck", "Light Van", "Refrigerated Truck", "No",
         "Vehicle category.", "vehicles.csv"),
        ("capacity_tonnes", "float", "25.0", "12.0", "3.5", "No",
         "Maximum payload capacity in tonnes.", "vehicles.csv"),
        ("age_years", "integer", "0", "6", "12", "No",
         "Vehicle age in years; affects fuel efficiency.", "vehicles.csv"),
        ("depot_city", "string", "Mumbai", "Delhi", "Chennai", "No",
         "Home depot city.", "vehicles.csv"),
        # drivers.csv
        ("driver_id", "string", "DRV001", "DRV025", "DRV050", "No",
         "Primary key for the driver dimension.", "drivers.csv"),
        ("base_score", "float", "64.0", "80.5", "97.0", "No",
         "Long-run behaviour baseline used to generate trip scores.", "drivers.csv"),
        ("experience_years", "integer", "1", "8", "15", "No",
         "Driving experience in years.", "drivers.csv"),
        ("depot_city", "string", "Mumbai", "Delhi", "Chennai", "No",
         "Driver's home depot city.", "drivers.csv"),
        # anomaly_labels.csv
        ("trip_id", "string", "TRP00042", "TRP02310", "TRP04877", "No",
         "Trip with an injected fault; joins to dataset.csv.", "anomaly_labels.csv"),
        ("anomaly_type", "string", "fuel_spike", "distance_outlier", "duplicate_trip", "No",
         "Ground-truth fault category.", "anomaly_labels.csv"),
        ("detail", "string", "fuel_L multiplied by 2.10", "duplicate of TRP01234",
         "distance_km multiplied by 3.20", "No",
         "Human-readable explanation of the injected fault.", "anomaly_labels.csv"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "column_name",
            "data_type",
            "sample_value_1",
            "sample_value_2",
            "sample_value_3",
            "nullable",
            "description_fill_in",
            "file",
        ],
    )


def run() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.DOCS.mkdir(parents=True, exist_ok=True)

    dataset, metadata, vehicles, drivers, anomaly_labels = build_dataset()

    dataset.to_csv(config.PROCESSED_DATASET, index=False)
    metadata.to_csv(config.TRIP_METADATA, index=False)
    anomaly_labels.to_csv(config.ANOMALY_LABELS, index=False)
    vehicles.to_csv(config.VEHICLES_CSV, index=False)
    drivers.to_csv(config.DRIVERS_CSV, index=False)
    _data_dictionary().to_csv(config.DATA_DICTIONARY, index=False)

    report = _quality_report(dataset, metadata, anomaly_labels)
    config.QUALITY_REPORT.write_text(report, encoding="utf-8")

    print(f"Wrote {len(dataset):,} trips to {config.PROCESSED_DATASET}")
    print(f"Wrote {len(anomaly_labels):,} anomaly labels to {config.ANOMALY_LABELS}")
    print(f"Wrote quality report to {config.QUALITY_REPORT}")
    print(f"Wrote data dictionary to {config.DATA_DICTIONARY}")


if __name__ == "__main__":
    run()
