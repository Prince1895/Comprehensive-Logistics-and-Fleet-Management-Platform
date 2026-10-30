"""Load the processed CSV files into the relational database."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from fleet import config
from fleet.models import Anomaly, Driver, Trip, Vehicle


def _parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def seed_database(session: Session, force: bool = False) -> dict[str, int]:
    """Idempotently load vehicles, drivers, trips and anomaly labels."""
    existing = session.scalar(select(func.count()).select_from(Trip)) or 0
    if existing and not force:
        return {
            "vehicles": session.scalar(select(func.count()).select_from(Vehicle)) or 0,
            "drivers": session.scalar(select(func.count()).select_from(Driver)) or 0,
            "trips": existing,
            "anomalies": session.scalar(select(func.count()).select_from(Anomaly)) or 0,
        }

    if force:
        for model in (Anomaly, Trip, Vehicle, Driver):
            session.query(model).delete()
        session.commit()

    vehicles = pd.read_csv(config.VEHICLES_CSV)
    drivers = pd.read_csv(config.DRIVERS_CSV)
    dataset = pd.read_csv(config.PROCESSED_DATASET)
    metadata = pd.read_csv(config.TRIP_METADATA)
    anomalies = pd.read_csv(config.ANOMALY_LABELS)

    session.bulk_insert_mappings(Vehicle, vehicles.to_dict(orient="records"))
    session.bulk_insert_mappings(Driver, drivers.to_dict(orient="records"))
    session.commit()

    trips = dataset.merge(metadata, on="trip_id", how="left").merge(
        vehicles[["vehicle_id", "age_years"]], on="vehicle_id", how="left"
    )
    trip_rows = []
    for row in trips.to_dict(orient="records"):
        trip_rows.append(
            {
                "trip_id": row["trip_id"],
                "vehicle_id": row["vehicle_id"],
                "driver_id": row["driver_id"],
                "origin": row["origin"],
                "destination": row["destination"],
                "distance_km": float(row["distance_km"]),
                "fuel_L": float(row["fuel_L"]),
                "delivery_status": row["delivery_status"],
                "driver_score": int(row["driver_score"]),
                "trip_start_time": _parse_timestamp(row["trip_start_time"]),
                "vehicle_class": row["vehicle_class"],
                "load_tonnes": float(row["load_tonnes"]),
                "data_source": row["data_source"],
                "route": row["route"],
                "month": int(row["month"]),
                "day_of_week": row["day_of_week"],
            }
        )
    session.bulk_insert_mappings(Trip, trip_rows)
    session.bulk_insert_mappings(
        Anomaly,
        [
            {
                "trip_id": row["trip_id"],
                "anomaly_type": row["anomaly_type"],
                "detail": row["detail"],
            }
            for row in anomalies.to_dict(orient="records")
        ],
    )
    session.commit()

    return {
        "vehicles": len(vehicles),
        "drivers": len(drivers),
        "trips": len(trip_rows),
        "anomalies": len(anomalies),
    }
