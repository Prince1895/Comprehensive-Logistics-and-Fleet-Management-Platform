"""Inference services shared by the API, dashboard and tests.

Models are loaded lazily from ``models/``. If a model has not been trained yet,
the service falls back to a transparent heuristic so the platform still runs.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from fleet import config
from fleet.ml.routes import optimize_route

DELAY_MODEL = "delay_model.joblib"
DELAY_FEATURES = "delay_features.json"
FUEL_MODEL = "fuel_anomaly_model.joblib"
FUEL_CONFIG = "fuel_anomaly_config.json"
DRIVER_CONFIG = "driver_score_config.json"


@lru_cache(maxsize=8)
def _load_joblib(name: str) -> Any | None:
    path = config.MODELS / name
    if not path.exists():
        return None
    return joblib.load(path)


@lru_cache(maxsize=8)
def _load_json(name: str) -> dict[str, Any] | None:
    path = config.MODELS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _delay_frame(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "distance_km": float(payload["distance_km"]),
                "driver_score": float(payload["driver_score"]),
                "load_tonnes": float(payload.get("load_tonnes", 5.0)),
                "vehicle_age_years": float(payload.get("vehicle_age_years", 5)),
                "month": int(payload["month"]),
                "is_monsoon": int(int(payload["month"]) in config.MONSOON_MONTHS),
                "origin": payload["origin"],
                "destination": payload["destination"],
                "vehicle_class": payload.get("vehicle_class", "Medium Truck"),
                "day_of_week": payload.get("day_of_week", "Monday"),
            }
        ]
    )


def predict_delay(payload: dict[str, Any]) -> dict[str, Any]:
    """Predict the probability that a trip arrives on time."""
    model = _load_joblib(DELAY_MODEL)
    frame = _delay_frame(payload)

    if model is not None:
        delay_probability = float(model.predict_proba(frame)[0, 1])
        probability = 1 - delay_probability
        model_name = "Calibrated HistGradientBoosting"
    else:
        # Transparent fallback used before training or if the model file is absent.
        risk = (
            0.45 * (100 - float(payload["driver_score"])) / 40
            + 0.25 * float(payload["distance_km"]) / 2200
            + 0.12 * int(int(payload["month"]) in config.MONSOON_MONTHS)
        )
        probability = float(np.clip(1 - risk, 0.01, 0.99))
        model_name = "heuristic_fallback"

    if probability >= 0.75:
        risk_level = "Low"
    elif probability >= 0.5:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "prediction": "On_Time" if probability >= 0.5 else "At_Risk",
        "on_time_probability": round(probability, 4),
        "risk_level": risk_level,
        "model": model_name,
        "features_used": {
            "distance_km": float(payload["distance_km"]),
            "driver_score": float(payload["driver_score"]),
            "month": int(payload["month"]),
            "origin": payload["origin"],
            "destination": payload["destination"],
        },
    }


def _fuel_frame(payload: dict[str, Any]) -> pd.DataFrame:
    distance = float(payload["distance_km"])
    fuel = float(payload["fuel_L"])
    return pd.DataFrame(
        [
            {
                "distance_km": distance,
                "fuel_L": fuel,
                "kmpl": distance / fuel if fuel else 0.0,
                "load_tonnes": float(payload.get("load_tonnes", 5.0)),
                "driver_score": float(payload.get("driver_score", 80)),
            }
        ]
    )


def predict_fuel_anomaly(payload: dict[str, Any]) -> dict[str, Any]:
    """Flag an unusual fuel reading using IsolationForest or a z-score fallback."""
    frame = _fuel_frame(payload)
    config_data = _load_json(FUEL_CONFIG) or {}
    baselines = config_data.get("class_baselines", {})
    vehicle_class = payload.get("vehicle_class", "Medium Truck")
    baseline = baselines.get(vehicle_class, {})
    expected_kmpl = float(baseline.get("median_kmpl", 5.0))
    mad_kmpl = float(baseline.get("mad_kmpl", 0.0))
    actual_kmpl = float(frame.loc[0, "kmpl"])
    deviation_pct = ((actual_kmpl - expected_kmpl) / expected_kmpl * 100) if expected_kmpl else 0.0
    z_score = (0.6745 * (actual_kmpl - expected_kmpl) / mad_kmpl) if mad_kmpl else 0.0
    z_flag = abs(z_score) > 3.5

    model = _load_joblib(FUEL_MODEL)
    if model is not None:
        iso_score = float(model.decision_function(frame)[0])
        iso_flag = bool(model.predict(frame)[0] == -1)
    else:
        iso_score = 0.0
        iso_flag = False

    # The robust z-score is the primary detector: it is explainable and
    # performed best on the held-out evaluation. IsolationForest provides a
    # multivariate second opinion for unusual combinations of features.
    is_anomaly = bool(z_flag or iso_flag)
    if z_flag:
        method = "robust_zscore"
    elif iso_flag:
        method = "isolation_forest"
    else:
        method = "robust_zscore"

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": round(-abs(z_score), 4) if z_flag else round(iso_score, 4),
        "z_score": round(z_score, 4),
        "kmpl": round(actual_kmpl, 2),
        "expected_kmpl": round(expected_kmpl, 2),
        "deviation_pct": round(deviation_pct, 2),
        "method": method,
        "vehicle_class": vehicle_class,
    }


def driver_score_breakdown(session: Session, driver_id: str) -> dict[str, Any] | None:
    """Compute the transparent driver score from that driver's trips."""
    from fleet.models import Anomaly, Trip

    trips = session.scalars(select(Trip).where(Trip.driver_id == driver_id)).all()
    if not trips:
        return None

    statuses = [trip.delivery_status for trip in trips]
    on_time_rate = statuses.count("On_Time") / len(statuses)
    scores = np.array([trip.driver_score for trip in trips], dtype=float)

    # Fuel component: actual km/L relative to the class-level median.
    class_medians: dict[str, float] = {}
    all_trips = session.scalars(select(Trip)).all()
    for vehicle_class in {trip.vehicle_class for trip in all_trips}:
        values = [
            trip.kmpl
            for trip in all_trips
            if trip.vehicle_class == vehicle_class and trip.kmpl is not None
        ]
        if values:
            class_medians[vehicle_class] = float(np.median(values))

    ratios = []
    for trip in trips:
        expected = class_medians.get(trip.vehicle_class)
        if expected and trip.kmpl is not None:
            ratios.append(trip.kmpl / expected)
    fuel_ratio = float(np.median(ratios)) if ratios else 1.0
    fuel_component = float(np.clip((fuel_ratio - 0.7) / 0.4, 0.0, 1.0))

    anomaly_rows = session.scalars(
        select(Anomaly).where(Anomaly.trip_id.in_([trip.trip_id for trip in trips]))
    ).all()
    anomaly_rate = len(anomaly_rows) / len(trips)
    incident_component = float(np.clip(1 - anomaly_rate / 0.10, 0.0, 1.0))

    consistency_component = float(np.clip(1 - scores.std() / 25, 0.0, 1.0))

    weights = {"on_time": 0.40, "fuel": 0.25, "incident": 0.20, "consistency": 0.15}
    raw = (
        weights["on_time"] * on_time_rate
        + weights["fuel"] * fuel_component
        + weights["incident"] * incident_component
        + weights["consistency"] * consistency_component
    )
    score = round(60 + 40 * raw, 1)
    grade = "A" if score >= 85 else "B" if score >= 75 else "C" if score >= 65 else "D"

    return {
        "driver_id": driver_id,
        "score": score,
        "grade": grade,
        "trips": len(trips),
        "components": {
            "on_time": round(on_time_rate, 4),
            "fuel_efficiency": round(fuel_component, 4),
            "incident_free": round(incident_component, 4),
            "consistency": round(consistency_component, 4),
        },
        "weights": weights,
        "metrics": {
            "on_time_rate": round(on_time_rate, 4),
            "median_fuel_ratio": round(fuel_ratio, 4),
            "anomaly_rate": round(anomaly_rate, 4),
            "score_std": round(float(scores.std()), 2),
        },
        "methodology_version": "1.0",
        "note": (
            "Transparent weighted score for coaching and recognition. "
            "Not an automated employment decision; see docs/ethics/."
        ),
    }


__all__ = [
    "driver_score_breakdown",
    "optimize_route",
    "predict_delay",
    "predict_fuel_anomaly",
]
