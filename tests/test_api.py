"""API integration tests against a temporary SQLite database."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

TEST_DIR = Path(tempfile.mkdtemp(prefix="fleet_api_test_"))
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DIR / 'test_fleet.db'}"

from fastapi.testclient import TestClient  # noqa: E402

from fleet.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["trips"] == 5000


def test_summary(client: TestClient) -> None:
    payload = client.get("/api/v1/analytics/summary").json()
    assert payload["total_trips"] == 5000
    assert payload["total_vehicles"] == 100
    assert payload["total_drivers"] == 50
    assert payload["on_time_rate"] == pytest.approx(0.88, abs=0.001)
    assert payload["anomaly_rate"] == pytest.approx(0.03, abs=0.001)
    assert 3.0 <= payload["median_kmpl"] <= 8.0


def test_trip_pagination_and_filter(client: TestClient) -> None:
    page = client.get("/api/v1/trips", params={"limit": 5}).json()
    assert page["total"] == 5000
    assert len(page["items"]) == 5

    filtered = client.get("/api/v1/trips", params={"vehicle_id": "VH001", "limit": 100}).json()
    assert filtered["total"] > 0
    assert all(item["vehicle_id"] == "VH001" for item in filtered["items"])


def test_trip_detail_and_not_found(client: TestClient) -> None:
    trip = client.get("/api/v1/trips/TRP00001")
    assert trip.status_code == 200
    assert trip.json()["trip_id"] == "TRP00001"
    assert client.get("/api/v1/trips/TRP99999").status_code == 404


def test_fleet_endpoints(client: TestClient) -> None:
    assert len(client.get("/api/v1/vehicles").json()) == 100
    assert len(client.get("/api/v1/drivers").json()) == 50


def test_analytics_endpoints(client: TestClient) -> None:
    monthly = client.get("/api/v1/analytics/monthly").json()
    assert len(monthly) == 12
    assert all(0 <= item["on_time_rate"] <= 1 for item in monthly)

    routes = client.get("/api/v1/analytics/routes").json()
    assert len(routes) > 0
    assert all(item["trips"] > 0 for item in routes)

    fuel = client.get("/api/v1/analytics/fuel-efficiency").json()
    assert len(fuel) == 4
    assert all(item["median_kmpl"] > 0 for item in fuel)


def test_anomalies(client: TestClient) -> None:
    anomalies = client.get("/api/v1/anomalies", params={"limit": 500}).json()
    assert len(anomalies) == 150
    assert {item["anomaly_type"] for item in anomalies} == {
        "fuel_spike",
        "fuel_drop",
        "distance_outlier",
        "duplicate_trip",
    }


def test_delay_prediction(client: TestClient) -> None:
    response = client.post(
        "/api/v1/predict/delay",
        json={
            "origin": "Mumbai",
            "destination": "Delhi",
            "distance_km": 1420,
            "driver_score": 85,
            "month": 7,
            "vehicle_class": "Heavy Truck",
            "load_tonnes": 18,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["on_time_probability"] <= 1
    assert payload["risk_level"] in {"Low", "Medium", "High"}
    assert payload["prediction"] in {"On_Time", "At_Risk"}


def test_fuel_anomaly_prediction(client: TestClient) -> None:
    normal = client.post(
        "/api/v1/predict/fuel-anomaly",
        json={
            "distance_km": 1000,
            "fuel_L": 280,
            "vehicle_class": "Heavy Truck",
            "load_tonnes": 18,
            "driver_score": 85,
        },
    ).json()
    assert normal["is_anomaly"] is False

    spike = client.post(
        "/api/v1/predict/fuel-anomaly",
        json={
            "distance_km": 1000,
            "fuel_L": 500,
            "vehicle_class": "Heavy Truck",
            "load_tonnes": 18,
            "driver_score": 85,
        },
    ).json()
    assert spike["is_anomaly"] is True
    assert spike["deviation_pct"] < -30


def test_driver_score(client: TestClient) -> None:
    response = client.get("/api/v1/drivers/DRV001/score")
    assert response.status_code == 200
    payload = response.json()
    assert 60 <= payload["score"] <= 100
    assert payload["grade"] in {"A", "B", "C", "D"}
    assert abs(sum(payload["weights"].values()) - 1.0) < 1e-9
    assert client.get("/api/v1/drivers/DRV999/score").status_code == 404


def test_route_optimisation(client: TestClient) -> None:
    response = client.post(
        "/api/v1/routes/optimize",
        json={
            "origin": "Mumbai",
            "stops": ["Delhi", "Pune", "Chennai"],
            "return_to_origin": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["optimized_distance_km"] <= payload["baseline_distance_km"]
    assert payload["savings_pct"] >= 0
    assert len(payload["legs"]) == 4

    invalid = client.post(
        "/api/v1/routes/optimize",
        json={"origin": "Atlantis", "stops": ["Delhi"]},
    )
    assert invalid.status_code == 422
