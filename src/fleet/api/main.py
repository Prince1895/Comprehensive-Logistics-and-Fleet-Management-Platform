"""FastAPI application for the fleet platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from fleet.api.schemas import (
    AnomalyOut,
    DelayRequest,
    DelayResponse,
    DriverOut,
    DriverScoreResponse,
    FuelAnomalyRequest,
    FuelAnomalyResponse,
    FuelEfficiencyOut,
    HealthOut,
    MonthlyOut,
    RouteAnalyticsOut,
    RouteRequest,
    RouteResponse,
    SummaryOut,
    TripOut,
    TripPage,
    VehicleOut,
)
from fleet.db import SessionLocal, get_db, init_db
from fleet.ml.service import (
    driver_score_breakdown,
    optimize_route,
    predict_delay,
    predict_fuel_anomaly,
)
from fleet.models import Anomaly, Driver, Trip, Vehicle
from fleet.seed import seed_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with SessionLocal() as session:
        seed_database(session)
    yield


app = FastAPI(
    title="Comprehensive Logistics & Fleet Management API",
    description=(
        "Fleet analytics, delivery-delay prediction, fuel-anomaly detection, "
        "transparent driver scoring and route optimisation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DbSession = Annotated[Session, Depends(get_db)]


@app.get("/health", response_model=HealthOut, tags=["system"])
def health(db: DbSession) -> HealthOut:
    trips = db.scalar(select(func.count()).select_from(Trip)) or 0
    return HealthOut(status="ok", database="connected", trips=trips)


@app.get("/api/v1/trips", response_model=TripPage, tags=["trips"])
def list_trips(
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    vehicle_id: str | None = None,
    driver_id: str | None = None,
    status: str | None = Query(default=None, alias="delivery_status"),
    origin: str | None = None,
    destination: str | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> TripPage:
    filters = []
    if vehicle_id:
        filters.append(Trip.vehicle_id == vehicle_id)
    if driver_id:
        filters.append(Trip.driver_id == driver_id)
    if status:
        filters.append(Trip.delivery_status == status)
    if origin:
        filters.append(Trip.origin == origin)
    if destination:
        filters.append(Trip.destination == destination)
    if start_date:
        filters.append(Trip.trip_start_time >= start_date)
    if end_date:
        filters.append(Trip.trip_start_time <= end_date)

    total = db.scalar(select(func.count()).select_from(Trip).where(*filters)) or 0
    items = db.scalars(
        select(Trip)
        .where(*filters)
        .order_by(Trip.trip_start_time.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return TripPage(total=total, limit=limit, offset=offset, items=items)


@app.get("/api/v1/trips/{trip_id}", response_model=TripOut, tags=["trips"])
def get_trip(trip_id: str, db: DbSession) -> Trip:
    trip = db.get(Trip, trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")
    return trip


@app.get("/api/v1/vehicles", response_model=list[VehicleOut], tags=["fleet"])
def list_vehicles(db: DbSession) -> list[Vehicle]:
    return list(db.scalars(select(Vehicle).order_by(Vehicle.vehicle_id)).all())


@app.get("/api/v1/drivers", response_model=list[DriverOut], tags=["fleet"])
def list_drivers(db: DbSession) -> list[Driver]:
    return list(db.scalars(select(Driver).order_by(Driver.driver_id)).all())


@app.get("/api/v1/analytics/summary", response_model=SummaryOut, tags=["analytics"])
def analytics_summary(db: DbSession) -> SummaryOut:
    total = db.scalar(select(func.count()).select_from(Trip)) or 0
    if total == 0:
        raise HTTPException(status_code=503, detail="Database is empty")

    status_counts = dict(
        db.execute(select(Trip.delivery_status, func.count()).group_by(Trip.delivery_status)).all()
    )
    avg_distance = db.scalar(select(func.avg(Trip.distance_km))) or 0.0
    avg_fuel = db.scalar(select(func.avg(Trip.fuel_L))) or 0.0
    total_distance = db.scalar(select(func.sum(Trip.distance_km))) or 0.0
    total_fuel = db.scalar(select(func.sum(Trip.fuel_L))) or 0.0
    avg_score = db.scalar(select(func.avg(Trip.driver_score))) or 0.0
    anomaly_count = db.scalar(select(func.count()).select_from(Anomaly)) or 0

    kmpl_values = db.scalars(select(Trip.distance_km / Trip.fuel_L)).all()
    median_kmpl = float(np.median(kmpl_values)) if kmpl_values else 0.0

    return SummaryOut(
        total_trips=total,
        total_vehicles=db.scalar(select(func.count()).select_from(Vehicle)) or 0,
        total_drivers=db.scalar(select(func.count()).select_from(Driver)) or 0,
        on_time_rate=round(status_counts.get("On_Time", 0) / total, 4),
        delayed_rate=round(status_counts.get("Delayed", 0) / total, 4),
        failed_rate=round(status_counts.get("Failed", 0) / total, 4),
        avg_distance_km=round(float(avg_distance), 2),
        avg_fuel_L=round(float(avg_fuel), 2),
        total_distance_km=round(float(total_distance), 2),
        total_fuel_L=round(float(total_fuel), 2),
        median_kmpl=round(median_kmpl, 2),
        anomaly_count=int(anomaly_count),
        anomaly_rate=round(anomaly_count / total, 4),
        avg_driver_score=round(float(avg_score), 2),
    )


@app.get("/api/v1/analytics/monthly", response_model=list[MonthlyOut], tags=["analytics"])
def analytics_monthly(db: DbSession) -> list[MonthlyOut]:
    rows = db.execute(
        select(
            Trip.month,
            func.count(),
            func.sum(case((Trip.delivery_status == "On_Time", 1), else_=0)),
            func.avg(Trip.distance_km),
            func.avg(Trip.distance_km / Trip.fuel_L),
        )
        .group_by(Trip.month)
        .order_by(Trip.month)
    ).all()
    return [
        MonthlyOut(
            month=int(month),
            trips=int(trips),
            on_time_rate=round(float(on_time) / int(trips), 4) if trips else 0.0,
            avg_distance_km=round(float(avg_distance or 0), 2),
            avg_kmpl=round(float(avg_kmpl or 0), 2),
        )
        for month, trips, on_time, avg_distance, avg_kmpl in rows
    ]


@app.get("/api/v1/analytics/routes", response_model=list[RouteAnalyticsOut], tags=["analytics"])
def analytics_routes(
    db: DbSession, limit: int = Query(default=50, ge=1, le=200)
) -> list[RouteAnalyticsOut]:
    rows = db.execute(
        select(
            Trip.origin,
            Trip.destination,
            func.count(),
            func.sum(case((Trip.delivery_status == "On_Time", 1), else_=0)),
            func.avg(Trip.distance_km),
            func.avg(Trip.fuel_L),
        )
        .group_by(Trip.origin, Trip.destination)
        .order_by(func.count().desc())
        .limit(limit)
    ).all()
    return [
        RouteAnalyticsOut(
            origin=origin,
            destination=destination,
            trips=int(trips),
            on_time_rate=round(float(on_time) / int(trips), 4) if trips else 0.0,
            avg_distance_km=round(float(avg_distance or 0), 2),
            avg_fuel_L=round(float(avg_fuel or 0), 2),
        )
        for origin, destination, trips, on_time, avg_distance, avg_fuel in rows
    ]


@app.get(
    "/api/v1/analytics/fuel-efficiency",
    response_model=list[FuelEfficiencyOut],
    tags=["analytics"],
)
def analytics_fuel(db: DbSession) -> list[FuelEfficiencyOut]:
    trips = db.scalars(select(Trip)).all()
    grouped: dict[str, list[float]] = {}
    for trip in trips:
        if trip.kmpl is not None:
            grouped.setdefault(trip.vehicle_class, []).append(trip.kmpl)
    results = []
    for vehicle_class, values in sorted(grouped.items()):
        array = np.array(values, dtype=float)
        results.append(
            FuelEfficiencyOut(
                vehicle_class=vehicle_class,
                trips=len(values),
                avg_kmpl=round(float(array.mean()), 2),
                median_kmpl=round(float(np.median(array)), 2),
                p10_kmpl=round(float(np.percentile(array, 10)), 2),
                p90_kmpl=round(float(np.percentile(array, 90)), 2),
            )
        )
    return results


@app.get("/api/v1/anomalies", response_model=list[AnomalyOut], tags=["analytics"])
def list_anomalies(
    db: DbSession,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    anomaly_type: str | None = None,
) -> list[AnomalyOut]:
    statement = select(Anomaly, Trip).join(Trip, Anomaly.trip_id == Trip.trip_id)
    if anomaly_type:
        statement = statement.where(Anomaly.anomaly_type == anomaly_type)
    rows = db.execute(
        statement.order_by(Anomaly.trip_id).offset(offset).limit(limit)
    ).all()
    return [
        AnomalyOut(
            trip_id=anomaly.trip_id,
            anomaly_type=anomaly.anomaly_type,
            detail=anomaly.detail,
            origin=trip.origin,
            destination=trip.destination,
            distance_km=trip.distance_km,
            fuel_L=trip.fuel_L,
            kmpl=trip.kmpl or 0.0,
            delivery_status=trip.delivery_status,
        )
        for anomaly, trip in rows
    ]


@app.post("/api/v1/predict/delay", response_model=DelayResponse, tags=["ml"])
def predict_delay_endpoint(payload: DelayRequest) -> DelayResponse:
    result = predict_delay(payload.model_dump())
    return DelayResponse(**result)


@app.post("/api/v1/predict/fuel-anomaly", response_model=FuelAnomalyResponse, tags=["ml"])
def predict_fuel_endpoint(payload: FuelAnomalyRequest) -> FuelAnomalyResponse:
    result = predict_fuel_anomaly(payload.model_dump())
    return FuelAnomalyResponse(**result)


@app.get(
    "/api/v1/drivers/{driver_id}/score",
    response_model=DriverScoreResponse,
    tags=["ml"],
)
def driver_score_endpoint(driver_id: str, db: DbSession) -> DriverScoreResponse:
    result = driver_score_breakdown(db, driver_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Driver {driver_id} not found")
    return DriverScoreResponse(**result)


@app.post("/api/v1/routes/optimize", response_model=RouteResponse, tags=["routes"])
def optimize_route_endpoint(payload: RouteRequest) -> RouteResponse:
    try:
        result = optimize_route(payload.origin, payload.stops, payload.return_to_origin)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return RouteResponse(**result)


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "name": "Comprehensive Logistics & Fleet Management API",
        "docs": "/docs",
        "health": "/health",
    }


__all__ = ["app"]
