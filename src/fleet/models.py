"""SQLAlchemy models for the fleet platform."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from fleet.db import Base


class Vehicle(Base):
    __tablename__ = "vehicles"

    vehicle_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    vehicle_class: Mapped[str] = mapped_column(String(40), nullable=False)
    capacity_tonnes: Mapped[float] = mapped_column(Float, nullable=False)
    age_years: Mapped[int] = mapped_column(Integer, nullable=False)
    depot_city: Mapped[str] = mapped_column(String(40), nullable=False)

    trips: Mapped[list[Trip]] = relationship(back_populates="vehicle")


class Driver(Base):
    __tablename__ = "drivers"

    driver_id: Mapped[str] = mapped_column(String(10), primary_key=True)
    base_score: Mapped[float] = mapped_column(Float, nullable=False)
    experience_years: Mapped[int] = mapped_column(Integer, nullable=False)
    depot_city: Mapped[str] = mapped_column(String(40), nullable=False)

    trips: Mapped[list[Trip]] = relationship(back_populates="driver")


class Trip(Base):
    __tablename__ = "trips"

    trip_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    vehicle_id: Mapped[str] = mapped_column(ForeignKey("vehicles.vehicle_id"), nullable=False)
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"), nullable=False)
    origin: Mapped[str] = mapped_column(String(40), nullable=False)
    destination: Mapped[str] = mapped_column(String(40), nullable=False)
    distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    fuel_L: Mapped[float] = mapped_column(Float, nullable=False)
    delivery_status: Mapped[str] = mapped_column(String(20), nullable=False)
    driver_score: Mapped[int] = mapped_column(Integer, nullable=False)
    trip_start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    vehicle_class: Mapped[str] = mapped_column(String(40), nullable=False)
    load_tonnes: Mapped[float] = mapped_column(Float, nullable=False)
    data_source: Mapped[str] = mapped_column(String(20), nullable=False, default="synthetic")
    route: Mapped[str] = mapped_column(String(100), nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False)
    day_of_week: Mapped[str] = mapped_column(String(12), nullable=False)

    vehicle: Mapped[Vehicle] = relationship(back_populates="trips")
    driver: Mapped[Driver] = relationship(back_populates="trips")
    anomaly: Mapped[Anomaly | None] = relationship(back_populates="trip", uselist=False)

    __table_args__ = (
        Index("ix_trips_start_time", "trip_start_time"),
        Index("ix_trips_vehicle", "vehicle_id"),
        Index("ix_trips_driver", "driver_id"),
        Index("ix_trips_status", "delivery_status"),
    )

    @property
    def kmpl(self) -> float | None:
        """Fuel efficiency in km per litre."""
        if not self.fuel_L:
            return None
        return round(self.distance_km / self.fuel_L, 2)


class Anomaly(Base):
    __tablename__ = "anomalies"

    trip_id: Mapped[str] = mapped_column(ForeignKey("trips.trip_id"), primary_key=True)
    anomaly_type: Mapped[str] = mapped_column(String(40), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False, default="")

    trip: Mapped[Trip] = relationship(back_populates="anomaly")


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trip_id: Mapped[str | None] = mapped_column(String(12), nullable=True)
    model_name: Mapped[str] = mapped_column(String(60), nullable=False)
    prediction: Mapped[str] = mapped_column(String(60), nullable=False)
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DriverScoreSnapshot(Base):
    __tablename__ = "driver_score_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    driver_id: Mapped[str] = mapped_column(ForeignKey("drivers.driver_id"), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    on_time_component: Mapped[float] = mapped_column(Float, nullable=False)
    fuel_component: Mapped[float] = mapped_column(Float, nullable=False)
    incident_component: Mapped[float] = mapped_column(Float, nullable=False)
    consistency_component: Mapped[float] = mapped_column(Float, nullable=False)
    trips: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RoutePlan(Base):
    __tablename__ = "route_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    origin: Mapped[str] = mapped_column(String(40), nullable=False)
    stops_json: Mapped[str] = mapped_column(Text, nullable=False)
    optimized_order_json: Mapped[str] = mapped_column(Text, nullable=False)
    total_distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    baseline_distance_km: Mapped[float] = mapped_column(Float, nullable=False)
    savings_pct: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
