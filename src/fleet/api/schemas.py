"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TripOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    trip_id: str
    vehicle_id: str
    driver_id: str
    origin: str
    destination: str
    distance_km: float
    fuel_L: float
    delivery_status: str
    driver_score: int
    trip_start_time: datetime
    vehicle_class: str
    load_tonnes: float
    route: str
    month: int
    day_of_week: str
    kmpl: float | None = None


class TripPage(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[TripOut]


class VehicleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vehicle_id: str
    vehicle_class: str
    capacity_tonnes: float
    age_years: int
    depot_city: str


class DriverOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    driver_id: str
    base_score: float
    experience_years: int
    depot_city: str


class SummaryOut(BaseModel):
    total_trips: int
    total_vehicles: int
    total_drivers: int
    on_time_rate: float
    delayed_rate: float
    failed_rate: float
    avg_distance_km: float
    avg_fuel_L: float
    total_distance_km: float
    total_fuel_L: float
    median_kmpl: float
    anomaly_count: int
    anomaly_rate: float
    avg_driver_score: float


class MonthlyOut(BaseModel):
    month: int
    trips: int
    on_time_rate: float
    avg_distance_km: float
    avg_kmpl: float


class RouteAnalyticsOut(BaseModel):
    origin: str
    destination: str
    trips: int
    on_time_rate: float
    avg_distance_km: float
    avg_fuel_L: float


class FuelEfficiencyOut(BaseModel):
    vehicle_class: str
    trips: int
    avg_kmpl: float
    median_kmpl: float
    p10_kmpl: float
    p90_kmpl: float


class AnomalyOut(BaseModel):
    trip_id: str
    anomaly_type: str
    detail: str
    origin: str
    destination: str
    distance_km: float
    fuel_L: float
    kmpl: float
    delivery_status: str


class DelayRequest(BaseModel):
    origin: str
    destination: str
    distance_km: float = Field(gt=0, le=5000)
    driver_score: float = Field(ge=0, le=100)
    month: int = Field(ge=1, le=12)
    vehicle_class: str = "Medium Truck"
    load_tonnes: float = Field(default=5.0, ge=0, le=40)
    day_of_week: str = "Monday"

    @field_validator("origin", "destination")
    @classmethod
    def city_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("City must not be blank")
        return value.strip()


class DelayResponse(BaseModel):
    prediction: str
    on_time_probability: float
    risk_level: str
    model: str
    features_used: dict


class FuelAnomalyRequest(BaseModel):
    distance_km: float = Field(gt=0, le=5000)
    fuel_L: float = Field(gt=0, le=1000)
    vehicle_class: str = "Medium Truck"
    load_tonnes: float = Field(default=5.0, ge=0, le=40)
    driver_score: float = Field(default=80, ge=0, le=100)


class FuelAnomalyResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    kmpl: float
    expected_kmpl: float
    deviation_pct: float
    method: str
    vehicle_class: str


class DriverScoreResponse(BaseModel):
    driver_id: str
    score: float
    grade: str
    trips: int
    components: dict[str, float]
    weights: dict[str, float]
    metrics: dict[str, float]
    methodology_version: str
    note: str


class RouteRequest(BaseModel):
    origin: str
    stops: list[str] = Field(min_length=1, max_length=12)
    return_to_origin: bool = True


class RouteLeg(BaseModel):
    from_: str = Field(alias="from")
    to: str
    distance_km: float

    model_config = ConfigDict(populate_by_name=True)


class RouteResponse(BaseModel):
    origin: str
    stops: list[str]
    return_to_origin: bool
    baseline_order: list[str]
    optimized_order: list[str]
    baseline_distance_km: float
    optimized_distance_km: float
    savings_km: float
    savings_pct: float
    legs: list[RouteLeg]


class HealthOut(BaseModel):
    status: str
    database: str
    trips: int
