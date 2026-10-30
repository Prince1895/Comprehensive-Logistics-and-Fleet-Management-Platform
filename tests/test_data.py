"""Tests for the reproducible data pipeline and the processed dataset."""

from __future__ import annotations

import pandas as pd
import pytest

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


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DATASET)


@pytest.fixture(scope="module")
def metadata() -> pd.DataFrame:
    return pd.read_csv(config.TRIP_METADATA)


@pytest.fixture(scope="module")
def anomalies() -> pd.DataFrame:
    return pd.read_csv(config.ANOMALY_LABELS)


def test_canonical_schema(dataset: pd.DataFrame) -> None:
    assert list(dataset.columns) == CANONICAL_COLUMNS
    assert len(dataset) == config.TOTAL_TRIPS


def test_no_missing_values(dataset: pd.DataFrame) -> None:
    assert int(dataset.isna().sum().sum()) == 0


def test_unique_trip_ids(dataset: pd.DataFrame) -> None:
    assert dataset["trip_id"].is_unique


def test_status_calibration(dataset: pd.DataFrame) -> None:
    shares = dataset["delivery_status"].value_counts(normalize=True)
    assert shares["On_Time"] == pytest.approx(0.88, abs=0.001)
    assert shares["Delayed"] == pytest.approx(0.09, abs=0.001)
    assert shares["Failed"] == pytest.approx(0.03, abs=0.001)
    assert set(dataset["delivery_status"]) <= {"On_Time", "Delayed", "Failed"}


def test_valid_city_routes(dataset: pd.DataFrame) -> None:
    assert set(dataset["origin"]) <= set(config.CITIES)
    assert set(dataset["destination"]) <= set(config.CITIES)
    assert (dataset["origin"] != dataset["destination"]).all()


def test_distance_and_fuel_ranges(dataset: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    clean = dataset[~dataset["trip_id"].isin(anomalies["trip_id"])]
    assert clean["distance_km"].between(50, 2500).all()
    assert (dataset["fuel_L"] > 0).all()
    kmpl = clean["distance_km"] / clean["fuel_L"]
    assert 3.0 <= kmpl.median() <= 8.0


def test_distance_fuel_correlation(dataset: pd.DataFrame) -> None:
    correlation = dataset["distance_km"].corr(dataset["fuel_L"])
    assert correlation > 0.7


def test_driver_score_range(dataset: pd.DataFrame) -> None:
    assert dataset["driver_score"].between(60, 100).all()
    assert dataset["driver_score"].nunique() > 20


def test_fleet_and_driver_coverage(dataset: pd.DataFrame) -> None:
    assert dataset["vehicle_id"].nunique() == config.VEHICLE_COUNT
    assert dataset["driver_id"].nunique() == config.DRIVER_COUNT


def test_anomaly_share(anomalies: pd.DataFrame, dataset: pd.DataFrame) -> None:
    share = len(anomalies) / len(dataset)
    assert 0.02 <= share <= 0.05
    assert anomalies["trip_id"].is_unique
    assert set(anomalies["trip_id"]) <= set(dataset["trip_id"])


def test_timestamps_are_chronological(metadata: pd.DataFrame) -> None:
    stamps = pd.to_datetime(metadata["trip_start_time"], format="%Y-%m-%d %H:%M:%S")
    assert stamps.is_monotonic_increasing
    assert stamps.dt.year.eq(config.YEAR).all()
    assert metadata["trip_start_time"].str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$").all()


def test_original_rows_preserved(metadata: pd.DataFrame) -> None:
    assert int((metadata["data_source"] == "original").sum()) == config.ORIGINAL_TRIPS


def test_data_dictionary_covers_every_column(dataset: pd.DataFrame) -> None:
    dictionary = pd.read_csv(config.DATA_DICTIONARY)
    described = set(dictionary.loc[dictionary["file"] == "dataset.csv", "column_name"])
    assert set(dataset.columns) <= described


def test_data_dictionary_has_no_placeholders() -> None:
    dictionary = pd.read_csv(config.DATA_DICTIONARY)
    assert not dictionary["description_fill_in"].str.contains("Learner to fill").any()
    assert dictionary["description_fill_in"].str.len().min() > 10
