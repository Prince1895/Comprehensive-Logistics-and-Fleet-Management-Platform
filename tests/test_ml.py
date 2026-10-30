"""Tests for trained models, metrics and the route optimiser."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fleet import config
from fleet.ml.routes import optimize_route


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads((config.REPORTS / "metrics.json").read_text(encoding="utf-8"))


def test_delay_model_quality(metrics: dict) -> None:
    delay = metrics["delay"]
    best = delay["models"][delay["best_model"]]
    assert best["roc_auc"] > 0.85
    assert best["f1"] > 0.5
    assert best["recall"] > 0.5
    assert delay["train_rows"] == 4000
    assert delay["test_rows"] == 1000


def test_delay_model_beats_baseline(metrics: dict) -> None:
    models = metrics["delay"]["models"]
    assert models[metrics["delay"]["best_model"]]["roc_auc"] > models["dummy_prior"]["roc_auc"]


def test_fuel_anomaly_quality(metrics: dict) -> None:
    fuel = metrics["fuel_anomaly"]
    assert fuel["primary_method"] == "robust_zscore"
    assert fuel["zscore_baseline"]["f1"] > 0.6
    assert fuel["isolation_forest"]["roc_auc"] > 0.7


def test_driver_scores_are_bounded(metrics: dict) -> None:
    driver = metrics["driver_score"]
    assert driver["drivers"] == 50
    assert 60 <= driver["min"] <= driver["max"] <= 100
    assert abs(sum(driver["weights"].values()) - 1.0) < 1e-9
    assert driver["fairness"]["max_depot_mean_gap"] < 10
    assert driver["fairness"]["flag"] is False


def test_driver_score_file(metrics: dict) -> None:
    scores = pd.read_csv(config.REPORTS / "driver_scores.csv")
    assert len(scores) == 50
    assert scores["driver_id"].is_unique
    assert scores["score"].between(60, 100).all()


def test_route_optimiser_finds_savings() -> None:
    result = optimize_route("Mumbai", ["Delhi", "Pune", "Chennai"], return_to_origin=True)
    assert result["optimized_distance_km"] <= result["baseline_distance_km"]
    assert result["savings_pct"] > 0
    assert result["optimized_order"] != result["baseline_order"]


def test_route_optimiser_is_deterministic() -> None:
    first = optimize_route("Delhi", ["Mumbai", "Bengaluru", "Chennai", "Hyderabad"])
    second = optimize_route("Delhi", ["Mumbai", "Bengaluru", "Chennai", "Hyderabad"])
    assert first == second


def test_route_optimiser_rejects_unknown_city() -> None:
    with pytest.raises(ValueError):
        optimize_route("Mumbai", ["Atlantis"])
