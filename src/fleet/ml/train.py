"""Train and evaluate the platform's ML services.

Outputs
-------
models/delay_model.joblib          best delay classifier (predicts is_delayed)
models/delay_logreg.joblib         interpretable logistic baseline
models/delay_features.json         feature contract for inference
models/fuel_anomaly_model.joblib   IsolationForest
models/fuel_anomaly_config.json    feature list + per-class baselines
models/driver_score_config.json    published score weights
reports/metrics.json               all evaluation metrics
reports/driver_scores.csv          per-driver score breakdown
reports/model_cards.md             model documentation
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fleet import config
from fleet.ml.routes import optimize_route

NUMERIC_FEATURES = [
    "distance_km",
    "driver_score",
    "load_tonnes",
    "vehicle_age_years",
    "month",
    "is_monsoon",
]
CATEGORICAL_FEATURES = ["origin", "destination", "vehicle_class", "day_of_week"]
FUEL_FEATURES = ["distance_km", "fuel_L", "kmpl", "load_tonnes", "driver_score"]
DRIVER_WEIGHTS = {"on_time": 0.40, "fuel": 0.25, "incident": 0.20, "consistency": 0.15}
DRIVER_METHOD_VERSION = "1.0"


def load_frame() -> pd.DataFrame:
    dataset = pd.read_csv(config.PROCESSED_DATASET)
    metadata = pd.read_csv(config.TRIP_METADATA)
    vehicles = pd.read_csv(config.VEHICLES_CSV)
    anomalies = pd.read_csv(config.ANOMALY_LABELS)

    frame = (
        dataset.merge(metadata, on="trip_id", how="left")
        .merge(vehicles[["vehicle_id", "age_years"]], on="vehicle_id", how="left")
        .merge(anomalies[["trip_id", "anomaly_type"]], on="trip_id", how="left")
    )
    frame["is_anomaly"] = frame["anomaly_type"].notna()
    frame["kmpl"] = frame["distance_km"] / frame["fuel_L"]
    frame["trip_start_time"] = pd.to_datetime(frame["trip_start_time"])
    frame["is_delayed"] = (frame["delivery_status"] != "On_Time").astype(int)
    frame["is_monsoon"] = frame["month"].isin(config.MONSOON_MONTHS).astype(int)
    frame["vehicle_age_years"] = frame["age_years"].astype(float)
    frame["route"] = frame["origin"] + " → " + frame["destination"]
    return frame.sort_values("trip_start_time").reset_index(drop=True)


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "positive_rate": round(float(predictions.mean()), 4),
    }


def train_delay(frame: pd.DataFrame) -> dict:
    split = int(len(frame) * 0.8)
    train, test = frame.iloc[:split], frame.iloc[split:]
    X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["is_delayed"]
    X_test, y_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test["is_delayed"]

    linear_preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    tree_preprocess = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    logistic = Pipeline(
        [
            ("preprocess", linear_preprocess),
            (
                "model",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.SEED),
            ),
        ]
    )
    gradient_boosting = Pipeline(
        [
            ("preprocess", tree_preprocess),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.07,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    random_state=config.SEED,
                ),
            ),
        ]
    )
    calibrated = CalibratedClassifierCV(gradient_boosting, method="sigmoid", cv=3)
    dummy = DummyClassifier(strategy="prior")

    results = {}
    for name, model in [
        ("logistic_regression", logistic),
        ("hist_gradient_boosting", gradient_boosting),
        ("calibrated_hist_gradient_boosting", calibrated),
        ("dummy_prior", dummy),
    ]:
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        results[name] = _metrics(y_test.to_numpy(), probabilities)

    best_name = max(
        [
            "logistic_regression",
            "hist_gradient_boosting",
            "calibrated_hist_gradient_boosting",
        ],
        key=lambda name: (results[name]["f1"], results[name]["roc_auc"]),
    )
    best_model = {
        "logistic_regression": logistic,
        "hist_gradient_boosting": gradient_boosting,
        "calibrated_hist_gradient_boosting": calibrated,
    }[best_name]

    # Interpretable coefficients from the logistic baseline.
    feature_names = logistic.named_steps["preprocess"].get_feature_names_out()
    coefficients = logistic.named_steps["model"].coef_[0]
    top_linear = sorted(
        [
            {"feature": str(name), "coefficient": round(float(value), 4)}
            for name, value in zip(feature_names, coefficients)
        ],
        key=lambda item: abs(item["coefficient"]),
        reverse=True,
    )[:10]

    # Permutation importance for the best tree/linear model.
    sample = test.sample(n=min(400, len(test)), random_state=config.SEED)
    importance = permutation_importance(
        best_model,
        sample[NUMERIC_FEATURES + CATEGORICAL_FEATURES],
        sample["is_delayed"],
        n_repeats=3,
        random_state=config.SEED,
        scoring="roc_auc",
        n_jobs=-1,
    )
    top_permutation = sorted(
        [
            {"feature": feature, "importance": round(float(value), 4)}
            for feature, value in zip(
                NUMERIC_FEATURES + CATEGORICAL_FEATURES, importance.importances_mean
            )
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )[:10]

    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, config.MODELS / "delay_model.joblib")
    joblib.dump(logistic, config.MODELS / "delay_logreg.joblib")
    (config.MODELS / "delay_features.json").write_text(
        json.dumps(
            {
                "numeric": NUMERIC_FEATURES,
                "categorical": CATEGORICAL_FEATURES,
                "target": "is_delayed",
                "note": "Predicts probability of a non-On_Time delivery.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "target": "is_delayed",
        "split": "time-based 80/20 (chronological)",
        "train_rows": len(train),
        "test_rows": len(test),
        "models": results,
        "best_model": best_name,
        "top_logistic_features": top_linear,
        "top_permutation_features": top_permutation,
    }


def _zscore_metrics(test: pd.DataFrame, baselines: dict) -> dict:
    medians = test["vehicle_class"].map(
        {name: value["median_kmpl"] for name, value in baselines.items()}
    )
    mads = test["vehicle_class"].map(
        {name: value["mad_kmpl"] for name, value in baselines.items()}
    )
    z_scores = 0.6745 * (test["kmpl"] - medians) / mads.replace(0, np.nan)
    predictions = (z_scores.abs() > 3.5).fillna(False).to_numpy()
    y_true = test["is_anomaly"].to_numpy()
    return {
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "flagged_rate": round(float(predictions.mean()), 4),
        "threshold": "|robust z| > 3.5",
    }


def train_fuel_anomaly(frame: pd.DataFrame) -> dict:
    split = int(len(frame) * 0.8)
    train, test = frame.iloc[:split], frame.iloc[split:]
    clean_train = train[~train["is_anomaly"]]

    isolation = IsolationForest(
        n_estimators=300,
        contamination=config.ANOMALY_SHARE,
        random_state=config.SEED,
        n_jobs=-1,
    )
    isolation.fit(clean_train[FUEL_FEATURES])

    test_scores = isolation.decision_function(test[FUEL_FEATURES])
    test_predictions = isolation.predict(test[FUEL_FEATURES]) == -1
    y_true = test["is_anomaly"].to_numpy()

    isolation_metrics = {
        "precision": round(float(precision_score(y_true, test_predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, test_predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, test_predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, -test_scores)), 4),
        "flagged_rate": round(float(test_predictions.mean()), 4),
        "contamination": config.ANOMALY_SHARE,
    }

    baselines = {}
    for vehicle_class, group in clean_train.groupby("vehicle_class"):
        median = float(group["kmpl"].median())
        mad = float((group["kmpl"] - median).abs().median())
        baselines[vehicle_class] = {
            "median_kmpl": round(median, 4),
            "mad_kmpl": round(mad, 6),
            "count": len(group),
        }

    zscore = _zscore_metrics(test, baselines)

    joblib.dump(isolation, config.MODELS / "fuel_anomaly_model.joblib")
    (config.MODELS / "fuel_anomaly_config.json").write_text(
        json.dumps(
            {
                "features": FUEL_FEATURES,
                "class_baselines": baselines,
                "contamination": config.ANOMALY_SHARE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "primary_method": "robust_zscore",
        "isolation_forest": isolation_metrics,
        "zscore_baseline": zscore,
        "features": FUEL_FEATURES,
        "class_baselines": baselines,
    }


def compute_driver_scores(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame[~frame["is_anomaly"]].copy()
    class_medians = clean.groupby("vehicle_class")["kmpl"].median().to_dict()
    clean["expected_kmpl"] = clean["vehicle_class"].map(class_medians)
    clean["fuel_ratio"] = (clean["kmpl"] / clean["expected_kmpl"]).clip(0.5, 1.5)

    anomaly_counts = frame.groupby("driver_id")["is_anomaly"].sum()
    rows = []
    for driver_id, group in frame.groupby("driver_id"):
        driver_clean = clean[clean["driver_id"] == driver_id]
        trips = len(group)
        on_time_rate = float((group["delivery_status"] == "On_Time").mean())
        fuel_ratio = (
            float(driver_clean["fuel_ratio"].median()) if len(driver_clean) else 1.0
        )
        fuel_component = float(np.clip((fuel_ratio - 0.7) / 0.4, 0.0, 1.0))
        anomaly_rate = float(anomaly_counts.get(driver_id, 0)) / trips
        incident_component = float(np.clip(1 - anomaly_rate / 0.10, 0.0, 1.0))
        consistency_component = float(
            np.clip(1 - group["driver_score"].std(ddof=0) / 25, 0.0, 1.0)
        )
        raw = (
            DRIVER_WEIGHTS["on_time"] * on_time_rate
            + DRIVER_WEIGHTS["fuel"] * fuel_component
            + DRIVER_WEIGHTS["incident"] * incident_component
            + DRIVER_WEIGHTS["consistency"] * consistency_component
        )
        score = round(60 + 40 * raw, 1)
        grade = "A" if score >= 85 else "B" if score >= 75 else "C" if score >= 65 else "D"
        rows.append(
            {
                "driver_id": driver_id,
                "score": score,
                "grade": grade,
                "trips": trips,
                "on_time_rate": round(on_time_rate, 4),
                "fuel_ratio": round(fuel_ratio, 4),
                "anomaly_rate": round(anomaly_rate, 4),
                "score_std": round(float(group["driver_score"].std(ddof=0)), 2),
                "on_time_component": round(on_time_rate, 4),
                "fuel_component": round(fuel_component, 4),
                "incident_component": round(incident_component, 4),
                "consistency_component": round(consistency_component, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def driver_score_metrics(scores: pd.DataFrame) -> dict:
    drivers = pd.read_csv(config.DRIVERS_CSV)
    enriched = scores.merge(drivers, on="driver_id", how="left")
    enriched["experience_band"] = pd.cut(
        enriched["experience_years"],
        bins=[0, 5, 10, 15],
        labels=["1–5 yrs", "6–10 yrs", "11–15 yrs"],
    )

    def group_summary(column: str) -> dict:
        summary = (
            enriched.groupby(column, observed=True)["score"]
            .agg(["count", "mean", "min", "max"])
            .round(2)
            .reset_index()
        )
        return summary.to_dict(orient="records")

    depot = group_summary("depot_city")
    experience = group_summary("experience_band")
    depot_means = [row["mean"] for row in depot]
    max_gap = round(max(depot_means) - min(depot_means), 2) if depot_means else 0.0

    config.MODELS.mkdir(parents=True, exist_ok=True)
    (config.MODELS / "driver_score_config.json").write_text(
        json.dumps(
            {
                "weights": DRIVER_WEIGHTS,
                "methodology_version": DRIVER_METHOD_VERSION,
                "score_range": [60, 100],
                "grade_bands": {"A": ">=85", "B": "75–84.9", "C": "65–74.9", "D": "<65"},
                "note": "Descriptive coaching score; not an automated employment decision.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "methodology_version": DRIVER_METHOD_VERSION,
        "weights": DRIVER_WEIGHTS,
        "drivers": len(scores),
        "mean": round(float(scores["score"].mean()), 2),
        "std": round(float(scores["score"].std(ddof=0)), 2),
        "min": round(float(scores["score"].min()), 2),
        "max": round(float(scores["score"].max()), 2),
        "grade_counts": scores["grade"].value_counts().to_dict(),
        "fairness": {
            "by_depot": depot,
            "by_experience": experience,
            "max_depot_mean_gap": max_gap,
            "flag": max_gap > 10,
            "note": (
                "No protected attributes are used. Depot/experience are monitored "
                "for unintended disparities and reviewed before any action."
            ),
        },
    }


def route_benchmark() -> dict:
    samples = [
        ("Mumbai", ["Delhi", "Pune", "Chennai", "Hyderabad"]),
        ("Delhi", ["Mumbai", "Bengaluru", "Chennai", "Hyderabad"]),
        ("Bengaluru", ["Mumbai", "Delhi", "Pune", "Chennai"]),
        ("Chennai", ["Mumbai", "Delhi", "Pune", "Hyderabad"]),
    ]
    results = []
    for origin, stops in samples:
        result = optimize_route(origin, stops)
        results.append(
            {
                "origin": origin,
                "stops": stops,
                "baseline_km": result["baseline_distance_km"],
                "optimized_km": result["optimized_distance_km"],
                "savings_pct": result["savings_pct"],
            }
        )
    return {
        "samples": results,
        "average_savings_pct": round(
            float(np.mean([item["savings_pct"] for item in results])), 2
        ),
        "method": "exact permutations for <=7 stops; nearest-neighbour + 2-opt otherwise",
    }


def model_cards(metrics: dict) -> str:
    delay = metrics["delay"]["models"][metrics["delay"]["best_model"]]
    fuel = metrics["fuel_anomaly"]
    zscore = fuel["zscore_baseline"]
    isolation = fuel["isolation_forest"]
    driver = metrics["driver_score"]
    return f"""# Model Cards

Generated {metrics["generated_at"]}.

## 1. Delivery delay classifier

- **Purpose:** estimate the probability that a trip is not delivered on time.
- **Target:** `is_delayed` (1 = Delayed or Failed).
- **Features:** distance, driver score, load, vehicle age, month, monsoon flag,
  origin, destination, vehicle class and day of week.
- **Split:** chronological 80/20 ({metrics["delay"]["train_rows"]:,} train /
  {metrics["delay"]["test_rows"]:,} test).
- **Best model:** {metrics["delay"]["best_model"]}.
- **Metrics:** accuracy {delay["accuracy"]:.3f}, precision {delay["precision"]:.3f},
  recall {delay["recall"]:.3f}, F1 {delay["f1"]:.3f}, ROC-AUC {delay["roc_auc"]:.3f}.
- **Intended use:** prioritise proactive customer communication and dispatch
  support. Not a basis for penalising drivers.
- **Limitations:** synthetic data; no traffic, weather or customer-priority
  features; class imbalance limits recall.

## 2. Fuel anomaly detector

- **Purpose:** flag unusual fuel readings for investigation.
- **Primary detector:** robust z-score per vehicle class (|z| > 3.5).
- **Primary metrics:** precision {zscore["precision"]:.3f}, recall {zscore["recall"]:.3f},
  F1 {zscore["f1"]:.3f}, flagged rate {zscore["flagged_rate"]:.3f}.
- **Secondary model:** IsolationForest trained on clean trips; precision
  {isolation["precision"]:.3f}, recall {isolation["recall"]:.3f},
  F1 {isolation["f1"]:.3f}, ROC-AUC {isolation["roc_auc"]:.3f}.
- **Intended use:** maintenance/leak/theft investigation queue.
- **Limitations:** anomalies are injected; real sensor drift and refuelling
  events need domain validation.

## 3. Driver behaviour score

- **Purpose:** transparent coaching and recognition score.
- **Formula:** 60 + 40 × (0.40·on-time + 0.25·fuel efficiency + 0.20·incident-free
  + 0.15·consistency).
- **Range:** 60–100; grades A–D.
- **Drivers scored:** {driver["drivers"]}; mean {driver["mean"]}, std {driver["std"]}.
- **Fairness:** depot mean gap {driver["fairness"]["max_depot_mean_gap"]} points;
  flag = {driver["fairness"]["flag"]}.
- **Intended use:** coaching, recognition and support. **Not** an automated
  employment decision. Appeals and human review are mandatory.
- **Limitations:** outcome-based; does not observe harsh braking, speeding or
  hours-of-service because the dataset has no telematics.

## 4. Route optimiser

- **Method:** {metrics["route_optimizer"]["method"]}.
- **Average distance saving on benchmark routes:**
  {metrics["route_optimizer"]["average_savings_pct"]:.2f}%.
- **Intended use:** multi-stop delivery sequencing.
- **Limitations:** static road-distance matrix; no live traffic or time windows.
"""


def run() -> dict:
    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)

    frame = load_frame()
    delay = train_delay(frame)
    fuel = train_fuel_anomaly(frame)
    scores = compute_driver_scores(frame)
    scores.to_csv(config.REPORTS / "driver_scores.csv", index=False)
    driver = driver_score_metrics(scores)
    routes = route_benchmark()

    metrics = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "dataset": {
            "rows": len(frame),
            "vehicles": int(frame["vehicle_id"].nunique()),
            "drivers": int(frame["driver_id"].nunique()),
            "date_range": [
                frame["trip_start_time"].min().strftime("%Y-%m-%d"),
                frame["trip_start_time"].max().strftime("%Y-%m-%d"),
            ],
            "on_time_rate": round(float((frame["delivery_status"] == "On_Time").mean()), 4),
            "anomaly_rate": round(float(frame["is_anomaly"].mean()), 4),
        },
        "delay": delay,
        "fuel_anomaly": fuel,
        "driver_score": driver,
        "route_optimizer": routes,
    }

    (config.REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (config.REPORTS / "model_cards.md").write_text(model_cards(metrics), encoding="utf-8")

    print(f"Delay best model: {delay['best_model']} -> {delay['models'][delay['best_model']]}")
    print(f"Fuel anomaly: {fuel['isolation_forest']}")
    print(f"Driver scores: {driver['drivers']} drivers, mean {driver['mean']}")
    print(f"Route savings: {routes['average_savings_pct']}%")
    return metrics


if __name__ == "__main__":
    run()
