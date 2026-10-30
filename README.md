# Comprehensive Logistics & Fleet Management Platform

An end-to-end fleet analytics prototype built for the HCLTech Career Shaper
III Year Industry Project (Project 24, Digital-Data-AI / AIML).

The platform turns a 5,000-trip fleet log into operational insight:

- **Delivery analytics** — on-time performance, delay drivers, route KPIs.
- **Fuel monitoring** — efficiency baselines and anomaly detection.
- **Driver behaviour scoring** — transparent, weighted and auditable.
- **Route optimisation** — nearest-neighbour + 2-opt over a six-city network.
- **ML services** — delay prediction, fuel-anomaly detection and driver segmentation.
- **REST API + dashboard** — FastAPI/OpenAPI and a Streamlit/Plotly UI.

> **Data note:** the supplied 500-row dataset is preserved in
> `data/raw/dataset_original.csv`. The processed 5,000-row dataset is a
> cleaned and seeded synthetic augmentation for academic use. It contains no
> real personal data. See `data/processed/quality_report.md` and
> `docs/ethics/`.

## Quickstart

```bash
# 1. Environment
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Reproduce the dataset
python scripts/run_pipeline.py

# 3. Train models
python scripts/train_models.py

# 4. Run the API (http://localhost:8000/docs)
uvicorn fleet.api.main:app --reload --app-dir src

# 5. Run the dashboard (http://localhost:8501)
streamlit run src/fleet/dashboard/app.py
```

With Docker:

```bash
docker compose up --build
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

The default database is SQLite (`fleet.db`) so the project runs with zero
setup. Set `DATABASE_URL` to a PostgreSQL DSN to use the Docker service.

## Project structure

```
data/raw/            original supplied dataset (untouched)
data/processed/      5,000-row dataset, metadata, anomaly labels, quality report
docs/                data dictionary, architecture, ethics pack, synopsis
notebooks/           EDA notebook
reports/             figures, metrics, project report
src/fleet/           data pipeline, models, API, dashboard
scripts/             pipeline and training entry points
tests/               pytest suite
```

## Dataset

`data/processed/dataset.csv` keeps the exact nine columns from the brief:

`trip_id, vehicle_id, driver_id, origin, destination, distance_km, fuel_L, delivery_status, driver_score`

Timestamps and operational context live in `trip_metadata.csv` so the
canonical schema is not broken. Anomaly ground truth lives in
`anomaly_labels.csv` and is never joined into model features.

## Ethics

Driver behaviour data can affect employment decisions. This project therefore:

1. Uses a published, weighted scoring formula — not an opaque model.
2. Reports the score's components and provides an appeals path.
3. Excludes protected attributes and PII.
4. Audits score distributions for fairness.
5. Uses synthetic/anonymised data only.

See `docs/ethics/` for the methodology, fairness audit and signed declaration.

## Academic context

HCLTech Career Shaper Programme · Academic Year 2025-26 · Individual Project
(4 weeks) · Project 24.
