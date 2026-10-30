.PHONY: setup data train api dashboard test lint

setup:
	uv venv .venv --python 3.12
	.venv/bin/uv pip install -e ".[dev]"

data:
	.venv/bin/python scripts/run_pipeline.py

train:
	.venv/bin/python scripts/train_models.py

api:
	.venv/bin/uvicorn fleet.api.main:app --reload --app-dir src

dashboard:
	.venv/bin/streamlit run src/fleet/dashboard/app.py

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests scripts
