# Convenience developer commands (use with: make <target>)

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check .
	black --check .
	isort --check-only .
	mypy src

format:
	ruff check --fix . || true
	black .
	isort .

test:
	pytest -q

coverage:
	pytest --cov=src --cov-report=term-missing

pre-commit:
	pre-commit install
	pre-commit run --all-files
