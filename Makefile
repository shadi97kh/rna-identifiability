# Same Energies, Different Folds: tests and lint.

PY ?= python

.PHONY: test lint

test:
	$(PY) -m pytest tests

lint:
	$(PY) -m ruff check scripts thermo tests
