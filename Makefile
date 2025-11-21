PYTHON = python3
PIP = pip
VENV = .venv
BIN = $(VENV)/bin

.PHONY: setup run test clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

run:
	PYTHONPATH=. $(BIN)/python src/main.py

test:
	$(BIN)/python -m unittest discover tests

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
