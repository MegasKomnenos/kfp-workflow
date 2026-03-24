PYTHON := $(shell if [ -x ./.venv/bin/python ]; then printf '%s' ./.venv/bin/python; else printf '%s' python3; fi)
PIP := $(shell if [ -x ./.venv/bin/pip ]; then printf '%s' ./.venv/bin/pip; else printf '%s' pip3; fi)

.PHONY: help venv install test spec-validate compile-pipeline

help:
	@printf '%s\n' \
	'Available targets:' \
	'  venv              Create project virtualenv' \
	'  install           Install package and dev tooling' \
	'  test              Run pytest' \
	'  spec-validate     Validate the default core experiment spec' \
	'  compile-pipeline  Compile the default core KFP pipeline'

venv:
	python3 -m venv .venv

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest

spec-validate:
	$(PYTHON) -m multirocket_new.cli.main spec validate --spec configs/experiments/fd_all_core_default.yaml

compile-pipeline:
	$(PYTHON) -m multirocket_new.cli.main pipeline compile --spec configs/experiments/fd_all_core_default.yaml --output compiled/fd_all_core_default.yaml
