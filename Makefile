PYTHON := $(shell if [ -x ./.venv/bin/python ]; then printf '%s' ./.venv/bin/python; else printf '%s' python3; fi)
PIP    := $(shell if [ -x ./.venv/bin/pip ]; then printf '%s' ./.venv/bin/pip; else printf '%s' pip3; fi)

.PHONY: help venv install test spec-validate compile-pipeline docker-build

help:
	@printf '%s\n' \
	'Available targets:' \
	'  venv              Create project virtualenv' \
	'  install           Install package and dev tooling' \
	'  test              Run pytest' \
	'  spec-validate     Validate sample pipeline spec' \
	'  compile-pipeline  Compile sample training pipeline' \
	'  docker-build      Build the single base Docker image'

venv:
	python3 -m venv .venv

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest

spec-validate:
	$(PYTHON) -m kfp_workflow.cli.main spec validate --spec configs/pipelines/sample_train.yaml

compile-pipeline:
	$(PYTHON) -m kfp_workflow.cli.main pipeline compile --spec configs/pipelines/sample_train.yaml --output pipelines/sample_train.yaml

docker-build:
	docker build -t kfp-workflow:latest -f docker/Dockerfile .
