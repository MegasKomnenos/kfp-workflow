PYTHON := $(shell if [ -x ./.venv/bin/python ]; then printf '%s' ./.venv/bin/python; else printf '%s' python3; fi)
PIP := $(shell if [ -x ./.venv/bin/pip ]; then printf '%s' ./.venv/bin/pip; else printf '%s' pip3; fi)

.PHONY: help venv install test spec-validate compile-pipeline

help:
	@printf '%s\n' \
	'Available targets:' \
	'  venv              Create project virtualenv' \
	'  install           Install package and dev tooling' \
	'  test              Run pytest' \
	'  spec-validate     Validate smoke experiment spec' \
	'  compile-pipeline  Compile smoke pipeline to pipelines/'

venv:
	python3 -m venv .venv

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest

spec-validate:
	softs-new spec validate --spec configs/experiments/fd001_smoke.yaml

compile-pipeline:
	mkdir -p pipelines
	softs-new pipeline compile \
	  --spec configs/experiments/fd001_smoke.yaml \
	  --output pipelines/fd001_smoke.yaml
