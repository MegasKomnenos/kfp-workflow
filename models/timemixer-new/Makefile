PYTHON := $(shell if [ -x ./.venv/bin/python ]; then printf '%s' ./.venv/bin/python; else printf '%s' python3; fi)
PIP := $(shell if [ -x ./.venv/bin/pip ]; then printf '%s' ./.venv/bin/pip; else printf '%s' pip3; fi)

.PHONY: help venv install test

help:
	@printf '%s\n' \
	'Available targets:' \
	'  venv              Create project virtualenv' \
	'  install           Install package and dev tooling' \
	'  test              Run pytest'

venv:
	python3 -m venv .venv

install:
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest
