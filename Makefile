.PHONY: clean clean-test clean-pyc clean-build build help
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean-pyc: ## clean python cache files
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.pytest_cache' -exec rm -fr {} +

clean-test: ## cleanup pytests leftovers
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr test_results/
	rm -f *report.html
	rm -f log.html
	rm -f test-results.html
	rm -f output.xml

test: clean-test ## Run pytest unit tests
	python3 -m pytest --verbosity=1

test-debug: ## Run unit tests with debugging enabled
	python3 -m pytest --pdb

test-coverage: clean-test ## Run unit tests and check code coverage
	PYTHONPATH=src python3 -m pytest --cov=src tests/ --disable-warnings

setup: requirements venv-create test-setup ## setup & run after downloaded repo

test-setup: ## installs pytest singular package for local testing
	python3 -m pip install pytest

requirements: ## installs all requirements
	python3 -m pip install -r requirements.txt

docs: # opens your browser to the webapps testing docs
	open http://localhost:5000/docs
	xdg-open http://localhost:5000/docs
	. http://localhost:5000/docs

venv-create: venv-remove ## cleans the .venv then creates a venv in the folder .venv
	python3 -m venv .venv

venv-remove: ## removes the .venv folder
	rm -rf .venv

freeze:
	pip freeze > requirements.txt