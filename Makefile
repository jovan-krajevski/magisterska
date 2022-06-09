export PIP_REQUIRE_VIRTUALENV=1

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' 'Makefile' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help

.PHONY: compile_deps
compile_deps:  ## Compile requirements txt
	pip-compile requirements/requirements.in

.PHONY: compile_deps
upgrade_deps:  ## Compile and upgrade requirements txt
	pip-compile requirements/requirements.in --upgrade

.PHONY: sync_deps
sync_deps:  ## Sync deps from requirements to venv
	pip install --upgrade pip
	pip install --upgrade setuptools wheel pip-tools
	pip install cython numpy
	pip-sync requirements/requirements.txt
	pip install --upgrade notebook
	pip install jupyter_contrib_nbextensions
	jupyter contrib nbextension install --user
