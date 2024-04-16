# WARN: Make sure to use tabs for indentation
# else will get errors.
# WARN: Make sure to include .PHONY for every command
# else command won't run if file with same name exists
# NOTE: To call another target use: $(MAKE) target_name

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' 'Makefile' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
.DEFAULT_GOAL := help

# ------------------------------------
# Dependency utilities
# ------------------------------------

# TODO(deps):
# Replace --extra with --only-extra when added to pip-tools
# and sync both requirements.txt files. See PR:
# - https://github.com/jazzband/pip-tools/pull/1960

export PIP_REQUIRE_VIRTUALENV=1

.PHONY: dep_outdated
dep_outdated:  ## Show outdated requirements
	python -m pip list --outdated

.PHONY: dep_update_piptools
dep_update_piptools: ## Update piptools and related requirements
	python -m pip install --upgrade pip
    # Setuptools and wheel can be removed in Python 3.12
	python -m pip install --upgrade setuptools wheel pip-tools build


.PHONY: dep_compile
dep_compile:  ## Compile requirements
	$(MAKE) dep_update_piptools
	python -m piptools compile -o requirements.txt
	python -m piptools sync requirements.txt

.PHONY: dep_upgrade
dep_upgrade:  ## Compile and upgrade requirements
	$(MAKE) dep_update_piptools
	python -m piptools compile --upgrade -o requirements.txt
	python -m piptools sync requirements.txt

.PHONY: dep_sync
dep_sync:  ## Sync deps
	$(MAKE) dep_update_piptools
	python -m piptools sync requirements.txt

# ------------------------------------
# Development utilities
# ------------------------------------

.PHONY: dev_full
dev_full: ## Migrate, translate and start dev server
	$(MAKE) dep_sync
	$(MAKE) dev

.PHONY: dev
dev: ## Start dev server
	jupyter notebook
