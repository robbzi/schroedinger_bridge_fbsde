.PHONY: check lint lintfix format checkformat typecheck pytest test deptry ccc check-lock lock install-venv

PACKAGE_DIR=src

lint:
	uv run ruff check ${PACKAGE_DIR}

lintfix:
	uv run ruff check --fix ${PACKAGE_DIR}

format:
	uv run ruff format ${PACKAGE_DIR}

checkformat:
	uv run ruff format --check --diff ${PACKAGE_DIR}

typecheck:
	PYRIGHT_PYTHON_PYLANCE_VERSION=latest-release uv run pyright

pytest:
	$(info Running tests...)
	uv run pytest ${PACKAGE_DIR}/tests -vv

test: pytest

check: checkformat lint typecheck test

deptry:
	uv run deptry .

ccc: check-lock format lintfix typecheck test

check-lock:
	uv lock --check

lock:
	uv lock

install-venv:
	uv sync 
