.PHONY: install lint type test check

install:
	poetry install

lint:
	poetry run ruff check .

type:
	poetry run mypy src

test:
	poetry run pytest

check: lint type test

