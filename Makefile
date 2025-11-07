.PHONY: install lint test docs serve

install:
\tpoetry install --no-root

lint:
\tpoetry run ruff check .
\tpoetry run black --check .
\tpoetry run isort --check-only .

test:
\tpoetry run pytest -q

docs:
\tpoetry run mkdocs build --strict

serve:
\tpoetry run mkdocs serve -a 0.0.0.0:8000
