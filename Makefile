.PHONY: install test lint demo notebooks
install: ; pip install -e ".[dev]"
test: ; pytest
lint: ; ruff check src tests
demo: ; mirage demo
notebooks: ; python kaggle/build_notebooks.py
