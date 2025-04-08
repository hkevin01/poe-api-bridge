.PHONY: start test test-verbose install deploy clean install-prod install-dev format

# Install both production and development dependencies
install:
	pip install -e ".[dev]"

# Install with production dependencies only
install-prod:
	pip install -e .

# Install with dev dependencies (same as install)
install-dev:
	pip install -e ".[dev]"

# Start the development server
start:
	python3 local_run.py


start-dev:
	python3 local_run.py --reload

# Run tests
test:
	pytest

# Run tests with verbose output
test-verbose:
	pytest -v

verify:
	python3 verify_regular_query.py

# Format code with black
format:
	black .

# Clean up artifacts
clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Deploy to Modal
deploy:
	modal deploy modal_app.py 