.PHONY: start test test-verbose install deploy clean install-prod install-dev format venv web-dev web-build

# Create virtual environment with Python 3.12
venv:
	/opt/homebrew/bin/python3.12 -m venv venv
	@echo "Virtual environment created with Python 3.12. Activate with: source venv/bin/activate"

# Install both production and development dependencies
install: venv
	./venv/bin/pip install -e ".[dev]"

# Install with production dependencies only
install-prod: venv
	./venv/bin/pip install -e .

# Install with dev dependencies (same as install)
install-dev: venv
	./venv/bin/pip install -e ".[dev]"

# Start the development server
start:
	./venv/bin/python local_run.py

# Start the development server with auto-reload
start-dev:
	./venv/bin/python local_run.py --reload

# Run tests
test:
	./venv/bin/pytest

# Run tests with verbose output
test-verbose:
	./venv/bin/pytest -v

verify:
	./venv/bin/python verify_regular_query.py

# Format code with black
format:
	./venv/bin/black .

# Clean up artifacts
clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Deploy to Modal
deploy:
	./venv/bin/modal deploy modal_app.py

# Web development - install deps and run dev server
web-dev:
	npm install --prefix web
	npm run dev --prefix web

# Build web app to static directory
web-build:
	npm install --prefix web
	npm run build --prefix web