.PHONY: start test test-verbose install deploy clean install-prod install-dev format venv web-dev web-build rotate-logs

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

# Rotate logs if they exist and are larger than 10MB
rotate-logs:
	@mkdir -p logs
	@if [ -f logs/server.log ] && [ $$(stat -f%z logs/server.log 2>/dev/null || echo 0) -gt 10485760 ]; then \
		timestamp=$$(date +"%Y%m%d_%H%M%S"); \
		mv logs/server.log logs/server_$$timestamp.log; \
		echo "Rotated server.log to server_$$timestamp.log"; \
	fi
	@find logs -name "server_*.log" -mtime +7 -delete 2>/dev/null || true

# Start the development server with auto-reload
start-dev: rotate-logs
	@mkdir -p logs
	script -q /dev/null ./venv/bin/python local_run.py --reload 2>&1 | tee logs/server.log

# Run tests
test:
	./venv/bin/pytest

# Run tests with verbose output
test-verbose:
	./venv/bin/pytest -v

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
deploy: web-build
	./venv/bin/modal deploy modal_app.py

# Web development - install deps and run dev server
web-dev:
	npm install --prefix web
	npm run dev --prefix web

# Build web app to static directory
web-build:
	npm install --prefix web
	npm run build --prefix web