-include .env
export DB_HOST DB_PORT DB_NAME DB_USER DB_PASS

VENV_DIR := $(abspath ../fyp_env)
VENV := $(VENV_DIR)
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

$(VENV):
	@echo "Creating virtual environment at $(VENV_DIR)..."
	python3 -m venv $(VENV)
	@echo "✅ Virtual environment created"

check_venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ Virtual environment not found. Run 'make install-dev' first."; \
		exit 1; \
	fi

install-dev: $(VENV)
	@echo "Installing dev dependencies (no Pi hardware)..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-dev.txt
	@echo "✅ Dev installation complete"
	@echo "Activate with: source ../fyp_env/bin/activate"

install-pi: $(VENV)
	@echo "Installing Pi dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-pi.txt
	@echo "✅ Pi installation complete"

lint: check_venv
	$(PYTHON) -m flake8

run: check_venv
	$(PYTHON) src/main.py

clean:
	@echo "Cleaning cache files..."
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean complete"

fclean: clean
	@echo "Removing virtual environment..."
	rm -rf $(VENV)
	@echo "✅ Full clean complete"

re: fclean install-dev

setup:
	@echo "Create a .env file in the project root with the following variables:"
	@echo ""
	@echo "  DB_HOST=localhost"
	@echo "  DB_PORT=5432"
	@echo "  DB_NAME=smart_fire"
	@echo "  DB_USER=your_db_user          ← change this"
	@echo "  DB_PASS=your_db_password      ← change this, do NOT use something obvious"
	@echo ""
	@echo "Then run:"
	@echo "  make db-install"
	@echo "  make db-start"
	@echo "  make db-setup"
	@echo "  make db-migrate"

db-install:
	@which psql > /dev/null 2>&1 && echo "✅ PostgreSQL already installed" || (sudo apt update && sudo apt install -y postgresql postgresql-contrib)

db-start:
	sudo pg_ctlcluster 16 main start

db-stop:
	sudo pg_ctlcluster 16 main stop

db-setup:
	sudo -u postgres psql -c "CREATE USER $(DB_USER) WITH PASSWORD '$(DB_PASS)';" || true
	sudo -u postgres psql -c "CREATE DATABASE $(DB_NAME) OWNER $(DB_USER);" || true
	sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE $(DB_NAME) TO $(DB_USER);"
	@echo "✅ DB user and database ready"

db-migrate:
	PGPASSWORD=$(DB_PASS) psql -U $(DB_USER) -d $(DB_NAME) -h $(DB_HOST) -f src/think/database/schema.sql
	@echo "✅ Schema applied"

db-reset:
	sudo -u postgres psql -c "DROP DATABASE IF EXISTS $(DB_NAME);"
	sudo -u postgres psql -c "CREATE DATABASE $(DB_NAME) OWNER $(DB_USER);"
	$(MAKE) db-migrate
	@echo "✅ DB reset complete"

db-shell:
	PGPASSWORD=$(DB_PASS) psql -U $(DB_USER) -d $(DB_NAME) -h $(DB_HOST)

setup_cam: check_venv
	@echo "Setting up IMX500 AI Camera on Raspberry Pi..."
	@echo ""
	@echo "Step 1 — Installing IMX500 firmware and tools..."
	sudo apt update
	sudo apt install -y imx500-all imx500-tools
	@echo "✅ IMX500 firmware and tools installed"
	@echo ""
	@echo "Step 2 — Installing Ultralytics for model export..."
	$(PIP) install ultralytics
	@echo "✅ Ultralytics installed"
	@echo ""
	@echo "Step 3 — Installing Picamera2 if not already present..."
	$(PIP) install picamera2 || sudo apt install -y python3-picamera2
	@echo "✅ Picamera2 ready"
	@echo ""
	@echo "⚠️  IMPORTANT: A reboot is required after installing IMX500 firmware."
	@echo "   Run: sudo reboot"
	@echo ""
	@echo "After reboot, verify the camera is detected with:"
	@echo "   rpicam-hello --list-cameras"
	@echo ""
	@echo "To package your trained model, run:"
	@echo "   make package_model MODEL=path/to/best.pt"

package_model: check_venv
	@if [ -z "$(MODEL)" ]; then \
		echo "❌ No model specified. Usage: make package_model MODEL=path/to/best.pt"; \
		exit 1; \
	fi
	@echo "Exporting $(MODEL) to IMX500 format..."
	$(PYTHON) -c "\
from ultralytics import YOLO; \
model = YOLO('$(MODEL)'); \
model.export(format='imx', data='data.yaml')"
	@echo "✅ Export complete — packerOut.zip generated"
	@echo ""
	@echo "Packaging .rpk file..."
	imx500-package -i $(dir $(MODEL))imx_model/packerOut.zip -o network.rpk
	@echo "✅ network.rpk ready — copy to Pi and load via Picamera2 at boot"

help:
	@echo "Available commands:"
	@echo ""
	@echo "  make setup        - Show required .env variables for first-time setup"
	@echo ""
	@echo "  make install-dev  - Create venv and install dev dependencies (Linux)"
	@echo "  make install-pi   - Create venv and install Pi dependencies"
	@echo "  make lint         - Run flake8 linter"
	@echo "  make run          - Run main.py"
	@echo "  make clean        - Remove cache files only"
	@echo "  make fclean       - Delete virtual environment"
	@echo "  make re           - Full rebuild (fclean + install-dev)"
	@echo ""
	@echo "  make db-install   - Install PostgreSQL if not already installed"
	@echo "  make db-start     - Start the PostgreSQL server"
	@echo "  make db-stop      - Stop the PostgreSQL server"
	@echo "  make db-setup     - Create DB user and database (run once)"
	@echo "  make db-migrate   - Apply schema.sql to the database"
	@echo "  make db-reset     - Drop and recreate database then re-migrate"
	@echo "  make db-shell     - Open a psql shell to inspect data"
	@echo ""
	@echo "  make setup_cam    - Install IMX500 camera tools and Ultralytics on Pi"
	@echo "  make package_model MODEL=path/to/best.pt"
	@echo "                    - Export trained YOLO model to IMX500 .rpk format"