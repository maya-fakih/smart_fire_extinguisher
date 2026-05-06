# Project Automation Setup

## What We Did Tonight

We set up the project structure so anyone can clone and run it without messing with their system.

---

## 1. Virtual Environment (`fyp_env`)

- All Python packages install inside this folder, not system-wide
- No dependency conflicts with other projects
- Easy to delete if something breaks

---

## 2. Makefile

Handles everything from one place:

| Command | What it does |
|---------|-------------|
| `make install` | Creates venv + installs requirements.txt |
| `make lint` | Runs flake8 inside venv |
| `make run` | Runs main.py inside venv (to be added) |
| `make clean` | Removes cache files only |
| `make fclean` | Deletes venv completely |
| `make re` | Full rebuild (fclean + install) |

**No need to activate the venv manually.** Make handles it.

---

## 3. `requirements.txt`

Current packages:

```txt
ultralytics   # YOLO for fire detection
flake8        # Code linter
```

## 4. GitHub Workflow (`.github/workflows/lint.yml`)

```yaml
name: Lint

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install flake8
      - run: flake8
```

Runs flake8 on every push and pull request

If code doesn't follow PEP8 rules, the workflow fails

Keeps the codebase clean automatically


### 5. How to run everything

From the project root (fire_robot/):

bash
# First time setup
make install

# Check code style
make lint

# Run the project (once main.py exists)
make run

# Clean up (keep venv)
make clean

# Delete everything and start fresh
make fclean
make re
Notes for Windows Users
Make doesn't come installed on Windows by default. Options:

Use WSL (Windows Subsystem for Linux) — recommended

Install Make via Chocolatey or MSYS2

Or run the commands manually:

cmd
python -m venv fyp_env
fyp_env\Scripts\pip install -r requirements.txt
fyp_env\Scripts\flake8
fyp_env\Scripts\python src/main.py


### 6. Why This Setup Matters
Benefit	        Explanation
Portable	    Works on any machine with Python
Reproducible	Same environment for everyone
Clean	        No system pollution
Automated	    One command to get started

