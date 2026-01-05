#!/usr/bin/env bash
set -e

echo "==> Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y graphviz libgraphviz-dev

echo "==> Installing Python package in development mode..."
pip install --upgrade pip
pip install -e .

echo "==> Installing additional development tools..."
pip install ruff mypy jupyterlab

echo "==> Verifying installation..."
python - <<'PY'
import importlib.metadata as md
import saga

try:
	version = saga.__version__
except AttributeError:
	version = md.version("anrg-saga")

print(f"SAGA version: {version}")
PY
dot -V

echo "==> Development environment setup complete!"
echo ""
echo "Quick start:"
echo "  - Run examples:    python scripts/examples/basic_example/main.py"
echo "  - Run tests:       pytest tests/"
