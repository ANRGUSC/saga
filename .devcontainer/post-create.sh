#!/usr/bin/env bash
set -e

echo "==> Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "==> Installing dependencies..."
uv sync

echo "==> Verifying installation..."
uv run python - <<'PY'
import importlib.metadata as md
import saga

try:
	version = saga.__version__
except AttributeError:
	version = md.version("anrg-saga")

print(f"SAGA version: {version}")
PY

echo "==> Development environment setup complete!"
echo ""
echo "Quick start:"
echo "  - Run examples:    uv run python scripts/examples/basic_example/main.py"
echo "  - Run tests:       uv run pytest tests/"
