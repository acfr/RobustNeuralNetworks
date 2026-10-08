#!/bin/bash
# Run the test suite. Extra arguments are passed to pytest, e.g.
#   ./run_tests.sh -k torch
#   ./run_tests.sh -m jax -n auto
set -euo pipefail

uv run pytest "$@"
