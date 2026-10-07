#!/bin/bash
set -euo pipefail


while true; do
  read -p "This will create a virtual environment (venv) at ./.venv/ with uv, would you like to continue? (y/n) " answer
  if [[ "$answer" =~ ^[yYnN]$ ]]; then break; fi
  echo "Please answer y or n."
done

# Exit if user does not want to continue
if [[ "$answer" == "n" ]]; then
    exit 1
fi

# Function to check if CUDA is available
check_cuda() {
    # Check if nvidia-smi exists and works
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "CUDA detected: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)"
            return 0
        fi
    fi

    # Check if nvcc (CUDA compiler) is available
    if command -v nvcc &> /dev/null; then
        echo "CUDA compiler detected: $(nvcc --version | grep release | awk '{print $6}' | cut -c2-)"
        return 0
    fi

    echo "No CUDA detected"
    return 1
}

# Check that uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "See https://docs.astral.sh/uv/getting-started/installation/ for Windows."
    exit 1
fi

# Use pinned Python version (see .python-version)
uv python install

# Install dependencies and the package itself (editable by default) into ./.venv,
# choosing the correct jax build based upon the available hardware
if check_cuda; then
    uv sync --extra examples --extra cuda13
else
    uv sync --extra examples --extra cpu
fi

echo
echo "Done. Run the tests with ./run_tests.sh, or a script with 'uv run python examples/<script_name>.py'."
