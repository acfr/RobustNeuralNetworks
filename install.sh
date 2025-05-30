#!/bin/bash

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

# Instantiate into a venv
python3 -m venv venv
source venv/bin/activate

# Install basic requirements
pip install pip --upgrade
pip install -r requirements.txt

# Install the correct jax based upon hardware
if check_cuda; then
    pip install "jax[cuda12_pip]==0.5.3"
else
    pip install "jax<=0.5.3"
fi

# Install the package locally in editable mode for dev
# Flag PEP517 to avoid issues with deprecated versions of local/editable 
# installs using setuptools (https://github.com/pypa/pip/issues/11457)
pip install -e . --use-pep517
