#!/bin/bash

python3 -m venv venv
source venv/bin/activate

pip install pip --upgrade
pip install -r requirements.txt
pip install -e .
pip install -U "jax[cuda12_pip]"
