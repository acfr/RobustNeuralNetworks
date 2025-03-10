#!/bin/bash

source venv/bin/activate
python examples/train_yoularen.py
python examples/train_observer.py
python examples/train_sysid.py