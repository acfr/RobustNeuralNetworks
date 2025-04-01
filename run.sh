#!/bin/bash

source venv/bin/activate
python examples/train_observer.py
python examples/train_yoularen.py
python examples/train_sysid.py
python examples/plot_performance.py

python examples/test_expressivity.py
python examples/time_expressivity.py
python examples/plot_expressivity.py
