#!/bin/bash
set -euo pipefail

uv run python test/test_lbdn.py
uv run python test/test_linren.py
uv run python test/test_ren.py
uv run python test/test_r2dn.py
uv run python test/test_bilipren.py
uv run python test/test_bilipren_torch.py
