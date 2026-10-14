# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Backend-agnostic loader for the golden input/output files in `tests/golden/`.

Each `tests/golden/<network>.json` holds a list of cases:

    {
      "name":    human readable case id,
      "config":  constructor arguments, using the JAX argument names,
      "params":  direct (unconstrained) parameters, in the JAX/Flax layout,
      "inputs":  model inputs,
      "outputs": expected outputs
    }

The JAX tests load `params` as-is; the PyTorch tests translate them to
a `state_dict` (see `backends.py`). Because both backends are checked against
the same file, passing both implies they agree with each other.

Regenerate the files with `uv run python tests/generate_golden.py`.
"""

import json
from pathlib import Path

import numpy as np

GOLDEN_DIR = Path(__file__).parent / "golden"

# Tolerances used when comparing against the stored outputs (float32).
ATOL = 1e-5
RTOL = 1e-4


def to_arrays(obj):
    """Recursively turn (nested) lists of numbers into float32 arrays.

    Zero-size arrays are stored as `{"shape": [...]}` since an empty list
    can't carry its shape.
    """
    if isinstance(obj, dict):
        if set(obj) == {"shape"}:
            return np.zeros(obj["shape"], dtype=np.float32)
        return {k: to_arrays(v) for k, v in obj.items()}
    if obj is None:
        return None
    return np.asarray(obj, dtype=np.float32)


def load(network):
    """Return the list of golden cases for `network`."""
    with open(GOLDEN_DIR / f"{network}.json") as f:
        data = json.load(f)
    cases = data["cases"]
    for case in cases:
        for key in ("params", "inputs", "outputs"):
            case[key] = to_arrays(case[key])
    return cases


def cases(network):
    """Cases for `pytest.mark.parametrize`, e.g.

        @pytest.mark.parametrize("case", golden.cases("plnet"), ids=golden.case_id)
    """
    return load(network)


def case_id(case):
    return case["name"]


def assert_close(actual, expected, atol=ATOL, rtol=RTOL, path="outputs"):
    """Compare (nested) outputs; `None` entries must match `None`."""
    if isinstance(expected, dict):
        assert set(actual) == set(expected), f"{path}: keys {set(actual)} != {set(expected)}"
        for k in expected:
            assert_close(actual[k], expected[k], atol, rtol, f"{path}.{k}")
    elif expected is None:
        assert actual is None, f"{path}: expected None"
    else:
        assert actual is not None, f"{path}: got None"
        np.testing.assert_allclose(
            np.asarray(actual), expected, atol=atol, rtol=rtol, err_msg=path
        )


# ---------------------------------------------------------------------------
# Writing (used by generate_golden.py)
# ---------------------------------------------------------------------------
def _fmt_array(a):
    a = np.asarray(a, dtype=np.float32)
    if a.size == 0:
        return json.dumps({"shape": list(a.shape)})
    if a.ndim == 0:
        return np.format_float_positional(np.float32(a), unique=True, trim="-")
    return "[" + ", ".join(_fmt_array(x) for x in a) + "]"


def _dump(obj, indent=0, key_is_data=False):
    pad = "  " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f'{pad}  {json.dumps(k)}: {_dump(v, indent + 1, key_is_data or k in DATA_KEYS)}'
            for k, v in obj.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if obj is None:
        return "null"
    if isinstance(obj, (np.ndarray, list, tuple)) and key_is_data:
        arr = np.asarray(obj)
        if arr.dtype == object:  # ragged / mixed list, recurse
            return "[" + ", ".join(_dump(x, indent + 1, True) for x in obj) + "]"
        return _fmt_array(arr)
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        items = [f"{pad}  {_dump(v, indent + 1)}" for v in obj]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(obj)


# top-level fields whose values are numeric data (written on one line per array)
DATA_KEYS = {"params", "inputs", "outputs"}


def dump(network, description, cases_):
    """Serialise cases to `tests/golden/<network>.json` (arrays inline)."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    text = _dump({"network": network, "description": description, "cases": cases_})
    json.loads(text)  # sanity check: must be valid JSON
    (GOLDEN_DIR / f"{network}.json").write_text(text + "\n")
