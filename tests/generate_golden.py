# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Regenerate the golden input/output files in `tests/golden/`.

    uv run python tests/generate_golden.py            # all networks
    uv run python tests/generate_golden.py plnet lbdn # just these

The JAX implementation is used as the reference: random (seeded) parameters and
inputs are generated, the outputs are computed with JAX, and everything is
written to JSON. The JAX *and* PyTorch tests then check against these files.

Only regenerate on purpose (e.g. when a network is deliberately changed or a case
is added) and review the diff: regenerating after an accidental behaviour change
would hide it from the JAX tests (the PyTorch tests would still flag it).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jax
import jax.numpy as jnp
import numpy as np

import golden
from backends import JaxNet, build_jax_model

BATCH = 4

# Parameters that act as (positive) scales are perturbed multiplicatively so
# they stay positive; everything else gets additive noise (so biases are non-zero).
_SCALE_KEYS = ("a", "fq", "p", "alpha")


def _is_scale(name):
    return name in _SCALE_KEYS or name.startswith("fab")


def perturb(params, rng, noise=0.3):
    def go(tree):
        out = {}
        for k, v in tree.items():
            if isinstance(v, dict):
                out[k] = go(v)
                continue
            v = np.asarray(v, dtype=np.float32)
            if _is_scale(k):
                v = v * np.exp(0.2 * rng.standard_normal(v.shape)).astype(np.float32)
            else:
                v = v + noise * rng.standard_normal(v.shape).astype(np.float32)
            out[k] = v.astype(np.float32)
        return out
    return go(params)


def _randn(rng, *shape):
    return rng.standard_normal(shape).astype(np.float32)


def _make_case(network, name, config, rng, inputs, init_args):
    model = build_jax_model(network, config)
    variables = model.init(jax.random.key(0), *init_args(inputs))
    params = perturb(jax.tree.map(np.asarray, variables["params"]), rng)
    jparams = {"params": jax.tree.map(jnp.asarray, params)}
    outputs = JaxNet(network, config, model, jparams).forward(inputs)
    return {
        "name": name,
        "config": config,
        "params": params,
        "inputs": inputs,
        "outputs": outputs,
    }


# --- init-argument adapters ---------------------------------------------------
def _x(i):
    return (jnp.asarray(i["x"]),)


def _state_u(i):
    return (jnp.asarray(i["state"]), jnp.asarray(i["u"]))


def _carry_u(i):
    c = i["carry"]
    carry = {
        "rens": [jnp.asarray(r) for r in c["rens"]],
        "dyn_in": None if c["dyn_in"] is None else jnp.asarray(c["dyn_in"]),
        "dyn_out": None if c["dyn_out"] is None else jnp.asarray(c["dyn_out"]),
    }
    return (carry, jnp.asarray(i["u"]))


# --- the cases ---------------------------------------------------------------
def static_cases(network, specs, size, seed):
    rng = np.random.default_rng(seed)
    return [
        _make_case(network, name, cfg, rng, {"x": _randn(rng, BATCH, size(cfg))}, _x)
        for name, cfg in specs
    ]


def ren_cases(network, specs, seed):
    rng = np.random.default_rng(seed)
    out = []
    for name, cfg in specs:
        inputs = {
            "state": _randn(rng, BATCH, cfg["state_size"]),
            "u": _randn(rng, BATCH, cfg["input_size"]),
        }
        out.append(_make_case(network, name, cfg, rng, inputs, _state_u))
    return out


def make_qsr(rng, nu, ny):
    X, Y, S = _randn(rng, ny, ny), _randn(rng, nu, nu), _randn(rng, nu, ny)
    Q = -X.T @ X
    R = S @ np.linalg.solve(Q, S.T) + Y.T @ Y
    return Q.astype(np.float32), S, R.astype(np.float32)


def gen_unitary():
    specs = [
        ("size3_bias", {"input_size": 3, "use_bias": True}),
        ("size1_bias", {"input_size": 1, "use_bias": True}),
        ("size5_bias", {"input_size": 5, "use_bias": True}),
    ]
    return static_cases("unitary", specs, lambda c: c["input_size"], 1)


def gen_monlipnet():
    base = {"input_size": 3, "mu": 0.5, "nu": 4.0, "tau": 8.0}
    specs = [
        ("units4x4_mu_nu_fixed", base | {"units": [4, 4], "is_mu_fixed": True, "is_nu_fixed": True}),
        ("units3_mu_tau_fixed", base | {"units": [3], "nu": 4.0, "tau": 8.0,
                                        "is_mu_fixed": True, "is_tau_fixed": True}),
        ("units3x3x3_nu_tau_fixed", base | {"units": [3, 3, 3], "is_nu_fixed": True,
                                            "is_tau_fixed": True}),
    ]
    return static_cases("monlipnet", specs, lambda c: c["input_size"], 2)


def gen_bilipnet():
    base = {"input_size": 3, "mu": 0.5, "nu": 4.0, "tau": 8.0,
            "is_mu_fixed": True, "is_nu_fixed": True}
    specs = [
        ("depth1_units4", base | {"units": [4], "depth": 1}),
        ("depth2_units4x4", base | {"units": [4, 4], "depth": 2}),
        ("depth3_units3", base | {"units": [3], "depth": 3}),
    ]
    return static_cases("bilipnet", specs, lambda c: c["input_size"], 3)


def gen_plnet():
    bilip = {"input_size": 3, "units": [4], "mu": 0.5, "nu": 4.0, "tau": 8.0,
             "is_mu_fixed": True, "is_nu_fixed": True, "depth": 2}
    specs = [
        ("default", {"BiLipBlock": bilip}),
        ("add_constant", {"BiLipBlock": bilip, "add_constant": True}),
        ("optimal_point", {"BiLipBlock": bilip, "optimal_point": [[1.0, -0.5, 0.3]]}),
    ]
    return static_cases("plnet", specs, lambda c: c["BiLipBlock"]["input_size"], 4)


def gen_lbdn():
    base = {"input_size": 5, "hidden_sizes": [8], "output_size": 2, "gamma": 3.0}
    specs = [
        ("relu", base | {"activation": "relu"}),
        ("tanh", base | {"activation": "tanh"}),
        ("deep_sigmoid_nobias", base | {"hidden_sizes": [6, 6], "activation": "sigmoid",
                                         "use_bias": False}),
    ]
    return static_cases("lbdn", specs, lambda c: c["input_size"], 5)


def gen_dyn_unitary():
    specs = [
        ("io3_nx4", {"input_size": 3, "state_size": 4}),
        ("io2_nx6", {"input_size": 2, "state_size": 6}),
    ]
    return ren_cases("dyn_unitary", specs, 6)


_REN = {"input_size": 3, "state_size": 4, "features": 5, "output_size": 2,
        "init_method": "long_memory"}


def gen_contracting_ren():
    specs = [
        ("relu", _REN | {"activation": "relu"}),
        ("tanh", _REN | {"activation": "tanh"}),
        ("tanh_no_polar", _REN | {"activation": "tanh", "do_polar_param": False}),
    ]
    return ren_cases("contracting_ren", specs, 7)


def gen_lipschitz_ren():
    specs = [
        ("relu_gamma2", _REN | {"activation": "relu", "gamma": 2.0}),
        ("tanh_gamma0.5", _REN | {"activation": "tanh", "gamma": 0.5}),
    ]
    return ren_cases("lipschitz_ren", specs, 8)


def gen_general_ren():
    rng = np.random.default_rng(90)
    Q, S, R = make_qsr(rng, 3, 2)
    qsr = {"Q": Q.tolist(), "S": S.tolist(), "R": R.tolist()}
    specs = [
        ("tanh", _REN | {"activation": "tanh"} | qsr),
        ("relu", _REN | {"activation": "relu"} | qsr),
    ]
    return ren_cases("general_ren", specs, 9)


def gen_bilipschitz_ren():
    base = {"input_size": 3, "state_size": 4, "features": 5, "output_size": 3,
            "mu": 0.5, "nu": 4.0, "init_method": "long_memory"}
    specs = [
        ("relu", base | {"activation": "relu"}),
        ("tanh", base | {"activation": "tanh"}),
        ("io1_relu", base | {"input_size": 1, "output_size": 1, "activation": "relu"}),
    ]
    return ren_cases("bilipschitz_ren", specs, 10)


def gen_linear_ren():
    lin = _REN | {"features": 0, "init_method": "long_memory"}
    lin.pop("activation", None)
    rng = np.random.default_rng(110)
    Q, S, R = make_qsr(rng, 3, 2)
    qsr = {"Q": Q.tolist(), "S": S.tolist(), "R": R.tolist()}
    return {
        "contracting_linear_ren": ren_cases("contracting_linear_ren", [("default", lin)], 11),
        "lipschitz_linear_ren": ren_cases("lipschitz_linear_ren", [("gamma2", lin | {"gamma": 2.0})], 12),
        "general_linear_ren": ren_cases("general_linear_ren", [("default", lin | qsr)], 13),
    }


def gen_r2dn():
    base = {"input_size": 3, "state_size": 4, "features": 5, "output_size": 2,
            "init_method": "long_memory"}
    specs = [
        ("hidden2x2", base | {"hidden": [2, 2]}),
        ("hidden4", base | {"hidden": [4]}),
    ]
    return ren_cases("r2dn", specs, 14)


def gen_ren_composition():
    rng = np.random.default_rng(15)
    base = {"input_size": 2, "state_size": 3, "features": 5, "num_layers": 2,
            "mu": 0.5, "nu": 5.0, "init_method": "long_memory", "dyn_state_multiplier": 2}
    specs = [
        ("plain", base),
        ("single_layer_tanh", base | {"num_layers": 1, "activation": "tanh"}),
        ("dyn_in", base | {"dyn_orth_at_input": True}),
        ("dyn_in_out", base | {"dyn_orth_at_input": True, "dyn_orth_at_output": True}),
    ]
    out = []
    for name, cfg in specs:
        nx, nio, L = cfg["state_size"], cfg["input_size"], cfg["num_layers"]
        dyn = cfg["dyn_state_multiplier"] * nx
        inputs = {
            "carry": {
                "rens": _randn(rng, L, BATCH, nx),
                "dyn_in": _randn(rng, BATCH, dyn) if cfg.get("dyn_orth_at_input") else None,
                "dyn_out": _randn(rng, BATCH, dyn) if cfg.get("dyn_orth_at_output") else None,
            },
            "u": _randn(rng, BATCH, nio),
        }
        out.append(_make_case("ren_composition", name, cfg, rng, inputs, _carry_u))
    return out


GENERATORS = {
    "unitary": ("Cayley-parameterised orthogonal layer.", gen_unitary),
    "monlipnet": ("Monotone Lipschitz network.", gen_monlipnet),
    "bilipnet": ("Bi-Lipschitz network (orthogonal + monotone layers).", gen_bilipnet),
    "plnet": ("Polyak-Lojasiewicz network (quadratic potential of a BiLipNet).", gen_plnet),
    "lbdn": ("Lipschitz-bounded deep network (JAX only).", gen_lbdn),
    "dyn_unitary": ("Dynamic (stateful) orthogonal layer.", gen_dyn_unitary),
    "contracting_ren": ("Contracting REN (JAX only).", gen_contracting_ren),
    "lipschitz_ren": ("Lipschitz-bounded REN (JAX only).", gen_lipschitz_ren),
    "general_ren": ("QSR-constrained REN (JAX only).", gen_general_ren),
    "bilipschitz_ren": ("Bi-Lipschitz REN.", gen_bilipschitz_ren),
    "r2dn": ("Contracting R2DN (JAX only).", gen_r2dn),
    "ren_composition": ("Composition of bi-Lipschitz RENs and orthogonal layers.", gen_ren_composition),
}
LINEAR = ("contracting_linear_ren", "lipschitz_linear_ren", "general_linear_ren")


def main(names):
    todo = names or [*GENERATORS, *LINEAR]
    linear = None
    for name in todo:
        if name in LINEAR:
            linear = linear or gen_linear_ren()
            golden.dump(name, f"Linear REN ({name}); JAX only.", linear[name])
        elif name in GENERATORS:
            description, fn = GENERATORS[name]
            golden.dump(name, description, fn())
        else:
            raise SystemExit(f"unknown network '{name}'")
        print(f"wrote tests/golden/{name}.json")


if __name__ == "__main__":
    main(sys.argv[1:])
