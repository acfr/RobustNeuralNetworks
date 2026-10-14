# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Shared helpers for the JAX tests."""

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from jax.flatten_util import ravel_pytree

import golden


ACTIVATIONS = [nn.relu, nn.tanh, lambda x: x]


def setup_ren(model, key, batches, nin, x_scale=1.0):
    k1, k2, k3 = jax.random.split(key, 3)
    states = model.initialize_carry(k1, (batches, nin))
    states = x_scale * jax.random.normal(k2, states.shape)
    inputs = jax.random.normal(k3, (batches, nin))
    params = model.init(key, states, inputs)
    return params, states, inputs


def make_qsr(key, nu, ny):
    kx, ky, ks = jax.random.split(key, 3)
    X = jax.random.normal(kx, (ny, ny))
    Y = jax.random.normal(ky, (nu, nu))
    S = jax.random.normal(ks, (nu, ny))
    Q = -X.T @ X
    R = S @ jnp.linalg.solve(Q, S.T) + Y.T @ Y
    return Q, S, R


def check_ren_forward(model, params, states, inputs, nx, ny):
    batches = inputs.shape[0]
    explicit = model.direct_to_explicit(params)
    xn, y = model.explicit_call(params, states, inputs, explicit)
    assert xn.shape == (batches, nx)
    assert y.shape == (batches, ny)
    xn2, y2 = model.apply(params, states, inputs)
    np.testing.assert_allclose(xn, xn2, atol=1e-5)
    np.testing.assert_allclose(y, y2, atol=1e-5)

    # jit gives the same answer, and gradients w.r.t. everything are finite
    jit_xn, jit_y = jax.jit(model.apply)(params, states, inputs)
    np.testing.assert_allclose(y, jit_y, atol=1e-4)

    def loss(p, x, u):
        a, b = model.apply(p, x, u)
        return jnp.sum(a ** 2) + jnp.sum(b ** 2)
    g = jax.grad(loss, argnums=(0, 1, 2))(params, states, inputs)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


def empirical_gains(f, key, n, size):
    """Empirical gain |f(a) - f(b)| / |a - b| over `n` random input pairs."""
    ka, kb = jax.random.split(key)
    ua = jax.random.normal(ka, (n, size))
    ub = jax.random.normal(kb, (n, size))
    return jnp.linalg.norm(f(ua) - f(ub), axis=1) / jnp.linalg.norm(ua - ub, axis=1)


# ---------------------------------------------------------------------------
# Golden-file support (see golden.py)
# ---------------------------------------------------------------------------

_ACT = {"relu": nn.relu, "tanh": nn.tanh, "sigmoid": nn.sigmoid, "identity": lambda x: x}


def _tup(v):
    return tuple(v) if isinstance(v, (list, tuple)) else v


def build_model(network, config):
    """Construct the JAX module for a golden-file `config` (JAX argument names)."""
    from robustnn import linear_ren, r2dn_jax
    from robustnn import ren_jax
    from robustnn.bilipnet_jax import BiLipNet
    from robustnn.dyn_orthogonal_jax import DynUnitary
    from robustnn.lbdn_jax import LBDN
    from robustnn.monlipnet_jax import MonLipNet
    from robustnn.orthogonal_jax import Unitary
    from robustnn.plnet_jax import PLNet
    from robustnn.ren_composition_jax import CompositionREN

    cfg = dict(config)
    if "activation" in cfg:
        cfg["activation"] = _ACT[cfg["activation"]]
    if "act_fn" in cfg:
        cfg["act_fn"] = _ACT[cfg["act_fn"]]
    for k in ("units", "hidden", "hidden_sizes"):
        if k in cfg:
            cfg[k] = _tup(cfg[k])
    for k in ("Q", "S", "R", "optimal_point"):
        if cfg.get(k) is not None:
            cfg[k] = jnp.asarray(cfg[k], dtype=jnp.float32)

    if network == "plnet":
        bilip_cfg = cfg.pop("BiLipBlock")
        return PLNet(BiLipBlock=build_model("bilipnet", bilip_cfg), **cfg)

    cls = {
        "unitary": Unitary,
        "monlipnet": MonLipNet,
        "bilipnet": BiLipNet,
        "lbdn": LBDN,
        "dyn_unitary": DynUnitary,
        "contracting_ren": ren_jax.ContractingREN,
        "lipschitz_ren": ren_jax.LipschitzREN,
        "general_ren": ren_jax.GeneralREN,
        "bilipschitz_ren": ren_jax.BiLipschitzREN,
        "contracting_linear_ren": linear_ren.ContractingLinREN,
        "lipschitz_linear_ren": linear_ren.LipschitzLinREN,
        "general_linear_ren": linear_ren.GeneralLinREN,
        "r2dn": r2dn_jax.ContractingR2DN,
        "ren_composition": CompositionREN,
    }[network]
    return cls(**cfg)


def _carry_in(c):
    return {
        "rens": [jnp.asarray(r) for r in c["rens"]],
        "dyn_in": None if c["dyn_in"] is None else jnp.asarray(c["dyn_in"]),
        "dyn_out": None if c["dyn_out"] is None else jnp.asarray(c["dyn_out"]),
    }


def _carry_out(c):
    return {
        "rens": np.stack([np.asarray(r) for r in c["rens"]]),
        "dyn_in": None if c["dyn_in"] is None else np.asarray(c["dyn_in"]),
        "dyn_out": None if c["dyn_out"] is None else np.asarray(c["dyn_out"]),
    }


_STATIC_X = ("unitary", "monlipnet", "bilipnet", "plnet", "lbdn")


def run_model(network, model, params, inputs, explicit=False):
    """Evaluate `model` on golden `inputs`, via `apply` or the explicit path."""
    if network in _STATIC_X:
        x = jnp.asarray(inputs["x"])
        if explicit:
            e = model.direct_to_explicit(params)
            y = model.explicit_call(params, x, e)
        else:
            y = model.apply(params, x)
        return {"y": np.asarray(y)}

    if network == "ren_composition":
        carry, u = _carry_in(inputs["carry"]), jnp.asarray(inputs["u"])
        if explicit:
            e = model.direct_to_explicit(params)
            new_carry, y = model.explicit_call(params, carry, u, e)
        else:
            new_carry, y = model.apply(params, carry, u)
        return {"carry": _carry_out(new_carry), "y": np.asarray(y)}

    # dyn_unitary and all the REN flavours: (state, u) -> (next_state, y)
    x, u = jnp.asarray(inputs["state"]), jnp.asarray(inputs["u"])
    if explicit:
        e = model.direct_to_explicit(params)
        xn, y = model.explicit_call(params, x, u, e)
    else:
        xn, y = model.apply(params, x, u)
    return {"next_state": np.asarray(xn), "y": np.asarray(y)}


def check_golden(network, case, explicit=False):
    """Check a golden case, via `apply` or `direct_to_explicit` + `explicit_call`."""
    model = build_model(network, case["config"])
    params = {"params": jax.tree.map(jnp.asarray, case["params"])}
    out = run_model(network, model, params, case["inputs"], explicit=explicit)
    golden.assert_close(out, case["outputs"])
