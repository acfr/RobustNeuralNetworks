# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Shared helpers for the PyTorch tests."""

import numpy as np
import pytest
import torch
import torch.nn as nn

import golden


def empirical_gains(f, n, size):
    ua, ub = torch.randn(n, size), torch.randn(n, size)
    with torch.no_grad():
        return (f(ua) - f(ub)).norm(dim=1) / (ua - ub).norm(dim=1)


def grads_finite(model):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(bool(torch.isfinite(g).all()) for g in grads)



def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


# ---------------------------------------------------------------------------
# Golden-file support (see golden.py)
# ---------------------------------------------------------------------------
_MODULE_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "identity": nn.Identity}
_FN_ACT = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid,
           "identity": lambda x: x}


def build_model(network, config):
    """Construct the torch module for a golden-file `config` (JAX argument names)."""
    from robustnn.bilipnet_torch import BiLipNet
    from robustnn.dyn_orthogonal_torch import DynUnitary
    from robustnn.monlipnet_torch import MonLipNet
    from robustnn.orthogonal_torch import Unitary
    from robustnn.plnet_torch import PLNet
    from robustnn.ren_composition_torch import CompositionREN
    from robustnn.ren_torch import BiLipschitzREN

    c = config
    if network == "unitary":
        return Unitary(c["input_size"], c["input_size"], bias=c.get("use_bias", True))
    if network in ("monlipnet", "bilipnet"):
        kw = dict(features=c["input_size"], unit_features=list(c["units"]),
                  mu=c.get("mu"), nu=c.get("nu"), tau=c.get("tau"),
                  is_mu_fixed=c.get("is_mu_fixed", False),
                  is_nu_fixed=c.get("is_nu_fixed", False),
                  is_tau_fixed=c.get("is_tau_fixed", False))
        if "act_fn" in c:
            kw["act"] = _MODULE_ACT[c["act_fn"]]()
        if network == "bilipnet":
            return BiLipNet(depth=c.get("depth", 2), **kw)
        return MonLipNet(**kw)
    if network == "plnet":
        op = c.get("optimal_point")
        return PLNet(
            BiLipBlock=build_model("bilipnet", c["BiLipBlock"]),
            add_constant=c.get("add_constant", False),
            optimal_point=None if op is None else torch.tensor(op, dtype=torch.float32),
        )
    if network == "dyn_unitary":
        return DynUnitary(c["input_size"], c["state_size"], bias=c.get("use_bias", True))
    if network == "bilipschitz_ren":
        act = _MODULE_ACT[c.get("activation", "relu")]()
        return BiLipschitzREN(c["input_size"], c["state_size"], c["features"],
                              mu=c["mu"], nu=c["nu"], activation=act)
    if network == "ren_composition":
        return CompositionREN(
            c["input_size"], c["state_size"], c["features"], c["num_layers"],
            mu=c["mu"], nu=c["nu"], activation=_FN_ACT[c.get("activation", "relu")],
            use_bias=c.get("use_bias", True),
            dyn_orth_at_input=c.get("dyn_orth_at_input", False),
            dyn_orth_at_output=c.get("dyn_orth_at_output", False),
            dyn_state_multiplier=c.get("dyn_state_multiplier", 50),
        )
    raise KeyError(f"no PyTorch implementation of '{network}'")


# --- JAX/Flax parameter layout -> torch state_dict ---------------------------
def _unitary_sd(p, prefix=""):
    sd = {f"{prefix}weight": p["W"], f"{prefix}alpha": p["a"]}
    if "b" in p:
        sd[f"{prefix}bias"] = p["b"]
    return sd


def _monlip_sd(p, prefix=""):
    sd = {f"{prefix}Fq": p["Fq"].T, f"{prefix}fq": p["fq"], f"{prefix}by": p["by"]}
    for k in range(sum(1 for name in p if name.startswith("Fab"))):
        sd[f"{prefix}Fr.{k}"] = p[f"Fab{k}"].T
        sd[f"{prefix}fr.{k}"] = p[f"fab{k}"]
        sd[f"{prefix}bs.{k}"] = p[f"b{k}"]
    return sd


def _bilip_sd(p, prefix=""):
    sd = {}
    for name, sub in p.items():
        kind, k = name.split("_")
        if kind == "uni":
            sd.update(_unitary_sd(sub, f"{prefix}orth_layers.{k}."))
        else:
            sd.update(_monlip_sd(sub, f"{prefix}mon_layers.{k}."))
    return sd


def _flat_sd(p, prefix=""):
    return {f"{prefix}{k}": v for k, v in p.items()}


def _ren_sd(p, prefix=""):
    # The JAX BiLipschitzREN carries a `D22` direct parameter that its QSR
    # parameterisation never reads (D22 is derived from Q, S, R); torch has none.
    return _flat_sd({k: v for k, v in p.items() if k != "D22"}, prefix)


def to_state_dict(network, params):
    """Translate golden (JAX-layout) `params` to a torch `state_dict`."""
    if network == "unitary":
        return _unitary_sd(params)
    if network == "monlipnet":
        return _monlip_sd(params)
    if network == "bilipnet":
        return _bilip_sd(params)
    if network == "plnet":
        sd = _bilip_sd(params["BiLipBlock"], "bln.")
        if "c" in params:
            sd["bias"] = params["c"]
        return sd
    if network == "dyn_unitary":
        return _flat_sd(params)
    if network == "bilipschitz_ren":
        return _ren_sd(params)
    if network == "ren_composition":
        sd = {}
        for name, sub in params.items():
            if name in ("input_orth", "output_orth"):
                sd.update(_unitary_sd(sub, f"{name}."))
            elif name.startswith("orths_"):
                sd.update(_unitary_sd(sub, f"orths.{name.split('_')[1]}."))
            elif name.startswith("rens_"):
                sd.update(_ren_sd(sub, f"rens.{name.split('_')[1]}."))
            else:  # dyn_in / dyn_out
                sd.update(_flat_sd(sub, f"{name}."))
        return sd
    raise KeyError(network)


# Bounds are constructor arguments (buffers/parameters named mu, nu, tau): they
# come from the config, not from the golden parameters.
_CONFIG_KEYS = ("mu", "nu", "tau")


def load_params(network, model, params):
    sd = {k: torch.tensor(np.asarray(v), dtype=torch.float32)
          for k, v in to_state_dict(network, params).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k.split(".")[-1] not in _CONFIG_KEYS]
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not missing, f"keys not set from golden params: {missing}"


def _t(a):
    return None if a is None else torch.tensor(np.asarray(a), dtype=torch.float32)


def _n(t):
    return None if t is None else t.detach().numpy()


def run_model(network, model, inputs, explicit=False):
    """Evaluate `model` on golden `inputs` (forward pass or explicit path).

    Returns None if the explicit path is not available for this network.
    """
    with torch.no_grad():
        if network in ("unitary", "monlipnet", "bilipnet", "plnet"):
            if explicit:
                x = np.asarray(inputs["x"])
                return {"y": np.asarray(model.explicit_call(x, model.direct_to_explicit()))}
            return {"y": _n(model(_t(inputs["x"])))}

        if network == "ren_composition":
            if explicit:
                return None
            c = inputs["carry"]
            carry = {"rens": [_t(r) for r in c["rens"]],
                     "dyn_in": _t(c["dyn_in"]), "dyn_out": _t(c["dyn_out"])}
            new, y = model(carry, _t(inputs["u"]))
            return {"carry": {"rens": np.stack([_n(r) for r in new["rens"]]),
                              "dyn_in": _n(new["dyn_in"]), "dyn_out": _n(new["dyn_out"])},
                    "y": _n(y)}

        x, u = _t(inputs["state"]), _t(inputs["u"])
        if explicit:
            if network != "bilipschitz_ren":
                return None
            xn, y = model.explicit_call(x, u, model.direct_to_explicit())
        else:
            xn, y = model(x, u)
        return {"next_state": _n(xn), "y": _n(y)}


def check_golden(network, case, explicit=False):
    """Check a golden case against the torch model (forward or explicit path)."""
    model = build_model(network, case["config"])
    load_params(network, model, case["params"])
    out = run_model(network, model, case["inputs"], explicit=explicit)
    if out is None:
        pytest.skip(f"no explicit path for torch '{network}'")
    golden.assert_close(out, case["outputs"])
