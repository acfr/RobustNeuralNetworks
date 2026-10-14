# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Thin adapters that let one test exercise both the JAX and PyTorch models.

A test builds a `Net` through a backend and then only talks to the `Net`:

    net = backend.build("bilipnet", config, inputs)   # config uses the JAX argument names
    out = net.forward(inputs)                          # {"y": ...}  (numpy)
    out = net.explicit(inputs)                         # same, via direct_to_explicit
    net.invert(inputs, out)                            # round trip, e.g. {"x": ...}
    net.gains(n)  /  net.bounds()  /  net.grads_finite(inputs)

`config`, `inputs` and `outputs` use the same layout as the golden files (see
`golden.py`), so the same numbers can be fed to either backend. Anything that
differs between the backends (constructor names, call signatures, parameter
layouts) is hidden in here; keep it that way so the tests stay backend-agnostic.
"""

import numpy as np
import pytest

import golden

# Tolerances shared by every comparison, for both backends.
ATOL = golden.ATOL
RTOL = golden.RTOL

STATIC = ("unitary", "monlipnet", "bilipnet", "plnet", "lbdn")
TORCH_NETWORKS = (
    "unitary", "monlipnet", "bilipnet", "plnet", "dyn_unitary",
    "bilipschitz_ren", "ren_composition",
)


# ---------------------------------------------------------------------------
# Backend-independent helpers
# ---------------------------------------------------------------------------
def input_size(network, config):
    return config["BiLipBlock"]["input_size"] if network == "plnet" else config["input_size"]


def make_inputs(network, config, batch=4, seed=0, zero_state=False):
    """Random numpy inputs in the golden layout (the same for both backends)."""
    rng = np.random.default_rng(seed)

    def randn(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    def state(*shape):
        return np.zeros(shape, np.float32) if zero_state else randn(*shape)

    nin = input_size(network, config)
    if network in STATIC:
        return {"x": randn(batch, nin)}
    if network == "ren_composition":
        nx, layers = config["state_size"], config["num_layers"]
        dyn = config.get("dyn_state_multiplier", 50) * nx
        carry = {
            "rens": state(layers, batch, nx),
            "dyn_in": state(batch, dyn) if config.get("dyn_orth_at_input") else None,
            "dyn_out": state(batch, dyn) if config.get("dyn_orth_at_output") else None,
        }
        return {"carry": carry, "u": randn(batch, nin)}
    return {"state": state(batch, config["state_size"]), "u": randn(batch, nin)}


def make_qsr(nu, ny, seed=0):
    """A valid (Q, S, R) supply-rate triple as numpy arrays: Q < 0 and R - S Q^-1 S^T > 0."""
    rng = np.random.default_rng(seed)
    X, Y, S = (rng.standard_normal(s).astype(np.float32)
               for s in ((ny, ny), (nu, nu), (nu, ny)))
    Q = -X.T @ X
    R = S @ np.linalg.solve(Q, S.T) + Y.T @ Y
    return Q.astype(np.float32), S, R.astype(np.float32)


def _primary(network, inputs):
    return inputs["x"] if network in STATIC else inputs["u"]


class Net:
    """A built model plus its parameters, with a backend-independent interface."""

    def __init__(self, network, config, model, params=None):
        self.network, self.config = network, config
        self.model, self.params = model, params

    # -- implemented per backend ------------------------------------------
    def forward(self, inputs):
        raise NotImplementedError

    def explicit(self, inputs):
        """Outputs via `direct_to_explicit`, or None if there is no such path."""
        raise NotImplementedError

    def invert(self, inputs, outputs, noncausal=False):
        """Recover the model inputs from `outputs`, in the layout of `inputs`.

        `noncausal` (ren_composition only) uses the saved dynamic-orthogonal
        states in `outputs["carry"]` to recover the true input."""
        raise NotImplementedError

    def bounds(self):
        raise NotImplementedError

    def grads_finite(self, inputs):
        raise NotImplementedError

    def check_jit(self, inputs):
        """Compare against a jit-compiled call (only meaningful for JAX)."""

    def orthogonal_matrix(self):
        """The (square) orthogonal matrix of an orthogonal layer."""
        raise NotImplementedError

    def dyn_in_output(self, inputs):
        """Output of the input dynamic-orthogonal layer of a ren_composition,
        which is what the causal `invert` returns in its place."""
        raise NotImplementedError

    # -- shared ---------------------------------------------------------
    def gains(self, n=200, seed=0):
        """Empirical gain |f(a) - f(b)| / |a - b| over `n` random input pairs
        (stateful models start from the zero state)."""
        a = make_inputs(self.network, self.config, n, seed, zero_state=True)
        b = make_inputs(self.network, self.config, n, seed + 1, zero_state=True)
        ya, yb = self.forward(a)["y"], self.forward(b)["y"]
        du = _primary(self.network, a) - _primary(self.network, b)
        return np.linalg.norm(ya - yb, axis=1) / np.linalg.norm(du, axis=1)


class Backend:
    name = None

    def supports(self, network):
        raise NotImplementedError

    def build(self, network, config, inputs, params=None):
        """Build a `Net`. `params` are golden (JAX-layout) parameters; if omitted
        the model keeps its random initialisation."""
        if not self.supports(network):
            pytest.skip(f"'{network}' is not implemented for {self.name}")
        return self._build(network, config, inputs, params)


def check_contract(net, inputs, shapes):
    """Checks every network should pass: output shapes, the explicit path
    agreeing with the forward pass, a jit-compiled call agreeing (JAX only)
    and finite gradients. `shapes` maps output name -> expected shape."""
    out = net.forward(inputs)
    for k, shape in shapes.items():
        assert out[k].shape == shape, k
    exp = net.explicit(inputs)
    if exp is not None:
        golden.assert_close(exp, out, atol=ATOL, rtol=RTOL)
    net.check_jit(inputs)
    assert net.grads_finite(inputs)
    return out


def check_golden(backend, network, case, explicit=False):
    """Check a golden case, via the forward pass or the explicit path."""
    net = backend.build(network, case["config"], case["inputs"], case["params"])
    out = net.explicit(case["inputs"]) if explicit else net.forward(case["inputs"])
    if out is None:
        pytest.skip(f"no explicit path for {backend.name} '{network}'")
    golden.assert_close(out, case["outputs"], atol=ATOL, rtol=RTOL)


def assert_close(actual, expected, **kw):
    kw.setdefault("atol", ATOL)
    kw.setdefault("rtol", RTOL)
    np.testing.assert_allclose(actual, expected, **kw)


# ---------------------------------------------------------------------------
# JAX
# ---------------------------------------------------------------------------
def _jax_carry(c):
    import jax.numpy as jnp

    return {
        "rens": [jnp.asarray(r) for r in c["rens"]],
        "dyn_in": None if c["dyn_in"] is None else jnp.asarray(c["dyn_in"]),
        "dyn_out": None if c["dyn_out"] is None else jnp.asarray(c["dyn_out"]),
    }


class JaxNet(Net):
    def _call(self, params, inputs, explicit=False):
        """Run the model; returns (nested) jax arrays."""
        import jax.numpy as jnp

        n, m = self.network, self.model
        if n in STATIC:
            x = jnp.asarray(inputs["x"])
            if explicit:
                return {"y": m.explicit_call(params, x, m.direct_to_explicit(params))}
            return {"y": m.apply(params, x)}

        if n == "ren_composition":
            carry, u = _jax_carry(inputs["carry"]), jnp.asarray(inputs["u"])
            if explicit:
                new, y = m.explicit_call(params, carry, u, m.direct_to_explicit(params))
            else:
                new, y = m.apply(params, carry, u)
            return {"carry": new, "y": y}

        x, u = jnp.asarray(inputs["state"]), jnp.asarray(inputs["u"])
        if explicit:
            xn, y = m.explicit_call(params, x, u, m.direct_to_explicit(params))
        else:
            xn, y = m.apply(params, x, u)
        return {"next_state": xn, "y": y}

    @staticmethod
    def _numpy(out):
        res = {}
        for k, v in out.items():
            if k == "carry":
                res[k] = {
                    "rens": np.stack([np.asarray(r) for r in v["rens"]]),
                    "dyn_in": None if v["dyn_in"] is None else np.asarray(v["dyn_in"]),
                    "dyn_out": None if v["dyn_out"] is None else np.asarray(v["dyn_out"]),
                }
            else:
                res[k] = np.asarray(v)
        return res

    def forward(self, inputs):
        return self._numpy(self._call(self.params, inputs))

    def explicit(self, inputs):
        return self._numpy(self._call(self.params, inputs, explicit=True))

    def check_jit(self, inputs):
        import jax

        jitted = jax.jit(lambda p: self._call(p, inputs))(self.params)
        golden.assert_close(self._numpy(jitted), self.forward(inputs), atol=ATOL, rtol=RTOL)

    def grads_finite(self, inputs):
        import jax
        import jax.numpy as jnp
        from jax.flatten_util import ravel_pytree

        def loss(p):
            out = self._call(p, inputs)
            return sum(jnp.sum(out[k] ** 2) for k in ("y", "next_state") if k in out)

        flat = ravel_pytree(jax.grad(loss)(self.params))[0]
        return bool(jnp.isfinite(flat).all())

    def bounds(self):
        return tuple(float(b) for b in self.model.get_bounds(self.params)[:2])

    def dyn_in_output(self, inputs):
        e = self.model.direct_to_explicit_inverse(self.params).dyn_in
        carry, u = inputs["carry"]["dyn_in"], inputs["u"]
        return np.asarray(carry @ np.asarray(e.C).T + u @ np.asarray(e.D).T + np.asarray(e.by))

    def orthogonal_matrix(self):
        import jax.numpy as jnp

        e = self.model.direct_to_explicit(self.params)
        if self.network == "unitary":
            return np.asarray(e.R)
        return np.asarray(jnp.block([[e.A, e.B], [e.C, e.D]]))  # dyn_unitary

    def invert(self, inputs, outputs, noncausal=False):
        import jax.numpy as jnp
        import flax.linen as nn

        n, m, p, c = self.network, self.model, self.params, self.config
        y = jnp.asarray(outputs["y"])
        if n == "unitary":
            e = m.direct_to_explicit(p)
            return {"x": np.asarray(m.inverse_call(p, y, e))}
        if n == "monlipnet":
            e = m.direct_to_explicit_inverse(p, alpha=0.1, iterations=500)
            return {"x": np.asarray(m.inverse_call(p, y, e))}
        if n == "bilipnet":
            d = c["depth"]
            e = m.direct_to_explicit_inverse(p, [0.1] * d, [nn.relu] * d, [500] * d, [1.0] * d)
            return {"x": np.asarray(m.inverse_call(p, y, e))}
        if n == "dyn_unitary":
            x_rec, u_rec = m.inverse_call(
                p, jnp.asarray(outputs["next_state"]), y, m.direct_to_explicit(p))
            return {"state": np.asarray(x_rec), "u": np.asarray(u_rec)}
        if n == "bilipschitz_ren":
            e = m.direct_to_explicit_inverse(p)
            _, u = m.inverse_call(p, jnp.asarray(inputs["state"]), y, e)
            return {"u": np.asarray(u)}
        if n == "ren_composition":
            e = m.direct_to_explicit_inverse(p)
            carry = _jax_carry(inputs["carry"])
            if noncausal:
                new_carry = _jax_carry(outputs["carry"])
                _, u = m.inverse_call_noncausal(p, carry, new_carry, y, e)
            else:
                _, u = m.inverse_call(p, carry, y, e)
            return {"u": np.asarray(u)}
        raise NotImplementedError(n)


_ACT = None


def _jax_act():
    global _ACT
    if _ACT is None:
        import flax.linen as nn

        _ACT = {"relu": nn.relu, "tanh": nn.tanh, "sigmoid": nn.sigmoid,
                "identity": lambda x: x}
    return _ACT


def _tup(v):
    return tuple(v) if isinstance(v, (list, tuple)) else v


def build_jax_model(network, config):
    """Construct the JAX module for a golden-file `config` (JAX argument names)."""
    import jax.numpy as jnp
    from robustnn import linear_ren, r2dn_jax, ren_jax
    from robustnn.bilipnet_jax import BiLipNet
    from robustnn.dyn_orthogonal_jax import DynUnitary
    from robustnn.lbdn_jax import LBDN
    from robustnn.monlipnet_jax import MonLipNet
    from robustnn.orthogonal_jax import Unitary
    from robustnn.plnet_jax import PLNet
    from robustnn.ren_composition_jax import CompositionREN

    cfg = dict(config)
    for k in ("activation", "act_fn"):
        if k in cfg:
            cfg[k] = _jax_act()[cfg[k]]
    for k in ("units", "hidden", "hidden_sizes"):
        if k in cfg:
            cfg[k] = _tup(cfg[k])
    for k in ("Q", "S", "R", "optimal_point"):
        if cfg.get(k) is not None:
            cfg[k] = jnp.asarray(cfg[k], dtype=jnp.float32)

    if network == "plnet":
        bilip_cfg = cfg.pop("BiLipBlock")
        return PLNet(BiLipBlock=build_jax_model("bilipnet", bilip_cfg), **cfg)

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


class JaxBackend(Backend):
    name = "jax"

    def supports(self, network):
        return True

    def init_params(self, model, network, inputs, seed=0):
        """Flax `init` with the right call signature for `network`."""
        import jax
        import jax.numpy as jnp

        if network in STATIC:
            args = (jnp.asarray(inputs["x"]),)
        elif network == "ren_composition":
            args = (_jax_carry(inputs["carry"]), jnp.asarray(inputs["u"]))
        else:
            args = (jnp.asarray(inputs["state"]), jnp.asarray(inputs["u"]))
        return model.init(jax.random.key(seed), *args)

    def _build(self, network, config, inputs, params):
        import jax
        import jax.numpy as jnp

        model = build_jax_model(network, config)
        if params is None:
            params = self.init_params(model, network, inputs)
        else:
            params = {"params": jax.tree.map(jnp.asarray, params)}
        return JaxNet(network, config, model, params)


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
def _t(a):
    import torch

    return None if a is None else torch.tensor(np.asarray(a), dtype=torch.float32)


def _n(t):
    return None if t is None else t.detach().numpy()


def _torch_carry(c):
    return {"rens": [_t(r) for r in c["rens"]], "dyn_in": _t(c["dyn_in"]),
            "dyn_out": _t(c["dyn_out"])}


class TorchNet(Net):
    def _forward_raw(self, inputs):
        """Forward pass returning torch tensors (keeps the autograd graph)."""
        n, m = self.network, self.model
        if n in STATIC:
            return {"y": m(_t(inputs["x"]))}
        if n == "ren_composition":
            new, y = m(_torch_carry(inputs["carry"]), _t(inputs["u"]))
            return {"carry": new, "y": y}
        xn, y = m(_t(inputs["state"]), _t(inputs["u"]))
        return {"next_state": xn, "y": y}

    def forward(self, inputs):
        import torch

        with torch.no_grad():
            out = self._forward_raw(inputs)
        res = {}
        for k, v in out.items():
            if k == "carry":
                res[k] = {"rens": np.stack([_n(r) for r in v["rens"]]),
                          "dyn_in": _n(v["dyn_in"]), "dyn_out": _n(v["dyn_out"])}
            else:
                res[k] = _n(v)
        return res

    def explicit(self, inputs):
        import torch

        n, m = self.network, self.model
        with torch.no_grad():
            if n in ("unitary", "monlipnet", "bilipnet", "plnet"):
                y = m.explicit_call(np.asarray(inputs["x"]), m.direct_to_explicit())
                return {"y": np.asarray(y)}
            if n == "bilipschitz_ren":
                xn, y = m.explicit_call(
                    _t(inputs["state"]), _t(inputs["u"]), m.direct_to_explicit())
                return {"next_state": _n(xn), "y": _n(y)}
        return None  # dyn_unitary, ren_composition: no explicit path in torch

    def grads_finite(self, inputs):
        import torch

        self.model.zero_grad()
        out = self._forward_raw(inputs)
        sum(out[k].pow(2).sum() for k in ("y", "next_state") if k in out).backward()
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        return bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)

    def bounds(self):
        return tuple(float(b) for b in self.model.get_bounds()[:2])

    def dyn_in_output(self, inputs):
        import torch

        with torch.no_grad():
            _, out = self.model.dyn_in(_t(inputs["carry"]["dyn_in"]), _t(inputs["u"]))
        return _n(out)

    def orthogonal_matrix(self):
        import torch

        if self.network == "unitary":
            return np.asarray(self.model.direct_to_explicit().Q)
        A, B, C, D = self.model._blocks()  # dyn_unitary
        return torch.cat([torch.cat([A, B], 1), torch.cat([C, D], 1)], 0).detach().numpy()

    def invert(self, inputs, outputs, noncausal=False):
        import torch

        n, m, c = self.network, self.model, self.config
        with torch.no_grad():
            y = _t(outputs["y"])
            if n == "unitary":
                return {"x": np.asarray(m.inverse(y))}
            if n == "monlipnet":
                return {"x": np.asarray(m.inverse(outputs["y"], alpha=0.1, iterations=500))}
            if n == "bilipnet":
                d = c["depth"]
                x = m.inverse(outputs["y"], alphas=[0.1] * d,
                              inverse_activation_fns=[lambda v: np.maximum(0, v)] * d,
                              iterations=[500] * d, Lambdas=[1.0] * d)
                return {"x": np.asarray(x)}
            if n == "dyn_unitary":
                x_rec, u_rec = m.inverse(_t(outputs["next_state"]), y)
                return {"state": _n(x_rec), "u": _n(u_rec)}
            if n == "bilipschitz_ren":
                _, u = m.inverse(_t(inputs["state"]), y)
                return {"u": _n(u)}
            if n == "ren_composition":
                carry = _torch_carry(inputs["carry"])
                if noncausal:
                    _, u = m.inverse_noncausal(carry, _torch_carry(outputs["carry"]), y)
                else:
                    _, u = m.inverse(carry, y)
                return {"u": _n(u)}
        raise NotImplementedError(n)


_MODULE_ACT = None
_FN_ACT = None


def _torch_acts():
    global _MODULE_ACT, _FN_ACT
    if _MODULE_ACT is None:
        import torch
        import torch.nn as nn

        _MODULE_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
                       "identity": nn.Identity}
        _FN_ACT = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid,
                   "identity": lambda x: x}
    return _MODULE_ACT, _FN_ACT


def build_torch_model(network, config):
    """Construct the torch module for a golden-file `config` (JAX argument names)."""
    import torch
    from robustnn.bilipnet_torch import BiLipNet
    from robustnn.dyn_orthogonal_torch import DynUnitary
    from robustnn.monlipnet_torch import MonLipNet
    from robustnn.orthogonal_torch import Unitary
    from robustnn.plnet_torch import PLNet
    from robustnn.ren_composition_torch import CompositionREN
    from robustnn.ren_torch import BiLipschitzREN

    module_act, fn_act = _torch_acts()
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
            kw["act"] = module_act[c["act_fn"]]()
        if network == "bilipnet":
            return BiLipNet(depth=c.get("depth", 2), **kw)
        return MonLipNet(**kw)
    if network == "plnet":
        op = c.get("optimal_point")
        return PLNet(
            BiLipBlock=build_torch_model("bilipnet", c["BiLipBlock"]),
            add_constant=c.get("add_constant", False),
            optimal_point=None if op is None else torch.tensor(op, dtype=torch.float32),
        )
    if network == "dyn_unitary":
        return DynUnitary(c["input_size"], c["state_size"], bias=c.get("use_bias", True))
    if network == "bilipschitz_ren":
        act = module_act[c.get("activation", "relu")]()
        return BiLipschitzREN(c["input_size"], c["state_size"], c["features"],
                              mu=c["mu"], nu=c["nu"], activation=act)
    if network == "ren_composition":
        return CompositionREN(
            c["input_size"], c["state_size"], c["features"], c["num_layers"],
            mu=c["mu"], nu=c["nu"], activation=fn_act[c.get("activation", "relu")],
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


def load_torch_params(network, model, params):
    sd = {k: _t(v) for k, v in to_state_dict(network, params).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k.split(".")[-1] not in _CONFIG_KEYS]
    assert not unexpected, f"unexpected keys: {unexpected}"
    assert not missing, f"keys not set from golden params: {missing}"


class TorchBackend(Backend):
    name = "torch"

    def supports(self, network):
        return network in TORCH_NETWORKS

    def _build(self, network, config, inputs, params):
        model = build_torch_model(network, config)
        if params is not None:
            load_torch_params(network, model, params)
        return TorchNet(network, config, model)


_BACKENDS = {"jax": JaxBackend, "torch": TorchBackend}


def get_backend(name):
    """Backend instance, skipping the test if its libraries are not installed."""
    if name == "jax":
        pytest.importorskip("jax")
        pytest.importorskip("flax")
    else:
        pytest.importorskip("torch")
    return _BACKENDS[name]()
