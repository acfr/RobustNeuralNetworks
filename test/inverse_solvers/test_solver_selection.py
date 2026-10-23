"""Regression tests for selectable MonLip/BiLip inverse solvers.

Run directly with:

    JAX_PLATFORMS=cpu PYTHONPATH=. conda run -n plnet_mjx \
        python test/inverse_solvers/test_solver_selection.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch

from robustnn.bilipnet_jax import BiLipNet as JaxBiLipNet
from robustnn.monlipnet_jax import MonLipNet as JaxMonLipNet
from robustnn.pbilipnet_jax import PBiLipNet as JaxPBiLipNet
from robustnn.pbilipnet_torch import PBiLipNet as TorchPBiLipNet
from robustnn.plnet_jax import PLNet as JaxPLNet
from robustnn.pplnet_jax import PPLNet as JaxPPLNet
from robustnn.pplnet_torch import PPLNet as TorchPPLNet
from robustnn.solvers import normalize_monlip_solver


def _relative_error(actual, expected):
    return np.linalg.norm(np.asarray(actual) - np.asarray(expected)) / (
        1e-8 + np.linalg.norm(np.asarray(expected))
    )


def test_solver_names_and_jax_monlip_round_trips():
    assert normalize_monlip_solver("DavisYinSplit") == "dys"
    assert normalize_monlip_solver("PeacemanRachfordSplit") == "prs"
    assert normalize_monlip_solver("Douglas-Rachford") == "drs"

    x = 3.0 * jax.random.normal(jax.random.key(1), (16, 4))
    model = JaxMonLipNet(
        input_size=4,
        units=(7, 6),
        mu=0.1,
        tau=20.0,
        is_mu_fixed=True,
        is_tau_fixed=True,
    )
    params = model.init(jax.random.key(2), x)
    y = model.apply(params, x)

    try:
        model.direct_to_explicit_inverse(params, solver="prs", alpha=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("PRS must reject a non-positive step.")

    for solver in ("dys", "prs", "drs"):
        explicit = model.direct_to_explicit_inverse(
            params, solver=solver, iterations=2000, tolerance=1e-6
        )
        recovered, residual, _ = model.inverse_call_with_diagnostics(
            params, y, explicit
        )
        recovered_jit = jax.jit(model.inverse_call)(params, y, explicit)
        assert explicit.solver == solver
        assert residual <= explicit.tolerance
        assert _relative_error(model.apply(params, recovered), y) < 2e-5
        assert _relative_error(model.apply(params, recovered_jit), y) < 2e-5


def test_jax_plnet_mixed_solver_core_inverse():
    x = 2.0 * jax.random.normal(jax.random.key(3), (12, 3))
    block = JaxBiLipNet(
        input_size=3,
        units=(5, 4),
        mu=0.2,
        tau=10.0,
        is_mu_fixed=True,
        is_tau_fixed=True,
        depth=2,
    )
    model = JaxPLNet(BiLipBlock=block)
    params = model.init(jax.random.key(4), x)
    y = model.apply(params, x, method=lambda module, value: module.BiLipBlock(value))
    explicit = model.direct_to_explicit_inverse(
        params, solvers=("prs", "drs")
    )
    recovered, residuals, _ = model.inverse_call_with_diagnostics(
        params, y, explicit
    )

    assert [layer.solver for layer in explicit.monlip_layers] == ["prs", "drs"]
    assert bool(jnp.all(residuals <= 1e-6))
    assert _relative_error(recovered, x) < 2e-5


def test_jax_conditioned_plnet_mixed_solver_core_inverse():
    x = 2.0 * jax.random.normal(jax.random.key(5), (12, 3))
    p = jax.random.normal(jax.random.key(6), (12, 2))
    block = JaxPBiLipNet(
        input_size=3,
        units=(5, 4),
        po_units=(6, 3),
        pb_units=(6, 9),
        mu=0.2,
        tau=10.0,
        is_mu_fixed=True,
        is_tau_fixed=True,
        depth=2,
    )
    model = JaxPPLNet(PBiLipBlock=block)
    params = model.init(jax.random.key(7), x, p)
    y = model.apply(
        params,
        x,
        p,
        method=lambda module, value, condition: module.PBiLipBlock(
            value, condition
        ),
    )
    explicit = model.direct_to_explicit_inverse(
        params, solvers=("drs", "prs")
    )
    recovered, residuals, _ = model.inverse_call_with_diagnostics(
        params, y, p, explicit
    )

    assert [layer.solver for layer in explicit.monlip_layers] == ["drs", "prs"]
    assert bool(jnp.all(residuals <= 1e-6))
    assert _relative_error(recovered, x) < 2e-5


def test_torch_conditioned_plnet_core_with_all_solvers():
    torch.manual_seed(5)
    x = 2.0 * torch.randn(12, 3)
    p = torch.randn(12, 2)
    block = TorchPBiLipNet(
        features=3,
        unit_features=(5, 4),
        po_units=(6, 3),
        pb_units=(6, 9),
        mu=0.2,
        tau=10.0,
        is_mu_fixed=True,
        is_tau_fixed=True,
        depth=2,
    )
    block(x, p)
    model = TorchPPLNet(block).eval()
    y = block(x, p).detach().numpy()

    for solvers in (("dys", "dys"), ("prs", "prs"), ("drs", "drs")):
        recovered, residuals, _, effective_alphas = (
            model.inverse_with_diagnostics(y, p.numpy(), solvers=solvers)
        )
        assert np.all(residuals <= 1e-6)
        assert np.all(np.isfinite(effective_alphas))
        assert _relative_error(recovered, x.numpy()) < 2e-5


if __name__ == "__main__":
    test_solver_names_and_jax_monlip_round_trips()
    test_jax_plnet_mixed_solver_core_inverse()
    test_jax_conditioned_plnet_mixed_solver_core_inverse()
    test_torch_conditioned_plnet_core_with_all_solvers()
    print("All selectable inverse-solver tests passed.")
