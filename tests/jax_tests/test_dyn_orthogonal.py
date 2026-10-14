# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `DynUnitary`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

import golden
from jax_utils import check_golden
from robustnn.dyn_orthogonal_jax import DynUnitary


# ------------------------------ DynUnitary --------------------------------
@pytest.mark.parametrize("batches", [1, 5])
@pytest.mark.parametrize("io,nx", [(1, 1), (3, 4), (2, 6)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_dyn_unitary(key, batches, io, nx, use_bias):
    layer = DynUnitary(io, nx, use_bias=use_bias)
    x = jax.random.normal(key, (batches, nx))
    u = jax.random.normal(jax.random.key(1), (batches, io))
    params = layer.init(key, x, u)

    assert layer.initialize_carry(key, (batches, io)).shape == (batches, nx)
    e = layer.direct_to_explicit(params)
    x1, y = layer.explicit_call(params, x, u, e)
    assert x1.shape == (batches, nx) and y.shape == (batches, io)
    x1b, yb = layer.apply(params, x, u)
    np.testing.assert_allclose(x1, x1b, atol=1e-5)
    np.testing.assert_allclose(y, yb, atol=1e-5)

    # Stacked state-space matrix is orthogonal.
    G = jnp.block([[e.A, e.B], [e.C, e.D]])
    np.testing.assert_allclose(G @ G.T, jnp.eye(nx + io), atol=1e-5)

    # Non-causal inverse recovers the previous state and the input.
    x_rec, u_rec = layer.inverse_call(params, x1, y, e)
    np.testing.assert_allclose(x_rec, x, atol=1e-4)
    np.testing.assert_allclose(u_rec, u, atol=1e-4)

    g = jax.grad(lambda p: sum(jnp.sum(a ** 2) for a in layer.apply(p, x, u)))(params)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


@pytest.mark.parametrize("case", golden.cases("dyn_unitary"), ids=golden.case_id)
@pytest.mark.parametrize("explicit", [False, True], ids=["forward", "explicit"])
def test_dyn_unitary_matches_golden(case, explicit):
    check_golden("dyn_unitary", case, explicit)
