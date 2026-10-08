# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""Tests for the JAX `Unitary` (Cayley) layer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from robustnn.orthogonal_jax import Unitary


@pytest.mark.parametrize("batches", [1, 7])
@pytest.mark.parametrize("size", [1, 2, 5])
@pytest.mark.parametrize("use_bias", [True, False])
def test_shape_and_explicit_matches_direct(key, batches, size, use_bias):
    layer = Unitary(input_size=size, use_bias=use_bias)
    x = jax.random.normal(key, (batches, size))
    params = layer.init(key, x)

    y = layer.apply(params, x)
    assert y.shape == (batches, size)

    explicit = layer.direct_to_explicit(params)
    y_exp = layer.explicit_call(params, x, explicit)
    np.testing.assert_allclose(y, y_exp, atol=1e-5)


@pytest.mark.parametrize("size", [2, 5])
def test_orthogonal_and_norm_preserving(key, size):
    layer = Unitary(input_size=size, use_bias=False)
    x = jax.random.normal(key, (16, size))
    params = layer.init(key, x)
    R = layer.direct_to_explicit(params).R
    np.testing.assert_allclose(R @ R.T, jnp.eye(size), atol=1e-5)

    y = layer.apply(params, x)
    np.testing.assert_allclose(jnp.linalg.norm(y, axis=1),
                               jnp.linalg.norm(x, axis=1), rtol=1e-4)


@pytest.mark.parametrize("use_bias", [True, False])
def test_inverse_round_trip(key, use_bias):
    layer = Unitary(input_size=4, use_bias=use_bias)
    x = jax.random.normal(key, (6, 4))
    params = layer.init(key, x)
    # Non-zero bias so that the inverse has to undo it.
    if use_bias:
        params = jax.tree.map(lambda p: p, params)
        params["params"]["b"] = jnp.arange(4, dtype=jnp.float32)
    explicit = layer.direct_to_explicit(params)
    y = layer.explicit_call(params, x, explicit)
    x_rec = layer.inverse_call(params, y, explicit)
    np.testing.assert_allclose(x_rec, x, atol=1e-4)


def test_jit_and_grad(key):
    layer = Unitary(input_size=3)
    x = jax.random.normal(key, (5, 3))
    params = layer.init(key, x)

    np.testing.assert_allclose(jax.jit(layer.apply)(params, x),
                               layer.apply(params, x), atol=1e-5)
    g = jax.grad(lambda p: jnp.sum(layer.apply(p, x) ** 2))(params)
    flat = ravel_pytree(g)[0]
    assert bool(jnp.isfinite(flat).all())
