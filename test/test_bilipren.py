# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE 

'''
Tests for the JAX bi-Lipschitz REN (`BiLipschitzREN`) and its composition
(`CompositionREN`): forward evaluation, inverse round-trip, empirical
bi-Lipschitz bounds, and gradient flow.
'''

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from robustnn import ren_jax as ren
from robustnn.ren_composition_jax import CompositionREN

# Avoid matmul precision discrepancies (see issue #15).
jax.config.update("jax_default_matmul_precision", "highest")


def test_bilipschitz_ren():
    rng = jax.random.key(0)
    rng, k1, k2 = jax.random.split(rng, 3)

    nu_io, nx, nv = 3, 4, 8
    mu, nu = 0.5, 4.0
    model = ren.BiLipschitzREN(nu_io, nx, nv, nu_io, mu=mu, nu=nu,
                               init_method="long_memory")
    model.check_valid_qsr()

    batches = 5
    states = model.initialize_carry(k1, (batches, nu_io))
    inputs = jax.random.normal(k2, (batches, nu_io))
    params = model.init(rng, states, inputs)

    explicit = model.direct_to_explicit(params)
    _, y = model.explicit_call(params, states, inputs, explicit)
    assert y.shape == (batches, nu_io)

    # Inverse round-trip recovers the inputs.
    explicit_inv = model.direct_to_explicit_inverse(params)
    _, u_rec = model.inverse_call(params, states, y, explicit_inv)
    rel_err = jnp.linalg.norm(u_rec - inputs) / jnp.linalg.norm(inputs)
    assert rel_err < 1e-4, f"inverse round-trip error too large: {rel_err}"

    # Empirical gains lie within the bi-Lipschitz bounds.
    rng, ka, kb = jax.random.split(rng, 3)
    ua = jax.random.normal(ka, (300, nu_io))
    ub = jax.random.normal(kb, (300, nu_io))
    z = jnp.zeros((300, nx))
    _, ya = model.explicit_call(params, z, ua, explicit)
    _, yb = model.explicit_call(params, z, ub, explicit)
    gain = jnp.linalg.norm(ya - yb, axis=1) / jnp.linalg.norm(ua - ub, axis=1)
    assert gain.min() >= mu - 1e-3
    assert gain.max() <= nu + 1e-3

    # Gradients are finite.
    def loss(p):
        e = model.direct_to_explicit(p)
        _, yy = model.explicit_call(p, states, inputs, e)
        return jnp.sum(yy ** 2)
    g = jax.grad(loss)(params)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


def test_composition_ren():
    rng = jax.random.key(1)
    rng, k1, k2 = jax.random.split(rng, 3)

    io, nx, nv, L = 2, 4, 8, 3
    mu, nu = 0.5, 5.0
    model = CompositionREN(io, nx, nv, num_layers=L, mu=mu, nu=nu,
                           init_method="long_memory")

    batches = 6
    states = model.initialize_carry(k1, (batches, io))
    inputs = jax.random.normal(k2, (batches, io))
    params = model.init(rng, states, inputs)

    new_states, y = model.apply(params, states, inputs)
    assert len(new_states["rens"]) == L
    assert y.shape == (batches, io)

    # Inverse round-trip.
    e_inv = model.direct_to_explicit_inverse(params)
    _, u_rec = model.inverse_call(params, states, y, e_inv)
    rel_err = jnp.linalg.norm(u_rec - inputs) / jnp.linalg.norm(inputs)
    assert rel_err < 1e-4, f"composition inverse error too large: {rel_err}"

    # Empirical gains within bounds.
    rng, ka, kb = jax.random.split(rng, 3)
    ua = jax.random.normal(ka, (300, io))
    ub = jax.random.normal(kb, (300, io))
    z = {"rens": [jnp.zeros((300, nx)) for _ in range(L)],
         "dyn_in": None, "dyn_out": None}
    e = model.direct_to_explicit(params)
    _, ya = model.explicit_call(params, z, ua, e)
    _, yb = model.explicit_call(params, z, ub, e)
    gain = jnp.linalg.norm(ya - yb, axis=1) / jnp.linalg.norm(ua - ub, axis=1)
    assert gain.min() >= mu - 1e-3
    assert gain.max() <= nu + 1e-3
    assert model.get_bounds(params) == (mu, nu)

    def loss(p):
        _, yy = model.apply(p, states, inputs)
        return jnp.sum(yy ** 2)
    g = jax.grad(loss)(params)
    assert bool(jnp.isfinite(ravel_pytree(g)[0]).all())


def test_composition_ren_dyn_orth():
    """Composition with a dynamic-orthogonal input layer: both inverse modes."""
    rng = jax.random.key(2)
    rng, k1, k2 = jax.random.split(rng, 3)

    io, nx, nv, L = 2, 3, 6, 2
    mu, nu = 0.5, 5.0
    model = CompositionREN(io, nx, nv, num_layers=L, mu=mu, nu=nu,
                           init_method="long_memory",
                           dyn_orth_at_input=True, dyn_state_multiplier=4)

    batches = 5
    carry = model.initialize_carry(k1, (batches, io))
    assert carry["dyn_in"] is not None and carry["dyn_out"] is None
    inputs = jax.random.normal(k2, (batches, io))
    params = model.init(rng, carry, inputs)

    new_carry, y = model.apply(params, carry, inputs)
    assert len(new_carry["rens"]) == L
    assert new_carry["dyn_in"].shape == (batches, 4 * nx)
    assert y.shape == (batches, io)

    e_inv = model.direct_to_explicit_inverse(params)

    # Non-causal inverse (uses the saved dyn-orth next state) recovers the
    # true input exactly.
    _, u_rec = model.inverse_call_noncausal(params, carry, new_carry, y, e_inv)
    rel_err = jnp.linalg.norm(u_rec - inputs) / jnp.linalg.norm(inputs)
    assert rel_err < 1e-4, f"noncausal inverse error too large: {rel_err}"

    # Causal "ignore" inverse recovers the signal at the dyn-orth output (i.e.
    # the dyn-orth forward output), not the true input.
    _, sig_rec = model.inverse_call(params, carry, y, e_inv)
    # The dyn-orth forward output: y_dyn = state @ C.T + inputs @ D.T + by.
    ed = e_inv.dyn_in
    expected = carry["dyn_in"] @ ed.C.T + inputs @ ed.D.T + ed.by
    rel_err2 = jnp.linalg.norm(sig_rec - expected) / jnp.linalg.norm(expected)
    assert rel_err2 < 1e-4, f"ignore inverse mismatch: {rel_err2}"


if __name__ == "__main__":
    test_bilipschitz_ren()
    test_composition_ren()
    test_composition_ren_dyn_orth()
    print("All JAX bi-Lipschitz REN tests passed.")
