# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

import jax
import jax.numpy as jnp
import flax.linen as nn

from robustnn.lbdn import LBDN, SandwichLayer

# Need this to avoid matrix multiplication discrepancy (see issue #15)
jax.config.update("jax_default_matmul_precision", "highest")

nu, ny = 5, 2
layers = (8, 16)
gamma = jnp.float32(10)


def _max_jacobian_norm(model, params, x):
    J = jax.vmap(jax.jacfwd(lambda u: model.apply(params, u[None])[0]))(x)
    return jnp.max(jnp.linalg.norm(J, ord=2, axis=(1, 2)))


def test_jit_and_grad():
    model = LBDN(nu, layers, ny, gamma=gamma, activation=nn.tanh)
    inputs = jnp.ones((4, nu))
    params = model.init(jax.random.key(0), inputs)

    # Separate parameter conversion and model call, same as jax.jit(model.apply)
    @jax.jit
    def jit_call(params, inputs):
        explicit = model.direct_to_explicit(params)
        return model.explicit_call(params, inputs, explicit)

    out = jit_call(params, inputs)
    gs = jax.jit(jax.grad(lambda u: jnp.sum(jit_call(params, u)**2)))(inputs)
    assert jnp.allclose(out, model.apply(params, inputs))
    assert bool(jnp.all(jnp.isfinite(gs)))


def test_linear_output_layer_and_lipschitz_bound():
    x = 3 * jax.random.normal(jax.random.key(1), (64, nu))
    for activation in (nn.relu, nn.tanh):
        for init_identity in (True, False):
            model = LBDN(nu, layers, ny, gamma=gamma, activation=activation,
                         init_identity=init_identity)
            params = model.init(jax.random.key(2), x)
            assert "d" not in params["params"]["layers_2"]
            assert _max_jacobian_norm(model, params, x) <= gamma * (1 + 1e-5)

    model = LBDN(nu, layers, ny, gamma=gamma, init_output_zero=True)
    params = model.init(jax.random.key(3), x)
    assert jnp.max(jnp.abs(model.apply(params, x))) == 0


def test_identity_init():
    x = 3 * jax.random.normal(jax.random.key(4), (64, nu))
    identity = lambda u: u

    # Without noise the linear part is exactly I for gamma >= 1 and gamma * I
    # below, with hidden layers wider than the input or no hidden layers at all.
    for g in (0.5, 1.0, 10.0, 1000.0):
        for hidden in (layers, (nu,), ()):
            model = LBDN(nu, hidden, nu, gamma=g, activation=identity,
                         identity_std=0.0)
            params = model.init(jax.random.key(5), x)
            assert jnp.allclose(model.apply(params, x), min(g, 1.0) * x,
                                rtol=1e-5, atol=1e-5)

    model = LBDN(nu, layers, nu, gamma=gamma, identity_std=0.0)
    params = model.init(jax.random.key(6), x)
    assert jnp.allclose(model.apply(params, x), nn.relu(x), atol=1e-5)

    # Every layer is sigma(E^T u), including the gain-scaled first layer
    model = LBDN(nu, layers, nu, gamma=gamma, activation=nn.tanh,
                 identity_std=0.0)
    params = model.init(jax.random.key(7), x)
    assert jnp.allclose(model.apply(params, x), jnp.tanh(jnp.tanh(x)), atol=1e-5)
    assert _max_jacobian_norm(model, params, x) <= gamma * (1 + 1e-5)

    # Only the input matrix carries the gain; the output side stays I / sqrt(2)
    gain = 0.3
    layer = SandwichLayer(nu, 8, activation=nn.tanh, **SandwichLayer.identity_init(gain))
    layer_params = layer.init(jax.random.key(10), x)
    explicit = layer.apply(layer_params, method="_direct_to_explicit")
    assert jnp.allclose(explicit.A_T[:, :nu], jnp.eye(8, nu) / jnp.sqrt(2), atol=1e-6)
    assert jnp.allclose(explicit.B, gain * jnp.eye(8, nu) / jnp.sqrt(2), atol=1e-6)
    assert jnp.allclose(layer.apply(layer_params, x),
                        jnp.tanh(gain * x @ jnp.eye(nu, 8)), atol=1e-5)

    # Output size differs: the identity is truncated to the first ny outputs
    model = LBDN(nu, layers, ny, gamma=gamma, activation=identity,
                 identity_std=0.0)
    params = model.init(jax.random.key(8), x)
    assert jnp.allclose(model.apply(params, x), x[:, :ny], atol=1e-5)

    # The default init is the noisy identity: close to x, with distinct padded units
    model = LBDN(nu, layers, nu, gamma=gamma, activation=identity)
    params = model.init(jax.random.key(9), x)
    y = model.apply(params, x)
    assert jnp.linalg.norm(y - x) / jnp.linalg.norm(x) < 0.1
    padded = params["params"]["layers_0"]["XY"][8:, nu:]
    assert jnp.min(jnp.linalg.norm(padded[:, :-1] - padded[:, 1:], axis=0)) > 0


if __name__ == "__main__":
    test_jit_and_grad()
    test_linear_output_layer_and_lipschitz_bound()
    test_identity_init()
    print("All LBDN tests passed.")
