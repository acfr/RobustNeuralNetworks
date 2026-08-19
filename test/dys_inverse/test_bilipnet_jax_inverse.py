import jax
import jax.numpy as jnp
from flax import linen as nn

from robustnn.bilipnet_jax import BiLipNet


jax.config.update("jax_default_matmul_precision", "highest")


def _model_case():
    key, input_key = jax.random.split(jax.random.key(7))
    inputs = jax.random.normal(input_key, (64, 5))
    model = BiLipNet(
        input_size=5,
        units=(8, 8),
        mu=0.1,
        tau=10.0,
        is_mu_fixed=True,
        is_tau_fixed=True,
        depth=1,
    )
    params = model.init(key, inputs)
    outputs = model.apply(params, inputs)
    return model, params, inputs, outputs


def _relative_error(actual, expected):
    return jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)


def test_default_inverse_converges_at_high_distortion():
    model, params, inputs, outputs = _model_case()

    inverse = model.direct_to_explicit_inverse(params)
    recovered, residuals, steps = model.inverse_call_with_diagnostics(
        params, outputs, inverse
    )
    round_trip = model.apply(params, recovered)

    assert bool(jnp.all(jnp.isfinite(recovered)))
    assert _relative_error(recovered, inputs) < 1e-4
    assert _relative_error(round_trip, outputs) < 1e-4
    assert residuals[0] <= inverse.monlip_layers[0].tolerance
    assert steps[0] <= inverse.monlip_layers[0].iterations


def test_unsafe_requested_alpha_is_capped_to_layer_bound():
    model, params, inputs, outputs = _model_case()

    inverse = model.direct_to_explicit_inverse(
        params,
        alphas=(1.0,),
        inverse_activation_fns=(nn.relu,),
        iterations=(2000,),
        Lambdas=(1.0,),
    )
    layer = inverse.monlip_layers[0]
    convergence_bound = layer.monlip.mu / layer.monlip.gam
    recovered = model.inverse_call(params, outputs, inverse)

    assert 0 < layer.alpha < convergence_bound
    assert bool(jnp.all(jnp.isfinite(recovered)))
    assert _relative_error(recovered, inputs) < 1e-4


def test_inverse_call_remains_jittable():
    model, params, inputs, outputs = _model_case()
    inverse = model.direct_to_explicit_inverse(params)

    recovered = jax.jit(model.inverse_call)(params, outputs, inverse)

    assert _relative_error(recovered, inputs) < 1e-4


def test_default_inverse_converges_across_distortion_and_depth():
    for tau, depth in ((2.0, 2), (50.0, 1), (50.0, 2)):
        key, input_key = jax.random.split(jax.random.key(int(tau) + depth))
        inputs = 5.0 * jax.random.normal(input_key, (32, 5))
        model = BiLipNet(
            input_size=5,
            units=(8, 8),
            mu=0.1,
            tau=tau,
            is_mu_fixed=True,
            is_tau_fixed=True,
            depth=depth,
        )
        params = model.init(key, inputs)
        outputs = model.apply(params, inputs)
        inverse = model.direct_to_explicit_inverse(params)

        recovered, residuals, _ = model.inverse_call_with_diagnostics(
            params, outputs, inverse
        )
        round_trip = model.apply(params, recovered)

        assert bool(jnp.all(jnp.isfinite(recovered)))
        assert _relative_error(round_trip, outputs) < 1e-4
        assert bool(jnp.all(residuals <= 1e-6))


def test_unit_distortion_uses_finite_step():
    key, input_key = jax.random.split(jax.random.key(31))
    inputs = jax.random.normal(input_key, (32, 4))
    model = BiLipNet(
        input_size=4,
        units=(6, 6),
        mu=0.1,
        tau=1.0,
        is_mu_fixed=True,
        is_tau_fixed=True,
        depth=1,
    )
    params = model.init(key, inputs)
    outputs = model.apply(params, inputs)
    inverse = model.direct_to_explicit_inverse(params)
    recovered = model.inverse_call(params, outputs, inverse)

    assert bool(jnp.isfinite(inverse.monlip_layers[0].alpha))
    assert _relative_error(recovered, inputs) < 1e-4


if __name__ == "__main__":
    test_default_inverse_converges_at_high_distortion()
    test_unsafe_requested_alpha_is_capped_to_layer_bound()
    test_inverse_call_remains_jittable()
    test_default_inverse_converges_across_distortion_and_depth()
    test_unit_distortion_uses_finite_step()
    print("All JAX BiLipNet DYS inverse tests passed.")
