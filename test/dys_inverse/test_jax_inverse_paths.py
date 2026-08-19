import jax
import jax.numpy as jnp

from robustnn.bilipnet_jax import BiLipNet
from robustnn.monlipnet_jax import MonLipNet
from robustnn.pbilipnet_jax import PBiLipNet
from robustnn.plnet_jax import PLNet
from robustnn.pmonlipnet_jax import PMonLipNet
from robustnn.pplnet_jax import PPLNet


jax.config.update("jax_default_matmul_precision", "highest")


def _relative_error(actual, expected):
    return jnp.linalg.norm(actual - expected) / (1e-8 + jnp.linalg.norm(expected))


def test_regular_and_conditioned_monlip_round_trips():
    x = 4.0 * jax.random.normal(jax.random.key(1), (32, 4))

    mon = MonLipNet(
        input_size=4, units=(7, 6), mu=0.1, tau=20.0,
        is_mu_fixed=True, is_tau_fixed=True,
    )
    variables = mon.init(jax.random.key(2), x)
    y = mon.apply(variables, x)
    inverse = mon.direct_to_explicit_inverse(variables, alpha=1.0)
    recovered, residual, _ = mon.inverse_call_with_diagnostics(
        variables, y, inverse
    )
    assert _relative_error(mon.apply(variables, recovered), y) < 1e-4
    assert residual <= inverse.tolerance
    assert inverse.alpha < inverse.monlip.mu / inverse.monlip.gam

    b = jax.random.normal(jax.random.key(3), (32, 13))
    pmon = PMonLipNet(
        input_size=4, units=(7, 6), mu=0.1, tau=20.0,
        is_mu_fixed=True, is_tau_fixed=True,
    )
    variables = pmon.init(jax.random.key(4), x, b)
    y = pmon.apply(variables, x, b)
    inverse = pmon.direct_to_explicit_inverse(variables, alpha=1.0)
    recovered, residual, _ = pmon.inverse_call_with_diagnostics(
        variables, y, b, inverse
    )
    assert _relative_error(pmon.apply(variables, recovered, b), y) < 1e-4
    assert residual <= inverse.tolerance


def test_plnet_bilip_core_round_trip_and_known_minimum():
    x = 3.0 * jax.random.normal(jax.random.key(5), (24, 4))
    x_optimal = jnp.asarray([[0.5, -1.0, 0.25, 2.0]])
    block = BiLipNet(
        input_size=4, units=(7, 6), mu=0.1, tau=25.0,
        is_mu_fixed=True, is_tau_fixed=True, depth=2,
    )
    model = PLNet(BiLipBlock=block, optimal_point=x_optimal, c=0.25)
    variables = model.init(jax.random.key(6), x)

    y = model.apply(
        variables, x,
        method=lambda module, value: module.BiLipBlock(value),
    )
    inverse = model.apply(
        variables,
        method=lambda module: module.BiLipBlock._direct_to_explicit_inverse(),
    )
    recovered, residuals, _ = model.apply(
        variables,
        y,
        inverse,
        method=lambda module, value, explicit: (
            module.BiLipBlock._explicit_inverse_call_with_diagnostics(value, explicit)
        ),
    )
    round_trip = model.apply(
        variables, recovered,
        method=lambda module, value: module.BiLipBlock(value),
    )
    minimum = model.apply(variables, x_optimal)

    assert _relative_error(round_trip, y) < 1e-4
    assert bool(jnp.all(residuals <= 1e-6))
    assert jnp.max(jnp.abs(minimum - 0.25)) < 1e-6


def test_pplnet_bilip_core_round_trip_and_known_minimum():
    x = 3.0 * jax.random.normal(jax.random.key(7), (24, 4))
    p = jax.random.normal(jax.random.key(8), (24, 3))
    x_optimal = jnp.asarray([[0.5, -1.0, 0.25, 2.0]])
    block = PBiLipNet(
        input_size=4,
        units=(7, 6),
        po_units=(8, 4),
        pb_units=(9, 13),
        mu=0.1,
        tau=25.0,
        is_mu_fixed=True,
        is_tau_fixed=True,
        depth=2,
    )
    model = PPLNet(PBiLipBlock=block, optimal_point=x_optimal, c=0.25)
    variables = model.init(jax.random.key(9), x, p)

    y = model.apply(
        variables, x, p,
        method=lambda module, value, condition: module.PBiLipBlock(value, condition),
    )
    inverse = model.apply(
        variables,
        method=lambda module: module.PBiLipBlock._direct_to_explicit_inverse(),
    )
    recovered, residuals, _ = model.apply(
        variables,
        y,
        p,
        inverse,
        method=lambda module, value, condition, explicit: (
            module.PBiLipBlock._explicit_inverse_call_with_diagnostics(
                value, condition, explicit
            )
        ),
    )
    round_trip = model.apply(
        variables, recovered, p,
        method=lambda module, value, condition: module.PBiLipBlock(value, condition),
    )
    minimum = model.apply(variables, x_optimal, p)

    assert _relative_error(round_trip, y) < 1e-4
    assert bool(jnp.all(residuals <= 1e-6))
    assert jnp.max(jnp.abs(minimum - 0.25)) < 1e-6


if __name__ == "__main__":
    test_regular_and_conditioned_monlip_round_trips()
    test_plnet_bilip_core_round_trip_and_known_minimum()
    test_pplnet_bilip_core_round_trip_and_known_minimum()
    print("All JAX PLNet/PPLNet inverse-path tests passed.")
