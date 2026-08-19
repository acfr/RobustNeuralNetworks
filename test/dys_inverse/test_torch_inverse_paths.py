import numpy as np
import torch

from robustnn.bilipnet_torch import BiLipNet
from robustnn.monlipnet_torch import MonLipNet
from robustnn.pbilipnet_torch import PBiLipNet
from robustnn.plnet_torch import PLNet
from robustnn.pmonlipnet_torch import PMonLipNet
from robustnn.pplnet_torch import PPLNet


def _relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / (1e-8 + np.linalg.norm(expected))


def test_regular_and_conditioned_monlip_round_trips():
    torch.manual_seed(11)
    x = 4.0 * torch.randn(32, 4)

    mon = MonLipNet(
        features=4, unit_features=(7, 6), mu=0.1, tau=20.0,
        is_mu_fixed=True, is_tau_fixed=True,
    ).eval()
    with torch.no_grad():
        for bias in mon.bs:
            bias.normal_(mean=0.5, std=0.25)
    y = mon(x).detach().numpy()
    explicit_y = mon.explicit_call(x.numpy(), mon.direct_to_explicit())
    recovered, residual, _, alpha = mon.inverse_with_diagnostics(y, alpha=1.0)
    round_trip = mon(torch.from_numpy(recovered)).detach().numpy()
    assert _relative_error(explicit_y, y) < 1e-5
    assert _relative_error(round_trip, y) < 1e-4
    assert residual <= 1e-6
    assert alpha < float((mon.mu / (mon.nu - mon.mu)).detach())

    b = torch.randn(32, 13)
    pmon = PMonLipNet(
        features=4, unit_features=(7, 6), mu=0.1, tau=20.0,
        is_mu_fixed=True, is_tau_fixed=True,
    ).eval()
    y = pmon(x, b).detach().numpy()
    explicit_y = pmon.explicit_call(
        x.numpy(), b.numpy(), pmon.direct_to_explicit()
    )
    recovered, residual, _, _ = pmon.inverse_with_diagnostics(
        y, b.numpy(), alpha=1.0
    )
    round_trip = pmon(torch.from_numpy(recovered), b).detach().numpy()
    assert _relative_error(explicit_y, y) < 1e-5
    assert _relative_error(round_trip, y) < 1e-4
    assert residual <= 1e-6


def test_unit_distortion_uses_finite_step():
    torch.manual_seed(14)
    x = torch.randn(16, 3)
    mon = MonLipNet(
        features=3, unit_features=(5, 4), mu=0.1, tau=1.0,
        is_mu_fixed=True, is_tau_fixed=True,
    ).eval()
    y = mon(x).detach().numpy()
    recovered, residual, _, alpha = mon.inverse_with_diagnostics(y)

    assert np.isfinite(alpha)
    assert np.isfinite(residual)
    assert _relative_error(recovered, x.numpy()) < 1e-5


def test_plnet_bilip_core_round_trip_and_known_minimum():
    torch.manual_seed(12)
    x = 3.0 * torch.randn(24, 4)
    x_optimal = torch.tensor([[0.5, -1.0, 0.25, 2.0]])
    block = BiLipNet(
        features=4, unit_features=(7, 6), mu=0.1, tau=25.0,
        is_mu_fixed=True, is_tau_fixed=True, depth=2,
    ).eval()
    with torch.no_grad():
        for layer in block.mon_layers:
            for bias in layer.bs:
                bias.normal_(mean=0.5, std=0.25)
    model = PLNet(block, optimal_point=x_optimal).eval()

    y = block(x).detach().numpy()
    explicit_y = block.explicit_call(x.numpy(), block.direct_to_explicit())
    recovered, residuals, _, _ = block.inverse_with_diagnostics(y)
    round_trip = block(torch.from_numpy(recovered)).detach().numpy()
    potential = model(x).detach().numpy()
    explicit_potential = model.explicit_call(x.numpy(), model.direct_to_explicit())

    assert _relative_error(explicit_y, y) < 1e-5
    assert _relative_error(round_trip, y) < 1e-4
    assert np.all(residuals <= 1e-6)
    assert _relative_error(explicit_potential, potential) < 1e-5
    assert np.max(np.abs(model(x_optimal).detach().numpy())) < 1e-6


def test_pplnet_bilip_core_round_trip_and_known_minimum():
    torch.manual_seed(13)
    x = 3.0 * torch.randn(24, 4)
    p = torch.randn(24, 3)
    x_optimal = torch.tensor([[0.5, -1.0, 0.25, 2.0]])
    block = PBiLipNet(
        features=4,
        unit_features=(7, 6),
        po_units=(8, 4),
        pb_units=(9, 13),
        mu=0.1,
        tau=25.0,
        is_mu_fixed=True,
        is_tau_fixed=True,
        depth=2,
    )
    block(x, p)
    block.eval()
    model = PPLNet(block, optimal_point=x_optimal, c=0.25).eval()

    y = block(x, p).detach().numpy()
    explicit_y = block.explicit_call(
        x.numpy(), p.numpy(), block.direct_to_explicit()
    )
    recovered, residuals, _, _ = block.inverse_with_diagnostics(y, p.numpy())
    round_trip = block(torch.from_numpy(recovered), p).detach().numpy()
    potential = model(x, p).detach().numpy()
    explicit_potential = model.explicit_call(
        x.numpy(), p.numpy(), model.direct_to_explicit()
    )

    assert _relative_error(explicit_y, y) < 1e-5
    assert _relative_error(round_trip, y) < 1e-4
    assert np.all(residuals <= 1e-6)
    assert _relative_error(explicit_potential, potential) < 1e-5
    assert np.max(np.abs(model(x_optimal, p).detach().numpy() - 0.25)) < 1e-6


if __name__ == "__main__":
    test_regular_and_conditioned_monlip_round_trips()
    test_unit_distortion_uses_finite_step()
    test_plnet_bilip_core_round_trip_and_known_minimum()
    test_pplnet_bilip_core_round_trip_and_known_minimum()
    print("All Torch PLNet/PPLNet inverse-path tests passed.")
