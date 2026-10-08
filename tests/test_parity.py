# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

"""JAX vs PyTorch parity: build the torch layer, copy its *explicit*
parameters into the JAX layer, and check both give the same outputs.

Initialisations differ between the two backends, so parameters have to be
copied across explicitly. Only the layers with a simple explicit
parameterisation are covered; MonLipNet / BiLipNet / REN parity is a TODO
(their explicit parameter layouts differ between backends).
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")

from robustnn.dyn_orthogonal_jax import DynUnitary as JaxDynUnitary
from robustnn.dyn_orthogonal_jax import ExplicitDynOrthogonalParams
from robustnn.dyn_orthogonal_torch import DynUnitary as TorchDynUnitary
from robustnn.orthogonal_jax import ExplicitOrthogonalParams, Unitary as JaxUnitary
from robustnn.orthogonal_torch import Unitary as TorchUnitary


@pytest.mark.parametrize("size", [2, 5])
def test_unitary_parity(size):
    torch.manual_seed(0)
    t_layer = TorchUnitary(size, size, bias=True)
    with torch.no_grad():
        t_layer.bias.copy_(torch.randn(size))
    x = torch.randn(6, size)
    y_torch = t_layer(x).detach().numpy()

    e = t_layer.direct_to_explicit()
    j_layer = JaxUnitary(input_size=size, use_bias=True)
    params = j_layer.init(jax.random.key(0), jnp.asarray(x.numpy()))
    j_explicit = ExplicitOrthogonalParams(R=jnp.asarray(e.Q), b=jnp.asarray(e.b))
    y_jax = j_layer.explicit_call(params, jnp.asarray(x.numpy()), j_explicit)
    np.testing.assert_allclose(np.asarray(y_jax), y_torch, atol=1e-5)


@pytest.mark.parametrize("io,nx", [(2, 3), (3, 6)])
def test_dyn_unitary_parity(io, nx):
    torch.manual_seed(0)
    t_layer = TorchDynUnitary(io, nx)
    x, u = torch.randn(5, nx), torch.randn(5, io)
    with torch.no_grad():
        x1_t, y_t = t_layer(x, u)
        A, B, C, D = [m.numpy() for m in t_layer._blocks()]

    j_layer = JaxDynUnitary(io, nx)
    jx, ju = jnp.asarray(x.numpy()), jnp.asarray(u.numpy())
    params = j_layer.init(jax.random.key(0), jx, ju)
    e = ExplicitDynOrthogonalParams(
        jnp.asarray(A), jnp.asarray(B), jnp.asarray(C), jnp.asarray(D),
        jnp.asarray(t_layer.bx.detach().numpy()),
        jnp.asarray(t_layer.by.detach().numpy()))
    x1_j, y_j = j_layer.explicit_call(params, jx, ju, e)
    np.testing.assert_allclose(np.asarray(x1_j), x1_t.numpy(), atol=1e-5)
    np.testing.assert_allclose(np.asarray(y_j), y_t.numpy(), atol=1e-5)
