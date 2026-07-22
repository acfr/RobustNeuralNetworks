# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/acfr/RobustNeuralNetworks/blob/main/LICENSE

'''
Define the Peaceman-Rachford splitting solver
Two-operator splitting applied to MonLip network inversion.
Author: Yurui Zhang
'''
import flax.linen as nn
from typing import Callable, Tuple
from flax.typing import Array

from robustnn.solver_DYS import mln_RA


def PeacemanRachford(uk, bz, e,
        inverse_activation_fn: Callable = nn.relu,
        Lambda: float = 1.0, alpha: float = 1.0) -> Tuple[Array, Array]:
    """
    Peaceman-Rachford splitting solver for MonLip networks.
    Args:
        uk (Array): Current value of u.
        bz (Array): Current value of b.
        e (ExplicitMonLipParams): ExplicitMonLipParams object containing the network parameters.
        inverse_activation_fn (Callable, optional): Inverse activation function. Defaults to nn.relu.
        Lambda (float, optional): Relaxation parameter; Lambda=1.0 is the standard PR update. Defaults to 1.0.
        alpha (float, optional): Step size parameter. Defaults to 1.0.
    Returns:
        Update once (zk+1, uk+1).
    """
    # prox_f(uk)
    zh = inverse_activation_fn(uk)
    # reflection: 2*prox_f(uk) - uk
    uh = 2*zh - uk
    # resolvent of second operator applied to reflected point
    zk = mln_RA(e.gam, e.mu, e.S, e.V, alpha, bz, zh, uh, e.units)
    # PR update: Lambda=1 gives standard 2*J_A(R_B(uk)) - R_B(uk)
    uk = uk + Lambda * (2*zk - uh - uk)

    return zk, uk
