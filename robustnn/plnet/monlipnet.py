# This file is a part of the RobustNeuralNetworks package. License is MIT: https://github.com/nic-barbara/RobustNeuralNetworks/blob/main/LICENSE 

'''
Monotone Lipschitz neural network layer 

Adapted from code in 
    "Monotone, Bi-Lipschitz, and Polyak-Łojasiewicz Networks" [https://arxiv.org/html/2402.01344v2]
Author: Dechuan Liu (May 2024)
'''
import jax.numpy as jnp
from flax import linen as nn 
from typing import Any, Sequence, Callable
from flax.typing import Array, PrecisionLike
from robustnn.utils import cayley
from flax.struct import dataclass
from robustnn.solver.DYS import DavisYinSplit

@dataclass
class DirectMonLipParams:
    """
    Data class to keep track of implicit params for Monontone Lipschitz layer.
    Note: mu, nu, and tau are not stored here as they can either be fixed or learned.
    They are calculated in the setup method. 
    One way to access mu, nu, and tau is to call the get_bounds method.
    """
    Fq: Array
    fq: Array
    Fabs: Array
    fabs: Array
    bs: Array
    by: Array
    
@dataclass
class ExplicitMonLipParams:
    """Data class to keep track of explicit params for Monontone Lipschitz layer."""
    mu: float
    nu: float
    gam: float
    units: Sequence[int]
    V: Array
    S: Array
    by: Array
    bh: Array
    sqrt_g2: Array
    sqrt_2g: Array
    STks: Array
    Ak_1s: Array
    BTks: Array
    bs: Array

@dataclass
class ExplicitInverseMonLipParams:
    """Data class to keep track of explicit params for Monontone Lipschitz layer."""
    monlip: ExplicitMonLipParams
    alpha: float # defined for inverse of Monlip layer
    inverse_activation_fn: Callable # inverse activation function
    iterations: int = 200 # number of iterations for the inverse call
    Lambda: float = 1.0 # step size for the update in DYS solver



class MonLipNet(nn.Module):
    '''
    Monotone Lipschitz neural network layer using Cayley transform.
    This layer applies a learned monotone Lipschitz transformation to the input
    using the Cayley map, preserving 2-norms in the transformation process.
    Example usage::

        >>> layer = MonLipNet(units=[4, 4])
        >>> x = jnp.ones((1, 4))
        >>> params = layer.init(jax.random.key(0), x)
        >>> y = layer.apply(params, x)
    Attributes:
        input_size: Size of the input features.
        units: Sequence of integers representing the number of output features for each layer.
        tau: Scaling factor for distortion (default: 10.0).
        mu: Monotone lower bound (default: 0.1).
        nu: Lipschitz upper bound (default: 10.0).
        is_mu_fixed: Whether to fix the value of mu (default: False).
        is_nu_fixed: Whether to fix the value of nu (default: False).
        is_tau_fixed: Whether to fix the value of tau (default: False). Note that you cannot have 
            is_tau_fixed, is_mu_fixed, and is_nu_fixed at the same time.
        act_fn: Activation function to be used in the layer (default: nn.relu).
    '''
    input_size: int
    units: Sequence[int]
    tau: float = 10.
    mu: float = 0.1 # Monotone lower bound
    nu: float = 10.0 # Lipschitz upper bound (nu > mu)
    is_mu_fixed: bool = False
    is_nu_fixed: bool = False
    is_tau_fixed: bool = False
    act_fn: Callable = nn.relu

    def _get_bounds(self):
        """Get the bounds for the MonLipNet layer."""
        if self.is_mu_fixed and self.is_nu_fixed and self.is_tau_fixed:
            raise ValueError("Cannot fix mu, nu, and tau at the same time.")
        elif self.is_mu_fixed and self.is_nu_fixed:
            mu = self.mu
            nu = self.nu
            tau = self.nu / self.mu
        elif self.is_mu_fixed and self.is_tau_fixed:
            mu = self.mu
            nu = self.tau * self.mu
            tau = self.tau
        elif self.is_nu_fixed and self.is_tau_fixed:
            nu = self.nu
            mu = self.nu / self.tau
            tau = self.tau
        elif self.is_mu_fixed:
            mu = self.mu
            log_nu = self.variables['params']['lognu']
            nu = jnp.exp(log_nu)
            tau = nu / mu
        elif self.is_nu_fixed:
            nu = self.nu
            log_mu = self.variables['params']['logmu']
            mu = jnp.exp(log_mu)
            tau = nu / mu
        elif self.is_tau_fixed:
            tau = self.tau
            log_mu = self.variables['params']['logmu']
            mu = jnp.exp(log_mu)
            nu = tau * mu
        else:
            log_mu = self.variables['params']['logmu']
            mu = jnp.exp(log_mu)
            log_nu = self.variables['params']['lognu']
            nu = jnp.exp(log_nu)
            tau = nu / mu
        return mu, nu, tau

    def setup(self):
        """Setup method for the MonLipNet layer."""
        # setup mu, nu, tau
        if self.is_mu_fixed and self.is_nu_fixed and self.is_tau_fixed:
            raise ValueError("Cannot fix mu, nu, and tau at the same time.")
        elif self.is_mu_fixed and self.is_nu_fixed:
            mu = self.mu
            nu = self.nu
            tau = self.nu / self.mu
        elif self.is_mu_fixed and self.is_tau_fixed:
            mu = self.mu
            nu = self.tau * self.mu
            tau = self.tau
        elif self.is_nu_fixed and self.is_tau_fixed:
            nu = self.nu
            mu = self.nu / self.tau
            tau = self.tau
        elif self.is_mu_fixed:
            mu = self.mu
            log_nu = self.param('lognu', nn.initializers.constant(jnp.log(self.nu)), (1,), jnp.float32)
            nu = jnp.exp(log_nu)
            tau = nu / mu
        elif self.is_nu_fixed:
            nu = self.nu
            log_mu = self.param('logmu', nn.initializers.constant(jnp.log(self.mu)), (1,), jnp.float32)
            mu = jnp.exp(log_mu)
            tau = nu / mu
        elif self.is_tau_fixed:
            tau = self.tau
            log_mu = self.param('logmu', nn.initializers.constant(jnp.log(self.mu)), (1,), jnp.float32)
            mu = jnp.exp(log_mu)
            nu = tau * mu
        else:
            log_mu = self.param('logmu', nn.initializers.constant(jnp.log(self.mu)), (1,), jnp.float32)
            mu = jnp.exp(log_mu)
            log_nu = self.param('lognu', nn.initializers.constant(jnp.log(self.nu)), (1,), jnp.float32)
            nu = jnp.exp(log_nu)
            tau = nu / mu

        by = self.param('by', nn.initializers.zeros_init(), (self.input_size,), jnp.float32)
        Fq = self.param('Fq', nn.initializers.glorot_normal(), (self.input_size, sum(self.units)), jnp.float32)
        fq = self.param('fq', nn.initializers.constant(jnp.linalg.norm(Fq)), (1,), jnp.float32)
        Fabs = []
        fabs = []
        bs = []
        nz_1 = 0
        for k, nz in enumerate(self.units):
            Fab = self.param(f'Fab{k}', nn.initializers.glorot_normal(), (nz+nz_1, nz), jnp.float32)
            fab = self.param(f'fab{k}', nn.initializers.constant(jnp.linalg.norm(Fab)), (1,), jnp.float32)
            bs.append(self.param(f'b{k}', nn.initializers.zeros_init(), (nz,), jnp.float32))
            Fabs.append(Fab)
            fabs.append(fab)
            nz_1 = nz 

        self.direct = DirectMonLipParams(Fq=Fq, fq=fq, Fabs=Fabs, fabs=fabs, bs=bs, by=by)

    def _direct_to_explicit(self) -> ExplicitMonLipParams:
        """
        Convert the direct parameters to explicit parameters.
        """
        mu, nu, tau = self._get_bounds()
        gam = nu-mu
        by = self.direct.by
        bs = self.direct.bs
        bh = jnp.concatenate(bs, axis=0)
        QT = cayley((self.direct.fq / jnp.linalg.norm(self.direct.Fq)) * self.direct.Fq)
        Q = QT.T
        sqrt_2g, sqrt_g2 = jnp.sqrt(2. * gam), jnp.sqrt(gam / 2.)

        V, S, bh = [], [], []
        STks, BTks = [], []
        Ak_1s = [jnp.zeros((0, 0))]
        idx, nz_1 = 0, 0
        for k, nz in enumerate(self.units):
            Qk = Q[idx:idx+nz, :] 
            Fab = self.direct.Fabs[k]
            fab = self.direct.fabs[k]
            ABT = cayley((fab / jnp.linalg.norm(Fab)) * Fab)
            ATk, BTk = ABT[:nz, :], ABT[nz:, :]
            QTk_1, QTk = QT[:, idx-nz_1:idx], QT[:, idx:idx+nz]
            STk = QTk @ ATk - QTk_1 @ BTk

            # calculate V and S
            if k > 0:
                Ak, Bk = ATk.T, BTk.T
                V.append(2 * Bk @ ATk_1)
                S.append(Ak @ Qk - Bk @ Qk_1)
            else:
                Ak = ATk.T
                S.append(ABT.T @ Qk)
            ATk_1, Qk_1 = Ak.T, Qk
            
            STks.append(STk)
            BTks.append(BTk)
            Ak_1s.append(ATk.T)
            idx += nz
            nz_1 = nz

        Ak_1s=Ak_1s[:-1]
        S = jnp.concatenate(S, axis=0)

        return ExplicitMonLipParams(
            mu=mu,
            nu=nu,
            gam=gam,
            units=self.units,
            V=V,
            S=S,
            by=by,
            bh=bh,
            sqrt_g2=sqrt_g2,
            sqrt_2g=sqrt_2g,
            STks=STks,
            Ak_1s=Ak_1s,
            BTks=BTks,
            bs=bs,
        )

    def _direct_to_explicit_inverse(self, alpha: float = 1.0,
                            inverse_activation_fn: Callable = nn.relu,
                            iterations: int = 200,
                            Lambda: float = 1.0,
                            ) -> ExplicitInverseMonLipParams:
        """Convert the direct parameters to explicit parameters for inverse call.
        Args:
            alpha: Scaling factor for the explicit parameters in inverse
                MonLipNet layer.
            inverse_activation_fn: Inverse activation function to be used (default: nn.relu).
            iterations: Number of iterations for the inverse call (DYS solver) (default: 200).
            Lambda: Step size for the update in DYS solver (default: 1.0).
        Returns:
            ExplicitInverseMonLipParams: Explicit parameters for inverse call.
        """

        return ExplicitInverseMonLipParams(
            monlip=self._direct_to_explicit(),
            alpha=alpha,
            inverse_activation_fn=inverse_activation_fn,
            iterations=iterations,
            Lambda=Lambda,
        )



    def _explicit_call(self, x: jnp.array, explicit: ExplicitMonLipParams) -> jnp.array:
        """Apply the explicit parameters to the input tensor."""
        # building equation 8 in paper [https://arxiv.org/html/2402.01344v2]
        # y = mu * x + by + sum(sqrt(g/2) * zk @ STk.T)
        y = explicit.mu * x + explicit.by
        zk = x[..., :0]
        for k, nz in enumerate(self.units):
            # zk = act(Vk @ zk + sqrt(2g) * x @ STk + bs)
            zk = self.act_fn(2 * (zk @ explicit.Ak_1s[k]) @ explicit.BTks[k] + explicit.sqrt_2g * x @ explicit.STks[k] + explicit.bs[k])
            y += explicit.sqrt_g2 * zk @ explicit.STks[k].T
        return y
    
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        """Call method for the MonLipNet layer.
        Args:
            x: Input tensor of shape (batch_size, input_dim).
        Returns:
            y: Output tensor of shape (batch_size, output_dim)."""
        explict = self._direct_to_explicit()
        return self._explicit_call(x, explict)
    
    def _explicit_inverse_call(self, y: jnp.array, e_inv: ExplicitInverseMonLipParams) -> Array:
        """
        Inverse call method for the MonLipNet layer using explicit parameters.
        Args:
            y: Output tensor of shape (batch_size, output_dim).
            e: ExplicitInverseMonLipParams object containing explicit inverse parameters.
        Note: The inverse activation function should be the inverse of the activation function used in the forward pass.
        Returns:
            x: Input tensor of shape (batch_size, input_dim).
        """
        e = e_inv.monlip

        # y to b
        # inverse of equation 12
        bz = e.sqrt_2g/e.mu * (y-e.by) @ e.S.T + e.bh
        uk = jnp.zeros(jnp.shape(bz))

        # iterate until converge for zk using DYS solver
        # todo: might change this for loop to jitable loop
        for i in range(e_inv.iterations):
            # iterate until converge for zk using DYS solver
            zk, uk = DavisYinSplit(uk, bz, e, 
                inverse_activation_fn=e_inv.inverse_activation_fn, Lambda=e_inv.Lambda)

        # z to x
        x = (y - e.by - e.sqrt_g2 * zk @ e.S) / e.mu

        return x


    #################### Convenient Wrappers ####################
    def explicit_call(self, params: dict, x: jnp.array, explicit: ExplicitMonLipParams):
        """
        Evaluate the explicit model for the MonLipNet layer.
        Args:
            params (dict): Flax model parameters dictionary.
            x (Array): model inputs.
            explicit (ExplicitMonLipParams): explicit params.
        Returns:
            Array: model outputs.
        """
        return self.apply(params, x, explicit, method="_explicit_call")
    
    def inverse_call(self, params: dict, y: Array, explicit: ExplicitInverseMonLipParams) -> Array:
        """Evaluate the inverse of the explicit model for an MonLipNet layer.
        Args:
            params (dict): Flax model parameters dictionary.
            y (Array): model outputs.
            explicit (ExplicitMonLipParams): explicit params.
        Returns:
            Array: model inputs.
        """
        return self.apply(params, y, explicit, method="_explicit_inverse_call")

    def direct_to_explicit(self, params: dict) -> ExplicitMonLipParams:
        """
        Convert from direct MonLipNet params to explicit form for eval.
        Args:
            params (dict): Flax model parameters dictionary.
        Returns:
            ExplicitMonLipParams: explicit MonLipNet params.
        """
        return self.apply(params, method="_direct_to_explicit")

    def direct_to_explicit_inverse(self, params: dict, alpha: float = 1.0,
                            inverse_activation_fn: Callable = nn.relu,
                            iterations: int = 200,
                            Lambda: float = 1.0) -> ExplicitInverseMonLipParams:
        """
        Convert from direct MonLipNet params to explicit form for inverse call.
        Args:
            params (dict): Flax model parameters dictionary.
            alpha: Scaling factor for the explicit parameters in inverse
                MonLipNet layer.
            inverse_activation_fn: Inverse activation function to be used (default: nn.relu).
            iterations: Number of iterations for the inverse call (DYS solver) (default: 200).
            Lambda: Step size for the update in DYS solver (default: 1.0).
        Returns:
            ExplicitInverseMonLipParams: explicit MonLipNet params.
        """
        return self.apply(params, alpha, inverse_activation_fn, iterations, Lambda, method="_direct_to_explicit_inverse")
    
    def get_bounds(self, params: dict = None) -> tuple:
        """Get the bounds for the MonLipNet layer."""
        return self.apply(params, method="_get_bounds")
    