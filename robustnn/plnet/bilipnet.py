
import jax.numpy as jnp
from flax import linen as nn 
from typing import Any, Sequence, Callable
from flax.typing import Array, PrecisionLike
from robustnn.utils import cayley
from flax.struct import dataclass
from robustnn.plnet.monlipnet import MonLipNet, ExplicitMonLipParams, DirectMonLipParams
from robustnn.plnet.orthogonal import Unitary, ExplicitOrthogonalParams, DirectOrthogonalParams

@dataclass
class DirectBiLipParams:
    """
    Data class to keep track of direct params for Monontone Lipschitz layer.
    Note: mu, nu, and tau are not stored here as they can either be fixed or learned.
    They are calculated in the setup method. 
    One way to access mu, nu, and tau is to call the get_bounds method.
    """
    monlip_layers: Sequence[DirectMonLipParams]
    unitary_layers: Sequence[DirectOrthogonalParams]
    
@dataclass
class ExplicitBiLipParams:
    """Data class to keep track of explicit params for Monontone Lipschitz layer."""
    monlip_layers: Sequence[ExplicitMonLipParams]
    unitary_layers: Sequence[ExplicitOrthogonalParams]

    # some constant for model properties
    lipmin: float
    lipmax: float
    distortion: float

class BiLipNet(nn.Module):
    """
    BiLipNet is a neural network architecture that combines Unitary and Monotone Lipschitz layers.
    
    Attributes:
        input_size: Size of the input features.
        units: Sequence of integers representing the number of output features for each layer.
        tau: Scaling factor for distortion (default: 10.0).
        mu: Monotone lower bound (default: 0.1).
        nu: Lipschitz upper bound (default: 10.0).
        is_mu_fixed: Whether to fix the value of mu (default: False).
        is_nu_fixed: Whether to fix the value of nu (default: False).
        is_tau_fixed: Whether to fix the value of tau (default: False).
        act_fn: Activation function to be used (default: nn.relu).
        depth: Number of layers in the network (default: 2).
        use_bias: Whether to include a learnable bias term (default: True).
    """
    input_size: int
    units: Sequence[int]
    tau: float = 10.
    mu: float = 0.1 # Monotone lower bound
    nu: float = 10.0 # Lipschitz upper bound (nu > mu)
    is_mu_fixed: bool = False
    is_nu_fixed: bool = False
    is_tau_fixed: bool = False
    act_fn: Callable = nn.relu
    depth: int = 2
    use_bias: bool = True

    def setup(self):
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

        # calculate mu, nu, tau for each layer
        layer_tau = (tau) ** (1/self.depth)
        layer_mu = (mu) ** (1/self.depth)
        layer_nu = (nu) ** (1/self.depth)

        # create layers
        uni, mon = [], []
        for _ in range(self.depth):
            uni.append(Unitary(input_size=self.input_size,
                               use_bias=self.use_bias))
            mon.append(MonLipNet(input_size=self.input_size,
                                 units=self.units, 
                                 tau=layer_tau,
                                 mu=layer_mu,
                                 nu=layer_nu,
                                 is_mu_fixed=self.is_mu_fixed,
                                 is_nu_fixed=self.is_nu_fixed,
                                 is_tau_fixed=self.is_tau_fixed,
                                 act_fn=self.act_fn))
        # append last layer
        uni.append(Unitary(input_size=self.input_size,
                               use_bias=self.use_bias))
        
        self.uni = uni
        self.mon = mon
        self.direct = DirectBiLipParams(monlip_layers=[mon[i].direct for i in range(self.depth)],
                                        unitary_layers=[uni[i].direct for i in range(self.depth+1)])

    def _direct_to_explicit(self) -> ExplicitBiLipParams:
        """Convert direct params to explicit params."""
        monlip_explict_layers = [
            layer._direct_to_explicit() for layer in self.mon
        ]
        unitary_explict_layers = [
            layer._direct_to_explicit() for layer in self.uni
        ]

        # get the bilipnet properties
        lipmin, lipmax, tau = self._get_bounds()
        return ExplicitBiLipParams(monlip_layers=monlip_explict_layers,
                                   unitary_layers=unitary_explict_layers,
                                   lipmin=lipmin,
                                   lipmax=lipmax,
                                   distortion=tau)
    
    def _explicit_call(self, x: jnp.array, explicit: ExplicitBiLipParams) -> Array:
        """Call method for the BiLipNet layer using explicit parameters."""
        for k in range(self.depth):
            x = self.uni[k]._explicit_call( x, explicit.unitary_layers[k])
            x = self.mon[k]._explicit_call( x, explicit.monlip_layers[k])
        x = self.uni[self.depth]._explicit_call( x, explicit.unitary_layers[self.depth])
        return x
    
    @nn.compact
    def __call__(self, x: jnp.array) -> jnp.array:
        """Call method for the BiLipNet layer."""
        explict = self._direct_to_explicit()
        return self._explicit_call( x, explict)
    
    def _get_bounds(self):
        """Get the bounds for the BiLipNet layer."""

        lipmin, lipmax, tau = 1., 1., 1.
        for k in range(self.depth):
            mu, nu, ta = self.mon[k]._get_bounds()
            lipmin *= mu 
            lipmax *= nu 
            tau *= ta 
        return lipmin, lipmax, tau
    
    def get_bounds(self, params: dict = None) -> tuple:
        """Get the bounds for the BiLipNet layer.
        Args:
            params (dict): Flax model parameters dictionary.
        Returns:
            tuple: (lipmin, lipmax, tau)
        """
        return self.apply(params, method="_get_bounds")

    def explicit_call(self, params: dict, x: Array, explicit: ExplicitBiLipParams):
        """Evaluate the explicit model for a BiLipNet layer.
        Args:
            params (dict): Flax model parameters dictionary.
            x (Array): model inputs.
            explicit (ExplicitBiLipParams): explicit params.
        Returns:
            Array: model outputs.
        """
        return self.apply(params, x, explicit, method="_explicit_call")
    
    def direct_to_explicit(self, params: dict) -> ExplicitBiLipParams:
        """Convert from direct BiLipNet params to explicit form for eval.
        Args:
            params (dict): Flax model parameters dictionary.
        Returns:
            ExplicitBiLipParams: explicit BiLipNet layer params.
        """
        return self.apply(params, method="_direct_to_explicit")
    
    # todo: add inverse function for this 
    def inverse(self, params: dict, x: Array, explicit: ExplicitBiLipParams):
        """Evaluate the inverse model for a BiLipNet layer.
        Args:
            params (dict): Flax model parameters dictionary.
            x (Array): model inputs.
            explicit (ExplicitBiLipParams): explicit params.
        Returns:
            Array: model outputs.
        """
        return self.apply(params, x, explicit, method="_inverse_call")
    
    def _inverse_call(self, x: jnp.array, explicit: ExplicitBiLipParams) -> Array:
        """Call method for the BiLipNet layer using explicit parameters."""
        # todo: implement inverse call
        # for k in range(self.depth):
        #     x = self.uni[k]._inverse_call( x, explicit.unitary_layers[k])
        #     x = self.mon[k]._inverse_call( x, explicit.monlip_layers[k])