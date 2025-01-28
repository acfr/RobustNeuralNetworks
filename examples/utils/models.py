import jax
import jax.numpy as jnp

from flax import linen as nn
from flax.linen import initializers as init
from robustrl.networks.utils import ActivationFn, Initializer
from robustrl.networks.typing import Array, Dtype
from typing import Sequence, Tuple

# TODO: Sort out all the dependencies. Just copy-pasting code for now.


class MLP(nn.Module):
  """
  Classic MLP module, edited from the BRAX codebase.
  """
  layer_sizes: Sequence[int]
  activation: ActivationFn = nn.relu
  kernel_init: Initializer = init.lecun_uniform()
  bias: bool = True
  init_output_zero: bool = False

  @nn.compact
  def __call__(self, data: jnp.ndarray):
    hidden = data
    for i, hidden_size in enumerate(self.layer_sizes):
      
      # Choose weight initialiser (bias is 0 on init)
      if i == len(self.layer_sizes) - 1 and self.init_output_zero:
        kinit = init.zeros_init()
      else:
        kinit = self.kernel_init
        
      # Construct the layer, activate if necessary
      hidden = nn.Dense(
          hidden_size,
          name=f'hidden_{i}',
          kernel_init=kinit,
          use_bias=self.bias
          )(hidden)
      if i != len(self.layer_sizes) - 1:
        hidden = self.activation(hidden)
    return hidden



class LSTM(nn.RNNCellBase):
    """
    Classic LSTM module with same interface as RENs.
    """
    features: int
    output_size: int = 0
    activation: ActivationFn = nn.tanh
    kernel_init: Initializer = init.lecun_normal()
    recurrent_kernel_init: Initializer = init.orthogonal()
    carry_init: Initializer = init.zeros_init()
    param_dtype: Dtype = jnp.float32
    init_output_zero: bool = False
    
    @nn.compact
    def __call__(
        self, state: Array, inputs: Array
    ) -> Tuple[Tuple[Array, Array], Array]:
        """
        Call an LSTM.
        
        This implementation treats the LSTM as a dynamical system
        to be evaluated at a single time. The syntax is
            states, out = lstm(states, in)
        There is a linear (dense) layer on the output to re-scale to
        the appropriate size.
        """
        
        # Convert state vector to a tuple of (c,h) for LSTMCell
        state = (state[..., :self.features], state[..., self.features:])
        
        # LSTM layer
        state, out = nn.OptimizedLSTMCell(
            self.features,
            activation_fn=self.activation,
            kernel_init=self.kernel_init,
            recurrent_kernel_init=self.recurrent_kernel_init,
            carry_init=self.carry_init
        )(state, inputs)
        
        # Linear layer to re-size for output
        kinit = init.zeros_init() if self.init_output_zero else self.kernel_init
        out = nn.Dense(
            self.output_size,
            kernel_init=kinit
        )(out)
        
        # Store in an array again and return
        state = jnp.concatenate(state, -1)
        return state, out
    
    @nn.nowrap
    def initialize_carry(
        self, rng: jax.Array, input_shape: Tuple[int, ...]
    ) -> Array:
        """Initialize the LSTM cell state (carry).

        Args:
        rng: random number generator passed to the init_fn.
        input_shape: a tuple providing the shape of the input to the cell.

        Returns:
        An initialized state (carry) vector for the LSTM cell.
        
        This function is slightly modified from flax.linen.recurrent's LSTMCell.
        """
        batch_dims = input_shape[:-1]
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_dims + (self.features,)
        c = self.carry_init(key1, mem_shape, self.param_dtype)
        h = self.carry_init(key2, mem_shape, self.param_dtype)
        return jnp.concatenate((c, h), -1)
