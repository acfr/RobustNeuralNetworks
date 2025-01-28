import jax
from typing import Any, Callable, Sequence
from jax import random, numpy as jnp
import flax
from flax import linen as nn

class MLP(nn.Module):                    # create a Flax Module dataclass
  out_dims: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(128)(x)                 # create inline Flax Module submodules
    x = nn.relu(x)
    x = nn.Dense(self.out_dims)(x)       # shape inference
    return x

model = MLP(out_dims=1)                 # instantiate the MLP model

# x = jnp.ones((1, 32, 1))
x = jax.random.uniform(random.key(42), (1, 32), minval=-1, maxval=1)            # generate random data
variables = model.init(random.key(42), x)# initialize the weights
y = model.apply(variables, x)            # make forward pass

print(y)
seed = 2206
key = jax.random.key(seed)
num_steps = int(1e4)
batch_size = 2**16
running_average = []
for i in range(num_steps):

  keys = jax.random.split(key, 3)
  key = keys[0]
  subkeys = keys[1:]
  x = jax.random.uniform(subkeys[0], (batch_size, 64), minval=-1, maxval=1)
  xd = jax.random.uniform(subkeys[1], (batch_size, 64), minval=-1e-3, maxval=1e-3)            # generate random data            # generate random data
  variables = model.init(subkeys[2], x)# initialize the weights
  y = model.apply(variables, x)
  yd = model.apply(variables, xd)
  # print((yd-y).shape)
  gamma_estimate = jnp.mean(jnp.linalg.norm(yd-y, axis=1) / jnp.linalg.norm(xd-x, axis=1))
  print(gamma_estimate)
  running_average.append(gamma_estimate)

print(jnp.mean(jnp.array(running_average)))

