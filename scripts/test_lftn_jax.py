import jax
from flax import linen
from jax import numpy as jnp
from lbnn.lftn_jax import LFTN

rng = jax.random.PRNGKey(0)
rng2 = jax.random.PRNGKey(1)

nu = 5
ny = 2
nlayers = (8,16,8,ny)

model = LFTN(
    layer_sizes=nlayers,
    activation=linen.tanh,
    gamma=jnp.float32(2.0),
    use_bias=True,
)

test = jax.random.normal(rng2, shape=(4, nu))
params = model.init(rng, jnp.ones((6,nu)))

y = model.apply(params, test)
jax.debug.print("Out {x}", x = y)
